import os
import random
import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import Dataset
from tqdm import tqdm
import hashlib

import dwd_dl.cfg as cfg
import dwd_dl.utils as utils


def timestamps_with_nans_handler(nan_days, max_nans, in_channels, out_channels):
    not_for_mean = []
    to_remove = []
    nan_to_num = []
    for date, row in nan_days.iterrows():
        if row.values[0] > max_nans:
            time_stamp = dt.datetime.strptime(date, '%y%m%d%H%M')
            dr = cfg.daterange(
                time_stamp - dt.timedelta(hours=in_channels + out_channels - 1),
                time_stamp,
                include_end=True
            )
            not_for_mean.append(time_stamp.strftime('%y%m%d%H%M'))
            for to_remove_ts in dr:
                if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                    to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))
        else:
            nan_to_num.append(date)

    return to_remove, not_for_mean, nan_to_num


def timestamps_at_training_period_end_handler(ranges_list, to_remove, in_channels, out_channels):
    for date_range in ranges_list:
        range_start, range_end = date_range
        try:
            if range_end + dt.timedelta(hours=1) in ranges_list[ranges_list.index(date_range) + 1]:
                continue
        except IndexError:
            pass
        dr = cfg.daterange(
            range_end - dt.timedelta(hours=in_channels + out_channels - 2),
            range_end,
            include_end=True
        )

        for to_remove_ts in dr:
            if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))

    return to_remove


class RadolanDataset(Dataset):
    """DWD Radolan for nowcasting"""

    in_channels = 6
    out_channels = 1

    def __init__(
        self,
        image_size=256,
        in_channels=in_channels,
        out_channels=out_channels,
        verbose=False,
        normalize=False,
        max_nans=10,
        min_weights_factor_of_max=0.0001,
        video_or_normal='normal',
    ):

        assert video_or_normal in ('normal', 'video')
        if video_or_normal == 'normal':
            timestamps_list = cfg.CFG.date_timestamps_list
            print("Using normal mode for dataset.")
            ranges = cfg.CFG.date_ranges
        else:
            timestamps_list = cfg.CFG.video_timestamps_list
            ranges = cfg.CFG.video_ranges
        self.min_weights_factor_of_max = min_weights_factor_of_max
        # read radolan files
        self.ds_classes = H5Dataset(
            ranges, mode='c', classes=cfg.CFG.CLASSES,
        )
        self.ds_raw = H5Dataset(
            ranges, mode='r', classes=cfg.CFG.CLASSES,
        )
        self.normalize = normalize
        print("reading images...")
        self._image_size = image_size

        self._tot_pre = self.ds_raw.ds.sum(dim=["lon", "lat"], skipna=True)
        self.nan_days = self.ds_raw.ds.isnull().sum(dim=["lon", "lat"])
        self.sequence = sorted(ranges)
        self.sorted_sequence = sorted(ranges)
        self._sequence_timestamps = sorted(ranges)

        to_remove, not_for_mean, nan_to_num = timestamps_with_nans_handler(
            nan_days, max_nans, in_channels, out_channels
        )

        to_remove = timestamps_at_training_period_end_handler(
            ranges, to_remove, in_channels, out_channels
        )

        self.to_remove = to_remove
        self.not_for_mean = not_for_mean
        self.nan_to_num = nan_to_num

        # create global index for sequence and true_rainfall (idx -> ([s_idxs], [t_idx]))
        self.indices_tuple_raw = [
            [
                [i + n for n in range(in_channels)],
                [i + in_channels + n for n in range(out_channels)]
            ] for i, _ in enumerate(self.sorted_sequence)
        ]

        self.indices_tuple = [
            self.indices_tuple_raw[i] for i, key in enumerate(self.sorted_sequence) if key not in self.to_remove
        ]

        self._list_of_firsts = [row[0][0] for row in self.indices_tuple]

    @property
    def list_of_firsts(self):
        return self._list_of_firsts

    @property
    def weights(self):
        w = []
        print('Computing weights.')
        for i in tqdm(range(self.__len__())):
            seq, tru = self.get_total_pre(i)

            seq, tru = np.array(seq), np.array(tru)
            w.append(np.sum(seq) + np.sum(tru))
        w = torch.tensor(w)
        # Add a constant. Otherwise torch.multinomial complains TODO: is this really true?
        return w + self.min_weights_factor_of_max*torch.max(w)

    def get_total_pre(self, idx):
        seq, tru = self.indices_tuple[idx]
        tot = {'seq': [], 'tru': []}
        for sub_period, indices in zip(tot, (seq, tru)):
            for t in indices:
                tot[sub_period].append(self._tot_pre[self.sorted_sequence[t]])

        return tot['seq'], tot['tru']

    def get_total_pre_time_series_with_timestamps(self) -> pd.Series:
        time_series_dict = dict()
        for i in range(len(self)):
            seq, tru = self.indices_tuple[i]
            timestamp = dt.datetime.strptime(self.sorted_sequence[seq[0]], cfg.CFG.TIMESTAMP_DATE_FORMAT)
            time_series_dict[timestamp] = 0
            for indices in (seq, tru):
                for t in indices:
                    time_series_dict[timestamp] += self._tot_pre[self.sorted_sequence[t]]
        return pd.Series(time_series_dict)

    def get_nonzero_intervals(self):
        precipitation_series = self.get_total_pre_time_series_with_timestamps()
        open_ = False
        start = end = None
        interval_list = []
        for i, (index, value) in enumerate(precipitation_series.iteritems()):
            if value == 0 and not open_:
                continue
            elif value > 0 and not open_:
                start = index
                end = index
                open_ = True
                continue
            elif value > 0 and open_:
                if not index - end == dt.timedelta(hours=1):
                    interval = pd.Interval(start, end, closed='both')
                    interval_list.append(interval)
                    open_ = False
                    continue
                end = index
                continue
            elif value == 0 and open_:
                interval = pd.Interval(start, end, closed='both')
                open_ = False
                interval_list.append(interval)
                continue
        return pd.Series(interval_list)

    def get_nonzero_timestamps(self):
        precipitation_series = self.get_total_pre_time_series_with_timestamps()
        return precipitation_series[precipitation_series > 0]

    def __len__(self):
        return len(self.indices_tuple)

    def __getitem__(self, idx):
        seq, tru = self.indices_tuple[idx]
        item_tensors = {'seq': [], 'tru': []}
        for sub_period, indices in zip(item_tensors, (seq, tru)):
            for t in indices:
                if sub_period == 'seq' and cfg.CFG.MODE == 'r':
                    data = self.raw_h5_file_handle[self.sorted_sequence[t]]
                else:
                    data = self.classes_h5_file_handle[self.sorted_sequence[t]]
                item_tensors[sub_period].append(data)
            item_tensors[sub_period] = torch.from_numpy(
                np.concatenate(item_tensors[sub_period], axis=0).astype(np.float32)
            )

        return item_tensors['seq'], item_tensors['tru']

    def indices_of_training(self):
        indices_list_of_training = []
        training_set_timestamps_list = cfg.CFG.training_set_timestamps_list
        for idx, _ in tqdm(enumerate(self.indices_tuple)):
            seq, tru = self.indices_tuple[idx]
            if self.sorted_sequence[seq[0]] in training_set_timestamps_list:
                indices_list_of_training.append(idx)
        return indices_list_of_training

    def indices_of_validation(self):
        indices_list_of_validation = []
        validation_set_timestamps_list = cfg.CFG.validation_set_timestamps_list
        for idx, _ in enumerate(self.indices_tuple):
            seq, tru = self.indices_tuple[idx]
            if self.sorted_sequence[seq[0]] in validation_set_timestamps_list:
                indices_list_of_validation.append(idx)
        return indices_list_of_validation

    def indices_of_test(self):
        indices_list_of_test = []
        test_set_timestamps_list = cfg.CFG.test_set_timestamps_list
        for idx, _ in enumerate(self.indices_tuple):
            seq, tru = self.indices_tuple[idx]
            if self.sorted_sequence[seq[0]] in test_set_timestamps_list:
                indices_list_of_test.append(idx)
        return indices_list_of_test

    def from_timestamp(self, timestamp):
        try:
            index_in_timestamps_list = self.sorted_sequence.index(timestamp)
            true_index = self._list_of_firsts.index(index_in_timestamps_list)
            item = self.__getitem__(true_index)
        except ValueError:
            raise ValueError('The timestamp {} could not be reached'.format(timestamp))

        return item

    def has_timestamp(self, timestamp):
        try:
            self.from_timestamp(timestamp)
            return True
        except ValueError:
            return False

    def close(self):
        for f in (self.raw_h5_file_handle, self.classes_h5_file_handle):
            f.close()


class RadolanSubset(RadolanDataset):
    """Radolan Dataset Subset, training, validation or testing

    """

    def __init__(self, dataset: RadolanDataset, subset, valid_cases=20, seed=42, random_=False):
        assert subset in ['train', 'valid', 'all', 'test']
        dataset_len = len(dataset)
        indices = [i for i in range(dataset_len)]
        if random_ and not subset == 'test':
            random.seed(seed)
            indices = sorted(dataset.indices_of_validation() + dataset.indices_of_training())
            valid_indices = random.sample(indices, k=round((valid_cases / 100) * len(indices)))
            if subset == "valid":
                indices = [indices[i] for i in sorted(valid_indices)]
            else:
                indices = [x for x in indices if indices.index(x) not in valid_indices]
        elif subset == 'train':
            indices = dataset.indices_of_training()
        elif subset == 'valid':
            indices = dataset.indices_of_validation()
        elif subset == 'test':
            indices = dataset.indices_of_test()
        self.dataset = dataset
        self.indices = indices
        self.timestamps = [x for n, x in enumerate(self.dataset.sorted_sequence) if n in self.dataset.list_of_firsts]
        self.timestamps = [self.timestamps[self.indices[n]] for n, _ in enumerate(self.indices)]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def from_timestamp(self, timestamp):
        try:
            timestamp_raw_idx = self.dataset.sorted_sequence.index(timestamp)
            index = self.dataset.list_of_firsts.index(timestamp_raw_idx)
            idx = self.indices.index(index)
            item = self.__getitem__(idx)
        except ValueError:
            raise ValueError('The timestamp {} could not be reached'.format(timestamp))

        return item

    @property
    def weights(self):
        ds_weights = self.dataset.weights
        w = [ds_weights[i] for i in self.indices]
        return torch.tensor(w)

    def get_total_pre(self, idx):
        return self.dataset.get_total_pre[self.indices[idx]]


@utils.init_safety
def create_h5(mode: str, classes=None, h5_or_ncdf='N', keep_open=True, height=256, width=256, verbose=False):

    if mode not in ('r', 'c'):
        raise ValueError(f"Need either 'r' or 'c' in mode but got {mode}")
    default_classes = {'0': (0, 0.1), '0.1': (0.1, 1), '1': (1, 2.5), '2.5': (2.5, np.infty)}
    histogram_bins = [n * 0.1 for n in range(501)]
    histogram_bins.append(np.infty)
    if classes is None:
        classes = default_classes

    to_create = check_datasets_missing(cfg.CFG.date_ranges, classes=classes)
    video_h5_to_create = check_datasets_missing(cfg.CFG.video_ranges, classes=classes)
    for h5_file in video_h5_to_create:
        if h5_file not in to_create:
            to_create.append(h5_file)

    if to_create:
        for m in ('c', 'r'):
            for ranges in (cfg.CFG.date_ranges, cfg.CFG.video_ranges):
                for year_month, file_name in zip(
                        utils.ym_tuples(ranges),
                        files_names_list_single_mode(ranges, h5_or_ncdf=h5_or_ncdf, mode=m),
                ):
                    if file_name in to_create:
                        date_range = cfg.MonthDateRange(*year_month).date_range()
                        data = np.empty(shape=(len(date_range), height, width))
                        time = np.array(date_range)
                        for n, date in tqdm(enumerate(date_range)):

                            try:
                                raw_data = utils.square_select(date, height=height, width=width, plot=False).data
                            except OverflowError:
                                # If not found then treat it as NaN-filled
                                raw_data = np.empty((height, width))
                                raw_data[:] = np.nan

                            if m == 'c':  # skipped if in raw mode
                                raw_data = np.array(
                                    [(classes[class_name][0] <= raw_data) &
                                     (raw_data < classes[class_name][1]) for class_name in classes]
                                ).astype(int)
                                raw_data = utils.to_class_index(raw_data)

                            data[n] = raw_data

                        xds = xarray.Dataset(
                            data_vars={
                                'precipitation': (['time', 'lon', 'lat'], data)
                            }, coords={
                                'time': time
                            })

                        xds.to_netcdf(
                            path=os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name),
                            engine='h5netcdf'
                        )


def h5_name(year: int, month: int, version_=None, mode=None, classes=None, with_extension=True):
    if version_ is None:
        version_ = cfg.CFG.CURRENT_H5_VERSION
    if mode is None:
        mode = cfg.CFG.MODE
    elif mode not in ('r', 'c'):
        raise ValueError(f"Mode not 'r' or 'c'. Got {mode}")

    if mode == 'c':
        if classes is None:
            classes = cfg.CFG.CLASSES
        elif not isinstance(classes, dict):
            raise TypeError(f"Check classes! Got {classes}")
        mode_classes = [classes[class_name][0] for class_name in classes]
        mode_classes_strings = [f"{float(c):05.3}" for c in mode_classes]
        mode = ''.join([mode] + [f"{len(mode_classes_strings)}"] + mode_classes_strings)
        # This creates a univocal naming. r is raw. c is classes. If classes mode then the numbers following it are:
        # first number is the number of classes. The next are the left limits of the classes with :05.3 format
        # e.g. v0.0.2-r-201203
        # or v0.0.2-c4000.0000.1001.0002.5-2012004 means 4 classes starting at 0 0.1 1 and 2.5
    file_name = '-'.join([f"v{version_}", f"{mode}", f"{year}{month:03}"])
    if with_extension:
        extension = '.h5'
        file_name += extension
    return file_name


def ncdf4_name(*args, **kwargs):
    h5 = h5_name(*args, **kwargs)
    if h5.endswith('.h5'):
        h5 = h5.replace('.h5', '.nc')
    return h5


def files_names_list_single_mode(date_ranges, h5_or_ncdf='N', **kwargs):
    if h5_or_ncdf == 'N':
        name_func = ncdf4_name
    elif h5_or_ncdf == 'H':
        name_func = h5_name
    else:
        raise ValueError(f'Got an unexpected value for h5_or_ncdf={h5_or_ncdf}. Value must be "N" or "H"')
    year_month_tuples = utils.ym_tuples(date_ranges)
    return [name_func(*ym, **kwargs) for ym in year_month_tuples]


def files_names_list_both_modes(date_ranges, h5_or_ncdf='N', **kwargs):
    try:
        kwargs.pop('mode')
    except KeyError:
        pass
    return files_names_list_single_mode(
        date_ranges, h5_or_ncdf=h5_or_ncdf, mode='c', **kwargs
    ) + files_names_list_single_mode(
        date_ranges, h5_or_ncdf=h5_or_ncdf, mode='r', **kwargs
    )


def check_datasets_missing(date_ranges,  h5_or_ncdf='N', **kwargs):
    unavailable = []
    required = files_names_list_both_modes(date_ranges, h5_or_ncdf=h5_or_ncdf, **kwargs)
    for file_name in required:
        if not os.path.isfile(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name)):
            unavailable.append(file_name)
    return unavailable


class H5Dataset:
    def __init__(self, date_ranges, mode, classes=None):
        self.mode = mode
        self.classes = classes
        self._date_ranges = date_ranges
        self._files_list = files_names_list_single_mode(self._date_ranges, h5_or_ncdf='N', mode=mode)
        self._files_paths = [os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), fn) for fn in self._files_list]
        self.ds = xarray.open_mfdataset(paths=self._files_paths, chunks={'time': -1, 'lon': 256, 'lat': 256},
                                        engine='h5netcdf')

    def __getitem__(self, item):
        if item == 'mode':
            return self.mode
        elif item == 'classes':
            return self.classes
        elif item.isdigit():
            timestamp = dt.datetime.strptime(item, cfg.CFG.TIMESTAMP_DATE_FORMAT)
            timestamps_grid = np.full(
                (1, cfg.CFG.HEIGHT, cfg.CFG.WIDTH),
                utils.normalized_time_of_day_from_string(item),
                dtype=np.float32
            )
            precipitation = self.ds.precipitation.loc[timestamp]
            coordinates_array = cfg.CFG.coordinates_array
            return np.concatenate((np.expand_dims(precipitation, 0), timestamps_grid, coordinates_array))
        else:
            raise KeyError(f"{item} is an invalid Key for this H5Dataset")

    def __iter__(self):
        return iter(self.ds.precpitation)


@contextmanager
def h5dataset_context_wrapper(*args, **kwargs):
    f = H5Dataset(*args, **kwargs)

    try:
        yield f
    finally:
        f.close()
