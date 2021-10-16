import os
import random
import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import hashlib

import dwd_dl.cfg as cfg
import dwd_dl.utils as utils


def timestamps_with_nans_handler(nan_days, max_nans, in_channels, out_channels):
    not_for_mean = []
    to_remove = []
    nan_to_num = []
    for date in nan_days:
        if nan_days[date] > max_nans:
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
        self.classes_h5_file_handle = H5Dataset(
            ranges, mode='c', classes=cfg.CFG.CLASSES,
        )
        self.raw_h5_file_handle = H5Dataset(
            ranges, mode='r', classes=cfg.CFG.CLASSES,
        )
        self.normalize = normalize
        print("reading images...")
        self._image_size = image_size
        tot_pre = {}
        nan_days = {}
        std = {}
        mean = {}
        classes_frequency = {}
        for date in tqdm(timestamps_list):
            tot_pre[date] = self.classes_h5_file_handle[date].attrs['tot_pre']
            nan_days[date] = self.classes_h5_file_handle[date].attrs['NaN']
            std[date] = self.classes_h5_file_handle[date].attrs['std']
            mean[date] = self.classes_h5_file_handle[date].attrs['mean']
            classes_frequency[date] = self.classes_h5_file_handle[date].attrs['classes_frequency']

        self._tot_pre = tot_pre
        self.classes_frequency = classes_frequency
        self.sequence = sorted(timestamps_list)
        self.sorted_sequence = sorted(self.sequence)
        self._sequence_timestamps = sorted(self.sequence)

        to_remove, not_for_mean, nan_to_num = timestamps_with_nans_handler(
            nan_days, max_nans, in_channels, out_channels
        )

        to_remove = timestamps_at_training_period_end_handler(
            ranges, to_remove, in_channels, out_channels
        )

        self.to_remove = to_remove
        self.not_for_mean = not_for_mean
        self.nan_to_num = nan_to_num

        print("Normalizing... Actually just computing incremental std and mean")
        self.mean = 0
        self.std = 0
        for n, date in enumerate(self.sequence):
            n_in_img = self._image_size ** 2
            if date not in not_for_mean:
                m = mean[date]
                s = std[date]
                self.std = utils.incremental_std(self.std, self.mean, n, n_in_img, s, m)
                self.mean = utils.incremental_mean(self.mean, n, n_in_img, m)

        print("done loading dataset")

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
def create_h5(mode: str, classes=None, keep_open=True, height=256, width=256, verbose=False):

    if mode not in ('r', 'c'):
        raise ValueError(f"Need either 'r' or 'c' in mode but got {mode}")
    default_classes = {'0': (0, 0.1), '0.1': (0.1, 1), '1': (1, 2.5), '2.5': (2.5, np.infty)}
    histogram_bins = [n * 0.1 for n in range(501)]
    histogram_bins.append(np.infty)
    if classes is None:
        classes = default_classes

    to_create = check_h5_missing_or_corrupt(cfg.CFG.date_ranges, classes=classes)
    video_h5_to_create = check_h5_missing_or_corrupt(cfg.CFG.video_ranges, classes=classes)
    for h5_file in video_h5_to_create:
        if h5_file not in to_create:
            to_create.append(h5_file)

    if to_create:
        for m in ('c', 'r'):
            for ranges in (cfg.CFG.date_ranges, cfg.CFG.video_ranges):
                for year_month, file_name in zip(
                        utils.ym_tuples(ranges),
                        h5_files_names_list_single_mode(ranges, mode=m),
                ):
                    if file_name in to_create:
                        with h5py.File(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name), 'a') as f:
                            date_range = cfg.MonthDateRange(*year_month).date_range()
                            data = np.empty((len(date_range), height * width))
                            data[...] = np.nan
                            for n, date in tqdm(enumerate(date_range)):
                                date_str = date.strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
                                if n == 0:
                                    first_date_str = date_str
                                if verbose:
                                    print('Processing {}'.format(date_str))
                                if date_str not in f.keys():
                                    binary_file_name = cfg.binary_file_name(time_stamp=date)  # TODO: refactor. Not correct for deviations
                                    try:
                                        raw_data = utils.square_select(date, height=height, width=width, plot=False).data
                                    except OverflowError:
                                        # If not found then treat it as NaN-filled
                                        raw_data = np.empty((height, width))
                                        raw_data[:] = np.nan
                                    # tot_nans = np.count_nonzero(np.isnan(raw_data))
                                    # raw_data = np.nan_to_num(raw_data)
                                    # min_ = np.min(raw_data)
                                    # max_ = np.max(raw_data)
                                    # class_frequency = dict()
                                    # histogram_frequency, histogram_edges = np.histogram(raw_data, bins=histogram_bins)
                                    if m == 'c':  # skipped if in raw mode
                                        raw_data = np.nan_to_num(raw_data)
                                        raw_data = np.array(
                                            [(classes[class_name][0] <= raw_data) &
                                             (raw_data < classes[class_name][1]) for class_name in classes]
                                        ).astype(int)
                                        class_frequency = {
                                            class_name: np.count_nonzero(raw_data[i]) for i, class_name in enumerate(classes)
                                        }
                                        raw_data = utils.to_class_index(raw_data)
                                    # raw_data = np.expand_dims(raw_data, 0)
                                    # data[n] = raw_data.ravel()
                                    # adding dimension for timestamp and coordinates
                                    # normalized_time_of_day = utils.normalized_time_of_day_from_string(
                                    #     timestamp_string=date_str
                                    # )
                                    # dim for concat
                                    # timestamps_grid = np.full(
                                    #     (1, height, width), normalized_time_of_day, dtype=np.float32
                                    # )
                                    # coordinates_array = cfg.CFG.coordinates_array
                                    # data = np.concatenate((raw_data, timestamps_grid, coordinates_array))

                                    f.create_dataset(date_str, data=raw_data.ravel())
                                    # f[date_str] = raw_data
                                    # f[date_str].attrs['classes_frequency'] = np.array(list(class_frequency.values()))
                                    # f[date_str].attrs['histogram_frequency'] = histogram_frequency
                                    # f[date_str].attrs['histogram_edges'] = histogram_edges
                                    # f[date_str].attrs['file_name'] = binary_file_name
                                    # f[date_str].attrs['NaN'] = tot_nans
                                    # f[date_str].attrs['img_size'] = height * width
                                    # f[date_str].attrs['tot_pre'] = np.nansum(raw_data)
                                    # f[date_str].attrs['mean'] = np.nanmean(raw_data)
                                    # f[date_str].attrs['std'] = np.nanstd(raw_data)
                                    # f[date_str].attrs['min'] = min_
                                    # f[date_str].attrs['max'] = max_
                            # f.create_dataset(first_date_str, data=data.T)
                            # f['mode'] = [0] if m == 'c' else [1]
                            # f[hashlib.md5(file_name.encode()).hexdigest()] = np.array([1])


def read_h5(filename):
    if not filename.endswith('.h5'):
        filename += '.h5'

    f = h5py.File(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), filename), 'a')

    return f


@contextmanager
def h5_handler(*args, **kwargs):

    file = create_h5(**kwargs)

    try:
        yield file
    finally:
        file.close()


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
    file_name = '-'.join([f"v{version_}b", f"{mode}", f"{year}{month:03}"])
    if with_extension:
        extension = '.h5'
        file_name += extension
    return file_name


def h5_files_names_list_single_mode(date_ranges, **kwargs):
    year_month_tuples = utils.ym_tuples(date_ranges)
    return [h5_name(*ym, **kwargs) for ym in year_month_tuples]


def h5_files_names_list_both_modes(date_ranges, **kwargs):
    try:
        kwargs.pop('mode')
    except KeyError:
        pass
    return h5_files_names_list_single_mode(
        date_ranges, mode='c', **kwargs
    ) + h5_files_names_list_single_mode(
        date_ranges, mode='r', **kwargs
    )


def ym_dictionary(date_ranges, **kwargs):
    dict_ = {}
    for ym, file_name in zip(utils.ym_tuples(date_ranges), h5_files_names_list_single_mode(date_ranges, **kwargs)):
        dict_[ym] = file_name
    return dict_


def check_h5_missing_or_corrupt(date_ranges, **kwargs):
    unavailable = []
    required = h5_files_names_list_both_modes(date_ranges, **kwargs)
    for file_name in required:
        if not os.path.isfile(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name)):
            unavailable.append(file_name)
        else:
            with h5py.File(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name), mode='a') as f:
                try:
                    hash_check = True # f[hashlib.md5(file_name.encode()).hexdigest()]
                    assert hash_check
                except (KeyError, AssertionError):
                    print(f"h5 file name not corresponding to content for file {file_name}")
                    hash_check = False

            if not hash_check:
                unavailable.append(file_name)
                os.remove(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name))

    return unavailable


class H5Dataset:
    def __init__(self, date_ranges, mode, classes=None):
        self.mode = mode
        self._keys = []
        self.classes = classes
        self._date_ranges = date_ranges
        self.year_month_file_names_dictionary = ym_dictionary(date_ranges, mode=mode)
        self.files_dictionary = {
            ym: h5py.File(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name), 'r')
            for (ym, file_name) in self.year_month_file_names_dictionary.items()
        }

    def __getitem__(self, item):
        if item == 'mode':
            return self.mode
        elif item == 'classes':
            return self.classes
        timestamp = dt.datetime.strptime(item, cfg.CFG.TIMESTAMP_DATE_FORMAT)
        return self.files_dictionary[(timestamp.year, timestamp.month)][item]

    def close(self):
        for ym in self.files_dictionary:
            self.files_dictionary[ym].close()

    def keys(self):
        if not self._keys:
            for ym in self.files_dictionary:
                self._keys.extend(self.files_dictionary[ym].keys())
        return self._keys

    def __iter__(self):
        def string_is_digit(element):
            if not isinstance(element, str):
                return False
            return element.isdigit()

        return filter(string_is_digit, self.keys())


@contextmanager
def h5dataset_context_wrapper(*args, **kwargs):
    f = H5Dataset(*args, **kwargs)

    try:
        yield f
    finally:
        f.close()
