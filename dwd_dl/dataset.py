import os
import random
import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import zarr

from dwd_dl import log
import dwd_dl.cfg as cfg
import dwd_dl.utils as utils


def timestamps_with_nans_handler(nan_days, max_nans, in_channels, out_channels):
    not_for_mean = []
    to_remove = []
    nan_to_num = []
    nan_days_computed = nan_days
    for timestamp in tqdm(nan_days_computed):
        datetime_timestamp = pd.Timestamp(timestamp.time.values).to_pydatetime()
        if timestamp.values > max_nans:
            dr = cfg.daterange(
                datetime_timestamp - dt.timedelta(hours=in_channels + out_channels - 1),
                datetime_timestamp,
                include_end=True
            )
            not_for_mean.append(datetime_timestamp)
            for to_remove_ts in dr:
                if to_remove_ts not in to_remove:
                    to_remove.append(to_remove_ts)
        else:
            nan_to_num.append(datetime_timestamp)

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
            if to_remove_ts not in to_remove:
                to_remove.append(to_remove_ts)

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
        normalize=False,
        max_nans=10,
        min_weights_factor_of_max=0.0001,
        video_or_normal='normal',
        threshold=None,
        mode=None,
        use_dask=True,
    ):

        log.info("Initializing %s", self.__class__.__name__)
        assert video_or_normal in ('normal', 'video')
        if video_or_normal == 'normal':
            log.info("Using normal mode for dataset.")
            timestamps_list = cfg.CFG.date_timestamps_list
            ranges = cfg.CFG.date_ranges
        else:
            timestamps_list = cfg.CFG.video_timestamps_list
            ranges = cfg.CFG.video_ranges
        self.min_weights_factor_of_max = min_weights_factor_of_max

        # read radolan files
        log.info("Instantiating %s for classes mode.", H5Dataset.__name__)
        self.ds_classes = H5Dataset(
            ranges, mode='c', classes=cfg.CFG.CLASSES,
        )
        log.info("Done instantiating.")
        log.info("Instantiating %s for raw mode.", H5Dataset.__name__)
        self.ds_raw = H5Dataset(
            ranges, mode='r', classes=cfg.CFG.CLASSES,
        )
        log.info("Done instantiating")

        self.normalize = normalize
        self._image_size = image_size

        if not mode == 'vis':
            log.info("Computing total precipitation.")
            self._tot_pre = self.ds_raw.ds.sum(dim=["lon", "lat"], skipna=True).precipitation.compute()

        if threshold is not None:
            log.info("Finding timestamps where threshold is exceeded.")
            selection = (self.ds_raw.ds.precipitation > threshold).any(dim=('lon', 'lat')).compute()
            possible_timestamps = self.ds_raw.ds.time[selection]
            possible_timestamps = possible_timestamps.compute()
            self._timestamps_over_threshold = [pd.Timestamp(x.time.values).to_pydatetime() for x in possible_timestamps]

        log.info("Computing how many nans per day.")
        self.nan_days = self.ds_raw.ds.isnull().sum(dim=["lon", "lat"]).precipitation.compute()

        self.sequence = sorted(timestamps_list)
        self.sorted_sequence = sorted(timestamps_list)
        self._sequence_timestamps = sorted(timestamps_list)

        log.info("Computing which timestamps not to consider for computations given their nan percentage.")
        to_remove, not_for_mean, nan_to_num = timestamps_with_nans_handler(
            self.nan_days, max_nans, in_channels, out_channels
        )

        log.info("Handling timestamps at the end of the ranges of interest.")
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

        log.info(f"{use_dask=}")
        if not use_dask:
            log.info("Initializing Datasets with Zarr only.")
            self.ds_raw = ZarrDataset(ranges, mode='r', classes=cfg.CFG.CLASSES)
            self.ds_classes = ZarrDataset(ranges, mode='c', classes=cfg.CFG.CLASSES)
            log.info("Initialization done.")

    @property
    def list_of_firsts(self):
        return self._list_of_firsts

    @property
    def weights(self):
        w = []
        log.info("Computing weights.")
        for i in tqdm(range(self.__len__())):
            seq, tru = self.get_total_pre(i)

            seq, tru = np.array(seq), np.array(tru)
            w.append(np.sum(seq) + np.sum(tru))
        w = torch.tensor(w)
        # Add a constant. Otherwise torch.multinomial complains TODO: is this really true?
        return w + self.min_weights_factor_of_max*torch.max(w)

    @property
    def timestamps_over_threshold(self):
        try:
            return self._timestamps_over_threshold
        except AttributeError:
            return None

    def get_total_pre(self, idx):
        log.debug("Getting total precipitation.")
        seq, tru = self.indices_tuple[idx]
        tot = {'seq': [], 'tru': []}
        for sub_period, indices in zip(tot, (seq, tru)):
            for t in indices:
                tot[sub_period].append(self._tot_pre.loc[self.sorted_sequence[t]])

        return tot['seq'], tot['tru']

    def __len__(self):
        return len(self.indices_tuple)

    def __getitem__(self, idx):
        seq, tru = self.indices_tuple[idx]
        item_tensors = {'seq': [], 'tru': []}
        for sub_period, indices in zip(item_tensors, (seq, tru)):
            for t in indices:
                if sub_period == 'seq' and cfg.CFG.MODE == 'r':
                    data = self.ds_raw[self.sorted_sequence[t]]
                else:
                    data = self.ds_classes[self.sorted_sequence[t]]
                item_tensors[sub_period].append(data)
            item_tensors[sub_period] = torch.from_numpy(
                np.concatenate(item_tensors[sub_period], axis=0).astype(np.float32)
            )

        return item_tensors['seq'], item_tensors['tru']

    @property
    def tot_pre(self):
        return self._tot_pre

    def indices_of_training(self):
        return self.indices_of('training')

    def indices_of_validation(self):
        return self.indices_of('validation')

    def indices_of_test(self):
        return self.indices_of('test')

    def indices_of(self, set):
        assert set in ('test', 'validation', 'training')
        indices_list_of = []
        if set == 'test':
            set_ranges = cfg.CFG.test_set_ranges
        elif set == 'validation':
            set_ranges = cfg.CFG.validation_set_ranges
        elif set == 'training':
            set_ranges = cfg.CFG.training_set_ranges
        else:
            raise KeyError(f"got {set} but expected either 'test', 'validation' or 'training'")
        set_timestamps_list = []
        for range_ in set_ranges:
            set_timestamps_list += range_.date_range()
        for idx, _ in enumerate(self.indices_tuple):
            seq, tru = self.indices_tuple[idx]
            if self.sorted_sequence[seq[0]] in set_timestamps_list:
                indices_list_of.append(idx)
        return indices_list_of

    def from_timestamp(self, timestamp):
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
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


class RadolanSubset(RadolanDataset):
    """Radolan Dataset Subset, training, validation or testing

    """

    def __init__(self, dataset: RadolanDataset, subset, valid_cases=20, seed=42, random_=False):
        log.info("Initializing %s", self.__class__.__name__)
        assert subset in ['train', 'valid', 'all', 'test']
        dataset_len = len(dataset)
        indices = [i for i in range(dataset_len)]
        log.debug(f"With {random_=} and {subset=}")
        if random_ and not subset == 'test':
            random.seed(seed)
            log.debug("Seed, set.")
            log.debug("Starting sort of indices.")
            indices = sorted(dataset.indices_of_validation() + dataset.indices_of_training())
            log.debug("Done sorting.")
            log.debug("Picking random sample.")
            valid_indices = random.sample(indices, k=round((valid_cases / 100) * len(indices)))
            log.debug("Done picking random sample.")
            log.debug("Dividing validation from training.")
            if subset == "valid":
                indices = [indices[i] for i in sorted(valid_indices)]
            else:
                indices = [x for x in indices if indices.index(x) not in valid_indices]
            log.debug("Done dividing.")
        elif subset == 'train':
            log.debug("Returning indices of training.")
            indices = dataset.indices_of_training()
        elif subset == 'valid':
            log.debug("Returning indices of validation.")
            indices = dataset.indices_of_validation()
        elif subset == 'test':
            log.debug("Returning indices of test.")
            indices = dataset.indices_of_test()

        log.debug("Setting dataset.")
        self.dataset = dataset
        log.debug("Setting indices.")
        self.indices = indices
        log.debug("Setting timestamps from lists of firsts.")
        self.timestamps = [x for n, x in enumerate(self.dataset.sorted_sequence) if n in self.dataset.list_of_firsts]
        log.debug("Setting timestamps using indices.")
        self.timestamps = [self.timestamps[self.indices[n]] for n, _ in enumerate(self.indices)]
        log.debug("Initialization done.")

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def from_timestamp(self, timestamp):
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
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


@utils.init_safety
def create_h5(mode: str, classes=None, filetype='Z', height=256, width=256, normal_ranges=None, video_ranges=None,
              path_to_folder=None, path_to_raw=None):

    if mode not in ('r', 'c'):
        raise ValueError(f"Need either 'r' or 'c' in mode but got {mode}")
    default_classes = {'0': (0, 0.5), '0.5': (0.5, 2), '2': (2, 5), '5': (5, 10), '10': (10, 30), '30': (30, np.infty)}
    histogram_bins = [n * 0.1 for n in range(501)]
    histogram_bins.append(np.infty)
    if classes is None:
        classes = default_classes

    to_create = []
    video_h5_to_create = []

    if normal_ranges is not None:
        normal_ranges_files_collection = DatasetFilesCollection(normal_ranges, filetype=filetype, classes=classes)
        to_create = normal_ranges_files_collection.missing_files(path_to_folder=path_to_folder, mode=mode)
    if video_ranges is not None:
        video_ranges_files_collection = DatasetFilesCollection(video_ranges, filetype=filetype, classes=classes)
        video_h5_to_create = video_ranges_files_collection.missing_files(path_to_folder=path_to_folder, mode=mode)
    if not to_create:
        to_create += video_h5_to_create
    else:
        for h5_file in video_h5_to_create:
            if h5_file not in to_create:
                to_create.append(h5_file)

    for file in to_create:
        year, month, file_name, mode = file.ymfm_tuple
        date_range = cfg.MonthDateRange(year=year, month=month).date_range()
        data = np.empty(shape=(len(date_range), height, width))
        time = np.array(date_range)
        for n, date in tqdm(enumerate(list(date_range))):

            try:
                raw_data = utils.square_select(date, height=height, width=width,
                                               plot=False, custom_path=path_to_raw).data
            except FileNotFoundError:
                # If not found then treat it as NaN-filled
                log.debug(f"Couldn't find raw data for {date}. Filling with NANs")
                raw_data = np.empty((height, width))
                raw_data[:] = np.nan

            if mode == 'c':  # skipped if in raw mode
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

        xds.to_zarr(
            store=os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name),
            mode='w', compute=True
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


def zarr_name(*args, **kwargs):
    h5 = h5_name(*args, **kwargs)
    if h5.endswith('.h5'):
        h5 = h5.replace('.h5', '.zarr')
    return h5


def files_names_list_single_mode(date_ranges, filetype='Z', **kwargs):
    if filetype == 'N':
        name_func = ncdf4_name
    elif filetype == 'H':
        name_func = h5_name
    elif filetype == 'Z':
        name_func = zarr_name
    else:
        raise ValueError(f'Got an unexpected value for h5_or_ncdf={filetype}. Value must be "N" or "H"')
    year_month_tuples = utils.ym_tuples(date_ranges)
    return [name_func(*ym, **kwargs) for ym in year_month_tuples]


def files_names_list_both_modes(date_ranges, filetype='Z', **kwargs):
    try:
        kwargs.pop('mode')
    except KeyError:
        pass
    return files_names_list_single_mode(date_ranges, filetype=filetype, mode='c',
                                        **kwargs) + files_names_list_single_mode(date_ranges, filetype=filetype,
                                                                                 mode='r', **kwargs)


def check_datasets_missing(date_ranges,  filetype='Z', **kwargs):
    unavailable = []
    required = files_names_list_both_modes(date_ranges, filetype=filetype, **kwargs)
    for file_name in required:
        if filetype == 'Z':
            if not os.path.isdir(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name)):
                unavailable.append(file_name)
            continue
        elif not os.path.isfile(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), file_name)):
            unavailable.append(file_name)
    return unavailable


class DatasetFilesCollection:
    def __init__(self, date_ranges, filetype='Z', **kwargs):
        self._ym_tuples = utils.ym_tuples(date_ranges)
        self._needed_files = {
            'raw': [DatasetFile(filetype=filetype, year=y, month=m, mode='r', **kwargs) for y, m in self._ym_tuples],
            'classes': [DatasetFile(filetype=filetype, year=y, month=m, mode='c', **kwargs) for y, m in self._ym_tuples]
        }

    def get_needed_files(self, mode='both'):
        if mode in ('r', 'raw'):
            return self._needed_files['raw']
        elif mode in ('c', 'classes'):
            return self._needed_files['classes']
        elif mode in ('b', 'both'):
            return self._needed_files['raw'] + self._needed_files['classes']
        else:
            raise ValueError(f"Unknown {mode=}")

    def missing_files(self, path_to_folder, mode):
        abs_path_to_folder = os.path.abspath(path_to_folder)
        files_list = self.get_needed_files(mode)
        missing = []
        for file in files_list:
            if not os.path.exists(os.path.join(abs_path_to_folder, file.file_name)):
                missing.append(file)
        return missing


class DatasetFile:
    def __init__(self, filetype, year: int, month: int, version_=None, mode=None, classes=None, with_extension=True):
        if filetype == 'N':
            name_func = ncdf4_name
        elif filetype == 'H':
            name_func = h5_name
        elif filetype == 'Z':
            name_func = zarr_name
        else:
            raise ValueError(f'Got an unexpected value for h5_or_ncdf={filetype}. Value must be "N" or "H"')
        self._file_name = name_func(
            year, month, version_=version_, mode=mode, classes=classes, with_extension=with_extension
        )
        self._year = year
        self._month = month
        self._mode = mode

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def file_name(self):
        return self._file_name

    @property
    def mode(self):
        return self._mode

    @property
    def ym_tuple(self):
        return self._year, self._month

    @property
    def ymf_tuple(self):
        return self._year, self._month, self._file_name

    @property
    def ymfm_tuple(self):
        return self._year, self._month, self._file_name, self._mode

    def __eq__(self, other):
        if not isinstance(other, DatasetFile):
            return False
        elif not other.ymfm_tuple == self.ymfm_tuple:
            return False
        return True

    def __str__(self):
        return f"<{self.__class__.__name__} called {self.file_name}>"


class H5Dataset:
    def __init__(self, date_ranges, mode, classes=None):
        self.mode = mode
        self.classes = classes
        self._date_ranges = date_ranges
        self._files_list = files_names_list_single_mode(self._date_ranges, filetype='Z', mode=mode)
        self._files_paths = [os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), fn) for fn in self._files_list]
        self.ds = xarray.open_mfdataset(paths=self._files_paths, chunks={'time': 20, 'lon': 256, 'lat': 256},
                                        engine='zarr') #, parallel=True)

    def __getitem__(self, item):
        if item == 'mode':
            return self.mode
        elif item == 'classes':
            return self.classes
        elif item:
            timestamp = item
            timestamps_grid = np.full(
                (1, cfg.CFG.HEIGHT, cfg.CFG.WIDTH),
                utils.normalized_time_of_day_from_string(item),
                dtype=np.float32
            )
            precipitation = self.ds.precipitation.loc[timestamp] #.to_numpy()  # .compute().to_numpy()
            precipitation = precipitation.compute()
            coordinates_array = cfg.CFG.coordinates_array
            return np.concatenate((np.expand_dims(precipitation, 0), timestamps_grid, coordinates_array))
        else:
            raise KeyError(f"{item} is an invalid Key for this H5Dataset")

    # def __iter__(self):
    #     return iter(self.ds.precpitation)


class ZarrDataset:
    def __init__(self, date_ranges, mode, classes=None):
        self.mode = mode
        self.classes = classes
        self._date_ranges = date_ranges
        self._files_list = files_names_list_single_mode(self._date_ranges, filetype='Z', mode=mode)
        self._files_paths = [os.path.join(os.path.abspath(cfg.CFG.RADOLAN_H5), fn) for fn in self._files_list]
        self._zarr_files = [PureZarrFile(file_name) for file_name in self._files_paths]
        self._index_of_last_checked = 0

    def __getitem__(self, item):
        out = None
        try:
            out = self._zarr_files[self._index_of_last_checked][item]
        except KeyError:
            pass

        if out is None:
            for i, zarr_file in enumerate(self._zarr_files):
                try:
                    out = zarr_file[item]
                    self._index_of_last_checked = i
                    break
                except KeyError:
                    pass
        if out is not None:
            return self.wrap_with_coordinates_and_time(out, item)
        raise KeyError(f"Key {item} is not in this {self.__class__.__name__}")

    @staticmethod
    def wrap_with_coordinates_and_time(out, item):
        coordinates_array = cfg.CFG.coordinates_array
        timestamps_grid = np.full(
            (1, cfg.CFG.HEIGHT, cfg.CFG.WIDTH),
            utils.normalized_time_of_day_from_string(item),
            dtype=np.float32
        )
        return np.concatenate((np.expand_dims(out, 0), timestamps_grid, coordinates_array))


class PureZarrFile:
    def __init__(self, file_name):
        self.file_name = file_name
        self._zarr_file = zarr.open(file_name, mode='r')
        self.start_time = dt.datetime.fromisoformat(self._zarr_file.time.attrs.get('units').split(' ', maxsplit=2)[2])
        self.max_hours_after_start = self._zarr_file.time[-1]

    def __getitem__(self, item):
        if not isinstance(item, dt.datetime):
            raise TypeError(f"Item must be of type {type(dt.datetime)} but got {type(item)}.")

        difference = item - self.start_time
        difference_hours = difference.seconds / 3600
        if not 0 <= difference_hours <= self.max_hours_after_start:
            raise KeyError(f"Key {item} is not in this {self.__class__.__name__}")

        index_ = int(difference_hours)
        return self._zarr_file.precipitation[index_]

    def __contains__(self, item):
        if not isinstance(item, dt.datetime):
            return False
        difference = item - self.start_time
        difference_hours = difference.seconds / 3600
        if not 0 <= difference_hours <= self.max_hours_after_start:
            return False
        return True



@contextmanager
def h5dataset_context_wrapper(*args, **kwargs):
    f = H5Dataset(*args, **kwargs)

    try:
        yield f
    finally:
        f.close()
