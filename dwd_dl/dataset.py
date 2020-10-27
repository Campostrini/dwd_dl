import os
import random
import datetime as dt
from contextlib import contextmanager

import numpy as np
import torch
# from skimage.io import imread
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import h5py
import hashlib

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from . import cfg
from .cfg import TrainingPeriod
from . import preproc


class RadolanDataset(Dataset):
    """DWD Radolan for nowcasting"""

    in_channels = 6
    out_channels = 1

    def __init__(
        self,
        h5file_handle,
        date_ranges_path,
        image_size=256,
        in_channels=in_channels,
        out_channels=out_channels,
        verbose=False,
        normalize=False,
    ):

        # read radolan files
        self.file_handle = h5file_handle
        self.normalize = normalize
        print("reading images...")
        self._image_size = image_size
        self.training_period = TrainingPeriod(date_ranges_path)
        tot_pre = {}
        nan_days = {}
        std = {}
        mean = {}
        for date in h5file_handle:
            tot_pre[date] = h5file_handle[date].attrs['tot_pre']
            nan_days[date] = h5file_handle[date].attrs['NaN']
            std[date] = h5file_handle[date].attrs['std']
            mean[date] = h5file_handle[date].attrs['mean']

        self._total_prec = tot_pre
        self.sequence = sorted(h5file_handle)
        self.sorted_sequence = sorted(self.sequence)
        self._sequence_timestamps = sorted(self.sequence)

        to_remove = []
        not_for_mean = []
        nan_to_num = []
        for date in nan_days:
            # TODO: Rewrite in one function
            if nan_days[date] > 10:   # max 10 nans, TODO : Remove hard coding.
                time_stamp = dt.datetime.strptime(date, '%y%m%d%H%M')
                dr = cfg.daterange(
                    time_stamp - dt.timedelta(hours=in_channels+out_channels-1),
                    time_stamp,
                    include_end=True
                )
                not_for_mean.append(time_stamp.strftime('%y%m%d%H%M'))
                for to_remove_ts in dr:
                    if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                        to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))
            else:
                nan_to_num.append(date)  # self.sequence[bad_ts] = np.nan_to_num(self.sequence[bad_ts])

        for _, range_end in self.training_period.ranges_list:
            dr = cfg.daterange(
                range_end - dt.timedelta(hours=in_channels+out_channels-2),
                range_end,
                include_end=True
            )

            for to_remove_ts in dr:
                if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                    to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))

        self.to_remove = to_remove
        self.not_for_mean = not_for_mean
        self.nan_to_num = nan_to_num

        print("Normalizing...")
        self.mean = 0
        self.std = 0
        for n, date in enumerate(self.sequence):
            n_in_img = self._image_size ** 2
            if date not in not_for_mean:
                m = mean[date]
                s = std[date]
                self.std = (((n * n_in_img - 1) * (self.std ** 2) + (n_in_img - 1)*(s**2)) / ((n + 1)*n_in_img - 1)) + (
                    (n * n_in_img * n_in_img) * ((self.mean - m) ** 2) / (((n+1) * n_in_img)*((n+1) * n_in_img - 1))
                )
                self.std = np.sqrt(self.std)
                self.mean = (n * n_in_img * self.mean + (m * n_in_img)) / (
                    (n+1) * n_in_img
                )

        print("done loading dataset")

        # create global index for sequence and true_rainfall (idx -> ([s_idxs], [t_idx]))
        # num_slices = [v.shape[0] for v, m in self.volumes]
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
    def weights(self):
        w = []
        print('Computing weights.')
        for i in tqdm(range(self.__len__())):
            seq, tru = self.get_total_prec(i)

            # Add a constant. Otherwise torch.multinomial complains TODO: is this really true?
            # TODO: revise weights
            seq, tru = np.array(seq), np.array(tru)
            w.append(np.sum(seq) + np.sum(tru))
        return torch.tensor(w) + 0.01

    def get_total_prec(self, idx):
        seq, tru = self.indices_tuple[idx]
        # TODO: remove duplicate
        tot_seq = []
        for t in seq:
            tot_seq.append(self._total_prec[self.sorted_sequence[t]])

        tot_tru = []
        for t in tru:
            tot_tru.append(self._total_prec[self.sorted_sequence[t]])

        return tot_seq, tot_tru

    def __len__(self):
        return len(self.indices_tuple)

    def __getitem__(self, idx):
        seq, tru = self.indices_tuple[idx]
        # TODO: Remove duplicate code
        sequence = []
        for t in seq:
            data = self.file_handle[self.sorted_sequence[t]]
            # if self.sorted_sequence[t] in self.nan_to_num:
            #     data = np.nan_to_num(data)
            sequence.append(data)
        sequence = np.stack(sequence)
        # if self.normalize:
        #     sequence = preproc.normalize(sequence, self.mean, self.std)

        true_rainfall = []
        for t in tru:
            data = self.file_handle[self.sorted_sequence[t]]
            # if self.sorted_sequence[t] in self.nan_to_num:
            #     data = np.nan_to_num(data)
            true_rainfall.append(data)
        true_rainfall = np.stack(true_rainfall)
        # if self.normalize:
        #     true_rainfall = preproc.normalize(true_rainfall, self.mean, self.std)
        sequence_tensor = torch.from_numpy(sequence.astype(np.float32))
        true_rainfall_tensor = torch.from_numpy(true_rainfall.astype(np.float32))

        # return tensors
        return sequence_tensor, true_rainfall_tensor

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


class RadolanSubset(RadolanDataset):
    """Radolan Dataset Subset, training, validation or testing

    """

    def __init__(self, dataset, subset, validation_cases=20, seed=42):
        assert subset in ['train', 'validation', 'all']
        dataset_len = len(dataset)
        indices = [i for i in range(dataset_len)]
        if not subset == "all":
            random.seed(seed)
            validation_indices = random.sample(indices, k=round((validation_cases/100) * dataset_len))
            if subset == "validation":
                indices = [indices[i] for i in sorted(validation_indices)]
            else:
                indices = [x for x in indices if indices.index(x) not in validation_indices]
        self.dataset = dataset
        self.indices = indices
        self.timestamps = [x for n, x in enumerate(self.dataset.sorted_sequence) if n in self.dataset._list_of_firsts]
        self.timestamps = [self.timestamps[self.indices[n]] for n, _ in enumerate(self.indices)]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def from_timestamp(self, timestamp):
        try:
            timestamp_raw_idx = self.dataset.sorted_sequence.index(timestamp)
            index = self.dataset._list_of_firsts.index(timestamp_raw_idx)
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

    def get_total_prec(self, idx):
        return self.dataset.get_total_prex[self.indices[idx]]


@cfg.init_safety
def create_h5(filename, keep_open=True, height=256, width=256, verbose=False):

    classes = {'0': (0, 0.1), '0.1': (0.1, 1), '1': (1, 2.5), '2.5': (2.5, np.infty)}

    f = read_h5(filename)
    try:
        hash_check = (f.attrs['hash'] == cfg.CFG.get_timestamps_hash())
    except KeyError:
        hash_check = False

    if not hash_check:
        string_for_md5 = ''
        training_period = cfg.CFG.date_ranges
        for date_range in training_period:
            for date in tqdm(date_range.date_range()):
                date_str = date.strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
                if verbose:
                    print('Processing {}'.format(date_str))
                if date_str not in f.keys():
                    file_name = cfg.binary_file_name(time_stamp=date)
                    data = preproc.square_select(date, height=height, width=width, plot=False).data
                    tot_nans = np.count_nonzero(np.isnan(data))
                    data = np.nan_to_num(data)
                    f[date_str] = np.array(
                        [(classes[class_name][0] <= data) & (data < classes[class_name][1]) for class_name in classes]
                    ).astype(int)
                    f[date_str].attrs['filename'] = file_name
                    f[date_str].attrs['NaN'] = tot_nans
                    f[date_str].attrs['img_size'] = height * width
                    f[date_str].attrs['tot_pre'] = np.nansum(data)
                    f[date_str].attrs['mean'] = np.nanmean(data)
                    f[date_str].attrs['std'] = np.nanstd(data)
                string_for_md5 += date_str
        f.attrs['hash'] = hashlib.md5(string_for_md5.encode()).hexdigest()
    if keep_open:
        return f
    else:
        f.close()


def read_h5(filename):
    if not filename.endswith('.h5'):
        filename += '.h5'

    f = h5py.File(os.path.join(os.path.abspath(cfg.CFG.RADOLAN_ROOT), filename), 'a')

    return f


@contextmanager
def h5_handler(*args, **kwargs):

    file = create_h5(*args, **kwargs)

    try:
        yield file
    finally:
        file.close()
