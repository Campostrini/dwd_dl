import os
import random
import datetime as dt

import numpy as np
import torch
# from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from . import config
from .config import TrainingPeriod
from . import preproc


class RadolanDataset(Dataset):
    """DWD Radolan for nowcasting"""

    in_channels = 6
    out_channels = 1

    def __init__(
        self,
        radolan_dir,
        date_ranges_path,
        image_size=256,
        subset="train",
        validation_cases=20,  # percentage
        seed=42,
        in_channels=in_channels,
        out_channels=out_channels,
        verbose=False
    ):
        assert subset in ["all", "train", "validation"]

        # read radolan files
        sequence = {}
        print("reading {} images...".format(subset))
        self.days_containing_nans = {}
        self.training_period = TrainingPeriod(date_ranges_path)
        for (dirpath, dirnames, filenames) in os.walk(radolan_dir):
            counter = 0
            for filename in tqdm(
                    sorted(
                        filter(lambda f: "dwd---bin" in f and f in self.training_period.file_names_list, filenames),
                        key=lambda x: int(x.split("-")[-5])
                    )
            ):
                # filepath = os.path.join(dirpath, filename)
                counter += 1
                key = filename.split("-")[-5]
                ts = dt.datetime.strptime(filename.split("-")[-5], '%y%m%d%H%M')
                file_data = preproc.square_select(ts, height=image_size, width=image_size, plot=False).data
                sequence[key] = file_data
                if np.isnan(file_data).any():
                    self.days_containing_nans[key] = (np.count_nonzero(np.isnan(file_data)),
                                                      np.argwhere(np.isnan(file_data)))
                if counter % 10 == 0 and verbose:
                    print("Loaded data for timestamp: " + key)

        self.sequence = sequence
        self.sorted_sequence = sorted(sequence)
        self._sequence_timestamps = sorted(sequence)

        to_remove = []
        not_for_mean = []
        for bad_ts in self.days_containing_nans:
            # TODO: Rewrite in one function
            if self.days_containing_nans[bad_ts][0] > 10:   # max 10 nans, TODO : Remove hard coding.
                time_stamp = dt.datetime.strptime(bad_ts, '%y%m%d%H%M')
                dr = config.daterange(
                    time_stamp - dt.timedelta(hours=in_channels+out_channels-1),
                    time_stamp,
                    include_end=True
                )
                not_for_mean.append(time_stamp.strftime('%y%m%d%H%M'))
                for to_remove_ts in dr:
                    if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                        to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))
            else:
                self.sequence[bad_ts] = np.nan_to_num(self.sequence[bad_ts])

        for _, range_end in self.training_period.ranges_list:
            dr = config.daterange(
                range_end - dt.timedelta(hours=in_channels+out_channels-2),
                range_end,
                include_end=True
            )

            for to_remove_ts in dr:
                if to_remove_ts.strftime('%y%m%d%H%M') not in to_remove:
                    to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))

        to_remove = list(dict.fromkeys(to_remove))
        self.to_remove = to_remove

        # select cases to subset
        # if not subset == "all":
        #     random.seed(seed)
        #     validation_timestamps = random.sample(
        #         self._sequence_timestamps,
        #         k=round((validation_cases/100) * (len(self._sequence_timestamps) - in_channels - out_channels + 1))
        #     )
        #     if subset == "validation":
        #         self._sequence_timestamps = sorted(validation_timestamps)
        #     else:
        #         self._sequence_timestamps = sorted(
        #             list(set(self._sequence_timestamps).difference(validation_timestamps))
        #         )

        print("Normalizing...")
        self.mean = np.mean(np.stack([self.sequence[x] for x in self.sequence if x not in not_for_mean]))
        self.std = np.std(np.stack([self.sequence[x] for x in self.sequence if x not in not_for_mean]))
        self.sequence = {k: preproc.normalize(self.sequence[k], self.mean, self.std) for k in tqdm(self.sequence)}

        print("done loading {} dataset".format(subset))

        # create global index for sequence and true_rainfall (idx -> ([s_idxs], [t_idx]))
        # num_slices = [v.shape[0] for v, m in self.volumes]
        self.indices_tuple_raw = [
            [
                [i + n for n in range(in_channels)],
                [i + in_channels + n for n in range(out_channels)]
            ] for i, _ in enumerate(self.sorted_sequence)
        ]

        self.indices_tuple = [
            self.indices_tuple_raw[i] for i, key in enumerate(self.sorted_sequence) if key not in to_remove
        ]

        if not subset == "all":
            random.seed(seed)
            validation_timestamps_indices = random.sample(
                [i for i, _ in enumerate(self.indices_tuple)], k=round((validation_cases/100) * len(self.indices_tuple))
            )
            if subset == "validation":
                self.indices_tuple = [
                    self.indices_tuple[i] for i in sorted(validation_timestamps_indices)
                ]
            else:
                self.indices_tuple = [
                    x for x in self.indices_tuple
                    if self.indices_tuple.index(x) not in validation_timestamps_indices
                ]

    @property
    def weights(self):
        w = []
        for i in range(self.__len__()):
            seq, tru = self.__getitem__(i)

            # Unnormalize and add a constant. Otherwise torch.multinomial complains TODO: is this really true?
            # TODO: revise weights
            seq, tru = seq * self.std + self.mean + 1, tru * self.std + self.mean + 1
            w.append(torch.sum(seq) + torch.sum(tru))
        return torch.stack(w)/torch.sum(torch.tensor(w))

    def __len__(self):
        return len(self.indices_tuple)

    def __getitem__(self, idx):
        seq, tru = self.indices_tuple[idx]

        sequence = np.stack([self.sequence[self.sorted_sequence[t]] for t in seq])
        true_rainfall = np.stack([self.sequence[self.sorted_sequence[t]] for t in tru])

        sequence_tensor = torch.from_numpy(sequence.astype(np.float32))
        true_rainfall_tensor = torch.from_numpy(true_rainfall.astype(np.float32))

        # return tensors
        return sequence_tensor, true_rainfall_tensor

    def from_timestamp(self, timestamp):
        list_of_firsts = [row[0][0] for row in self.indices_tuple]
        try:
            index_in_timestamps_list = self.sorted_sequence.index(timestamp)
            true_index = list_of_firsts.index(index_in_timestamps_list)
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
