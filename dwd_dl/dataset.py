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
from . import preproc


class RadolanDataset(Dataset):
    """DWD Radolan for nowcasting"""

    in_channels = 6
    out_channels = 1

    def __init__(
        self,
        radolan_dir,
        time_period,  # datetime.datetime list of 2
        # transform=None,
        image_size=256,
        subset="train",
        intrinsic_random_sampling=False,
        validation_cases=20,  # percentage
        seed=42,
        in_channels=in_channels,
        out_channels=out_channels,
        verbose=False
    ):
        assert subset in ["all", "train", "validation"]

        # check if time period is valid
        for t in time_period:
            assert config.START_DATE <= t <= config.END_DATE
        assert time_period[0] < time_period[1]

        # read radolan files
        sequence = {}
        print("reading {} images...".format(subset))
        self.days_containing_nans = {}
        for (dirpath, dirnames, filenames) in os.walk(radolan_dir):
            counter = 0
            for filename in tqdm(
                    sorted(
                        filter(lambda f: "dwd---bin" in f and f in config.used_files(*time_period), filenames),
                        key=lambda x: int(x.split("-")[-5])
                    )
            ):
                # filepath = os.path.join(dirpath, filename)
                counter += 1
                key = filename.split("-")[-5]
                ts = dt.datetime.strptime(filename.split("-")[-5], '%y%m%d%H%M')
                file_data = preproc.square_select(ts, plot=False).data
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

            if self.days_containing_nans[bad_ts][0] > 10:   # max 10 nans, TODO : Remove hard coding.
                time_stamp = dt.datetime.strptime(bad_ts, '%y%m%d%H%M')
                dr = config.daterange(time_stamp - dt.timedelta(hours=in_channels+out_channels-1),
                                      time_stamp + dt.timedelta(hours=in_channels+out_channels-1),
                                      include_end=True)
                not_for_mean.append(time_stamp.strftime('%y%m%d%H%M'))
                for to_remove_ts in dr:
                    to_remove.append(to_remove_ts.strftime('%y%m%d%H%M'))
            else:
                self.sequence[bad_ts] = np.nan_to_num(self.sequence[bad_ts])

            to_remove = list(dict.fromkeys(to_remove))

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
        self.sequence_sample_true_rainfall_index_raw = [
            (
                [i + n for n in range(in_channels)],
                [i + in_channels + n for n in range(out_channels)]) for i in range(
                len(self.sorted_sequence) - in_channels - out_channels + 1
            )
        ]

        self.sequence_sample_true_rainfall_index = [
            self.sequence_sample_true_rainfall_index_raw[
                i
            ] for i, key in enumerate(
                self.sorted_sequence
            ) if key not in to_remove and i < len(self.sequence_sample_true_rainfall_index_raw)
        ]  # TODO: Refactor, looks really bad.

        if not subset == "all":
            random.seed(seed)
            validation_timestamps_indeces = random.sample(
                [i for i, _ in enumerate(self.sequence_sample_true_rainfall_index)],
                k=round(
                    (validation_cases/100) * (
                            len(self.sequence_sample_true_rainfall_index) - in_channels - out_channels + 1)
                )
            )
            if subset == "validation":
                self.sequence_sample_true_rainfall_index = [
                    self.sequence_sample_true_rainfall_index[i] for i in sorted(validation_timestamps_indeces)
                ]
            else:
                self.sequence_sample_true_rainfall_index = [
                    x for x in self.sequence_sample_true_rainfall_index
                    if self.sequence_sample_true_rainfall_index.index(x) not in validation_timestamps_indeces
                ]

                # sorted(
                #     list(set(self.sequence_sample_true_rainfall_index).difference(set(validation_timestamps_indeces)))
                # )

        self.intrinsic_random_sampling = intrinsic_random_sampling
        print("Intrinsic Random Sampling has been set to {}".format(intrinsic_random_sampling))
        # self.transform = transform

    @property
    def weights(self):
        w = []
        for i in range(self.__len__()):
            seq, tru = self.__getitem__(i)

            # Unnormalize and add a constant. Otherwise torch.multinomial complains
            seq, tru = seq * self.std + self.mean + 1, tru * self.std + self.mean + 1
            w.append(torch.sum(seq) + torch.sum(tru))
        return torch.stack(w)/torch.sum(torch.tensor(w))

    def __len__(self):
        return len(self.sequence_sample_true_rainfall_index)

    def __getitem__(self, idx):
        seq, tru = self.sequence_sample_true_rainfall_index[idx]

        # The actual random sampling is already taken care of by the sampler.
        if self.intrinsic_random_sampling:
            seq, tru = self.sequence_sample_true_rainfall_index[np.random.randint(
                len(self.sequence_sample_true_rainfall_index))]


            # slice_n = np.random.choice(
            #     range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            # )

        # v, m = self.volumes[patient]
        # image = v[slice_n]
        # mask = m[slice_n]

        sequence = np.stack([self.sequence[self.sorted_sequence[t]] for t in seq])
        true_rainfall = np.stack([self.sequence[self.sorted_sequence[t]] for t in tru])

        # if self.transform is not None:
        #     image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        # image = image.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)

        sequence_tensor = torch.from_numpy(sequence.astype(np.float32))
        true_rainfall_tensor = torch.from_numpy(true_rainfall.astype(np.float32))

        # return tensors
        return sequence_tensor, true_rainfall_tensor
