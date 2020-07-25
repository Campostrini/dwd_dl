import os
import random
import datetime as dt

import numpy as np
import torch
# from skimage.io import imread
from torch.utils.data import Dataset

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
        random_sampling=True,
        validation_cases=10,  # percentage
        seed=42,
        in_channels=in_channels,
        out_channels=out_channels
    ):
        assert subset in ["all", "train", "validation"]

        # check if time period is valid
        for t in time_period:
            assert config.START_DATE <= t <= config.END_DATE
        assert time_period[0] < time_period[1]

        # read radolan files
        sequence = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(radolan_dir):
            for filename in sorted(
                filter(lambda f: "dwd---bin" in f and f in config.used_files(*time_period), filenames),
                key=lambda x: int(x.split("-")[-5]),
            ):
                # filepath = os.path.join(dirpath, filename)
                ts = dt.datetime.strptime(filename.split("-")[-5], '%y%m%d%H%M')
                file_data = preproc.square_select(ts, plot=False).data
                sequence[filename.split("-")[-5]] = file_data
                print("Loaded data for timestamp: " + filename.split("-")[-5])

        self.sequence = sequence
        self._sequence_timestamps = sorted(sequence)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_timestamps = random.sample(
                self._sequence_timestamps,
                k=round(validation_cases/100) * (len(self._sequence_timestamps) - in_channels - out_channels + 1)
            )
            if subset == "validation":
                self._sequence_timestamps = sorted(validation_timestamps)
            else:
                self._sequence_timestamps = sorted(
                    list(set(self._sequence_timestamps).difference(validation_timestamps))
                )

        # print("preprocessing {} volumes...".format(subset))
        # # create list of tuples (volume, mask)
        # self.volumes = [(volumes[k], masks[k]) for k in self.patients]
        #
        # print("cropping {} volumes...".format(subset))
        # # crop to smallest enclosing volume
        # self.volumes = [crop_sample(v) for v in self.volumes]
        #
        # print("padding {} volumes...".format(subset))
        # # pad to square
        # self.volumes = [pad_sample(v) for v in self.volumes]
        #
        # print("resizing {} volumes...".format(subset))
        # # resize
        # self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
        #
        # print("normalizing {} volumes...".format(subset))
        # # normalize channel-wise
        # self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]
        #
        # # probabilities for sampling slices based on masks
        # self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        # self.slice_weights = [
        #     (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        # ]
        #
        # # add channel dimension to masks
        # self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done loading {} dataset".format(subset))

        # create global index for sequence and true_rainfall (idx -> ([s_idxs], [t_idx]))
        # num_slices = [v.shape[0] for v, m in self.volumes]
        self.sequence_sample_true_rainfall_index = [
            (
                [i + n for n in range(in_channels)],
                [i + in_channels + n for n in range(out_channels)]) for i in range(
                len(self._sequence_timestamps) - in_channels - out_channels + 1
            )
        ]

        self.random_sampling = random_sampling
        print("Random Sampling has been set to {}".format(random_sampling))
        # self.transform = transform

    def __len__(self):
        return len(self.sequence_sample_true_rainfall_index)

    def __getitem__(self, idx):
        seq, tru = self.sequence_sample_true_rainfall_index[idx]

        if self.random_sampling:
            seq, tru = self.sequence_sample_true_rainfall_index[np.random.randint(len(self.sequence))]
            # slice_n = np.random.choice(
            #     range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            # )

        # v, m = self.volumes[patient]
        # image = v[slice_n]
        # mask = m[slice_n]

        sequence = np.stack([self.sequence[self._sequence_timestamps[t]] for t in seq])
        true_rainfall = np.stack([self.sequence[self._sequence_timestamps[t]] for t in tru])

        # if self.transform is not None:
        #     image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        # image = image.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)

        sequence_tensor = torch.from_numpy(sequence.astype(np.float32))
        true_rainfall_tensor = torch.from_numpy(true_rainfall.astype(np.float32))

        # return tensors
        return sequence_tensor, true_rainfall_tensor
