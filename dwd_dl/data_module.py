import datetime as dt

from pytorch_lightning import LightningDataModule
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

from dwd_dl.dataset import create_h5, H5Dataset, RadolanDataset as Dataset, RadolanSubset as Subset
from dwd_dl import cfg


class RadolanDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, image_size):
        super().__init__()
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def prepare_data(self):
        create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)

    def setup(self, stage=None, random_=False):
        self.dataset = Dataset(
            image_size=self.image_size
        )

        self.train_dataset = Subset(
            dataset=self.dataset,
            subset='train',
            random_=random_,
            valid_cases=20,  # Percentage if random_
        )

        self.valid_dataset = Subset(
            dataset=self.dataset,
            subset='valid',
            random_=random_,
            valid_cases=20  # Percentage if random_
        )

        self.test_dataset = Subset(
            dataset=self.dataset,
            subset='test',
        )

    def train_dataloader(self):
        weighted_random_sampler = WeightedRandomSampler(
            weights=self.train_dataset.weights,
            num_samples=len(self.train_dataset),
            replacement=False
        )

        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), sampler=weighted_random_sampler,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), pin_memory=True)

    @property
    def legal_timestamps(self):
        return [x for n, x in enumerate(self.dataset.sorted_sequence) if n in self.dataset.list_of_firsts]

    def legal_datetimes(self):
        return self.legal_timestamps

    def close(self):
        pass


class VideoDataModule(RadolanDataModule):
    def __init__(self, *args):
        super(VideoDataModule, self).__init__(*args)

    def setup(self, stage=None):
        self.dataset = Dataset(
            image_size=self.image_size,
            video_or_normal='video'
        )

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), pin_memory=True)

    def train_dataloader(self):
        raise NotImplementedError("This DataModule should not be used for training.")

    def val_dataloader(self):
        raise NotImplementedError("This DataModule should not be used for validation.")


class RadolanLiveDataModule(RadolanDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def legal_timestamps(self):
        return sorted(self.valid_dataset.timestamps + self.train_dataset.timestamps)

    def all_timestamps(self):
        return self.dataset.sorted_sequence

    def which_dataset(self, timestamp):
        valid = self.valid_dataset.has_timestamp(timestamp)
        train = self.train_dataset.has_timestamp(timestamp)
        if not valid and not train:
            raise ValueError('Current timestamp {} is not present, neither '
                             'in validation nor in training dataset.'.format(timestamp))
        elif valid and train:
            raise ValueError('Current timestamp {} is present in both validation and training dataset.'
                             'This error should never occur. Check code.'.format(timestamp))
        elif valid:
            return 'valid'
        else:
            return 'train'

