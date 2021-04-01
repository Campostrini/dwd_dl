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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def prepare_data(self):
        create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)

    def setup(self, stage=None):
        f = H5Dataset(cfg.CFG.date_ranges, mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)
        self.dataset = Dataset(
            h5file_handle=f,
            date_ranges_path=cfg.CFG.DATE_RANGES_FILE_PATH,
            image_size=self.image_size
        )

        self.train_dataset = Subset(
            dataset=self.dataset,
            subset='train',
            valid_cases=20,  # Percentage
        )

        self.valid_dataset = Subset(
            dataset=self.dataset,
            subset='valid',
            valid_cases=20  # Percentage
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
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

