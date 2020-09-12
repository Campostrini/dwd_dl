"""various utilities

"""

import datetime as dt
import os

import torch
from torch.utils.data import DataLoader
import numpy as np

from .dataset import RadolanDataset


def unet_saver(trained_network, path=None, fname=None, timestamp=None):
    if path is None:
        path = input('Enter a path for saving the trained model. ')

    if fname is None:
        fname = network_name_timestamp(timestamp=timestamp)

    path_to_file = os.path.join(os.path.abspath(path), fname)
    basedir = os.path.dirname(path_to_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    torch.save(trained_network.state_dict(), path_to_file)

    return path_to_file


def network_name_timestamp(timestamp=None):
    name_template = 'UNet_run_{}.pt'
    if timestamp is not None:
        name = name_template.format(timestamp)
    else:
        name = name_template.format(dt.datetime.now())

    return name


def network_loader(path_to_saved_model, network_class, *args, **kwargs):
    model = network_class(*args, **kwargs)
    model.load_state_dict(torch.load(path_to_saved_model))
    model.eval()

    return model


class ModelEvaluator:
    def __init__(self, path_to_saved_model, network_class, radolan_dir, date_ranges_path, device='cuda:0'):
        self.model = network_loader(path_to_saved_model, network_class)
        if torch.cuda.is_available():
            self._device = device
        else:
            self._device = 'cpu'
        self._train_dataset = RadolanDataset(
            radolan_dir=radolan_dir,
            date_ranges_path=date_ranges_path,
            image_size=256,
            subset='train',
            validation_cases=20,
            seed=42,
            in_channels=6,
            out_channels=1
        )
        self._valid_dataset = RadolanDataset(
            radolan_dir=radolan_dir,
            date_ranges_path=date_ranges_path,
            image_size=256,
            subset='validation',
            validation_cases=20,
            seed=42,
            in_channels=6,
            out_channels=1
        )

        def worker_init(worker_id):
            np.random.seed(42 + worker_id)

        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=1,
            num_workers=8,
            worker_init_fn=worker_init
        )
        self._valid_loader = DataLoader(
            self._valid_dataset,
            batch_size=1,
            num_workers=8,
            worker_init_fn=worker_init
        )

        self._dataset = {'train': self._train_dataset, 'valid': self._valid_dataset}

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        assert device in ['cpu', 'cuda:0']  # TODO: write better implementation
        self._device = device
        print('Warning, this way of setting device may be not complete.')

    def on_timestamp(self, timestamp):
        valid = self._valid_dataset.has_timestamp(timestamp)
        train = self._train_dataset.has_timestamp(timestamp)
        if not valid and not train:
            raise ValueError('Current timestamp {} is not present, neither '
                             'in validation nor in training dataset.'.format(timestamp))
        elif valid and train:
            raise ValueError('Current timestamp {} is present in both validation and training dataset.'
                             'This error should never occur. Check code.'.format(timestamp))
        elif valid:
            return self._evaluate('valid', timestamp)
        else:
            return self._evaluate('train', timestamp)

    def _evaluate(self, phase, timestamp, to_numpy=True):
        assert phase in ['valid', 'train']
        data = self._dataset[phase].from_timestamp(timestamp)
        model = self.model
        device = self.device
        model.to(device)
        x, y_true = data
        x, y_true = torch.unsqueeze(x, 0), torch.unsqueeze(y_true, 0)
        x, y_true = x.to(device), y_true.to(device)
        y = model(x)
        x, y, y_true = x.cpu().detach().numpy(), y.cpu().detach().numpy(), y_true.cpu().detach().numpy()
        return x, y, y_true
