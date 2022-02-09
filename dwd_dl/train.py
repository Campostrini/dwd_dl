import json
import os
import datetime as dt
from contextlib import contextmanager
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from dwd_dl.dataset import RadolanDataset as Dataset, RadolanSubset as Subset, create_h5, H5Dataset
from dwd_dl.unet import UNet
from dwd_dl import cfg
from dwd_dl import utils
from dwd_dl.cli import RadolanParser
import dwd_dl as dl
from dwd_dl import yaml_utils


def main(device, verbose, batch_size, workers,
         image_size, lr, epochs, save, cat, **kwargs):
    device = torch.device("cpu" if not torch.cuda.is_available() else device)
    print(f"Using: {device}")

    with data_loaders(batch_size=batch_size, workers=workers, image_size=image_size) as loaders_:
        loader_train, loader_valid = loaders_
        loaders = {"train": loader_train, "valid": loader_valid}

        unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels, cat=cat)
        unet.to(device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        optimizer = optim.Adadelta(unet.parameters(), lr=lr)

        loss_train = []
        loss_valid = []
        run = str(dt.datetime.now())
        writer = SummaryWriter(os.path.join(cfg.CFG.RADOLAN_ROOT, 'Logs', run))

        step = 0

        for epoch in tqdm(range(epochs), total=epochs):
            last_epoch = epoch == epochs - 1
            epoch_loss_train = []
            epoch_loss_valid = []
            pr_data = {'train': {'pred': [], 'true': []}, 'valid': {'pred': [], 'true': []}}
            epoch_accuracy_train = 0
            total_elements_train = 0
            epoch_accuracy_valid = 0
            total_elements_valid = 0
            for phase in ["train", "valid"]:
                if phase == "train":
                    unet.train()
                else:
                    unet.eval()

                for step, data in enumerate(loaders[phase]):

                    x, y_true = data
                    y_true = y_true[:, ::5, ...].to(dtype=torch.long)
                    x, y_true = x.to(device), y_true.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = unet(x)

                        loss = cross_entropy_loss(y_pred, y_true)

                        # Needed to compute stats
                        batch_elements = 1
                        for dim_ in y_true.shape:
                            batch_elements *= dim_
                        batch_elements /= 4  # number of classes

                        if phase == "valid":
                            loss_valid.append(loss.item())
                            epoch_loss_valid.append(loss.item())
                            y_pred_class_indices = torch.topk(y_pred, 1, dim=1).indices
                            y_true_class_indices = torch.unsqueeze(y_true, dim=1)
                            correct = (
                                    y_pred_class_indices == y_true_class_indices
                            ).float().sum()
                            epoch_accuracy_valid += correct
                            total_elements_valid += batch_elements

                        if phase == "train":
                            loss_train.append(loss.item())
                            epoch_loss_train.append(loss.item())
                            loss.backward()
                            optimizer.step()
                            total_elements_train += batch_elements
                            y_pred_class_indices = torch.topk(y_pred, 1, dim=1).indices
                            y_true_class_indices = torch.unsqueeze(y_true, dim=1)
                            correct = (
                                    y_pred_class_indices == y_true_class_indices
                            ).float().sum()
                            epoch_accuracy_train += correct

                        if last_epoch:
                            pr_data[phase]['pred'].append(y_pred_class_indices.cpu().detach().numpy())
                            pr_data[phase]['true'].append(y_true_class_indices.cpu().detach().numpy())

                    if phase == "train" and (step + 1) % 10 == 0:
                        if verbose:
                            print(loss_train, step)
                            print('In Training.')
                        loss_train = []

                    if phase == "valid" and (step + 1) % 10 == 0:
                        if verbose:
                            print(loss_valid, step)
                            print('In Validation.')
                        loss_valid = []

            writer.add_scalar('Loss/train', np.array(epoch_loss_train).mean(), epoch)
            writer.add_scalar('Loss/valid', np.array(epoch_loss_valid).mean(), epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy_train/total_elements_train, epoch)
            writer.add_scalar('Accuracy/valid', epoch_accuracy_valid/total_elements_valid, epoch)
            # writer.add_pr_curve('Precision & Recall', )
            writer.flush()
            print(f"Epoch: {epoch} finished.")

        for phase in pr_data:
            for set_ in pr_data[phase]:
                pr_data[phase][set_] = np.concatenate(pr_data[phase][set_], axis=0)

        # writer.add_figure('Confusion Matrix', sklearn.metrics.plot_confusion_martix())

        if save:
            saved_name_path = utils.unet_saver(
                unet,
                path=os.path.join(cfg.CFG.RADOLAN_ROOT, 'Models', run),
                timestamp=run
            )
            print('Saved Unet state_dict: {}'.format(saved_name_path))


@contextmanager
def data_loaders(batch_size, workers, **kwargs):
    dataset_train, dataset_valid = datasets(**kwargs)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    weighted_rand_sampler = WeightedRandomSampler(
        weights=dataset_train.weights,
        num_samples=len(dataset_train),
        replacement=False
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
        sampler=weighted_rand_sampler
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init
    )

    try:
        yield loader_train, loader_valid

    finally:
        loader_train.dataset.dataset.file_handle.close()
        loader_valid.dataset.dataset.file_handle.close()


def datasets(image_size):
    create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES, normal_ranges=cfg.CFG.date_ranges,
              video_ranges=cfg.CFG.video_ranges, path_to_folder=cfg.CFG.RADOLAN_H5)
    f = H5Dataset(cfg.CFG.date_ranges, mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)
    dataset = Dataset(
        h5file_handle=f,
        date_ranges_path=cfg.CFG.DATE_RANGES_FILE_PATH,
        image_size=image_size
    )

    train = Subset(
        dataset=dataset,
        subset='train',
        valid_cases=20,  # Percentage
    )

    valid = Subset(
        dataset=dataset,
        subset='valid',
        valid_cases=20  # Percentage
    )

    return train, valid


def makedirs(weights, logs):
    os.makedirs(weights, exist_ok=True)
    os.makedirs(logs, exist_ok=True)


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    kwargs_ = vars(parser.parse_args())
    if 'filename' in kwargs_:
        warnings.warn("The --filename option is no longer supported. It will be ignored.", DeprecationWarning)
        del kwargs_['filename']
    yaml_utils.log_dump(**kwargs_)
    main(**kwargs_)
