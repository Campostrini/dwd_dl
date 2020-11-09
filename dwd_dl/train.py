import argparse
import json
import os
import datetime as dt
from contextlib import contextmanager

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from dwd_dl.dataset import RadolanDataset as Dataset, RadolanSubset as Subset, create_h5
from dwd_dl.unet import UNet
from dwd_dl import cfg
from dwd_dl import utils
from dwd_dl.cli import RadolanParser
import dwd_dl as dl


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    verbose = args.verbose

    with data_loaders(args) as loaders_:
        loader_train, loader_valid = loaders_
        loaders = {"train": loader_train, "valid": loader_valid}

        unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        unet.to(device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        optimizer = optim.Adadelta(unet.parameters(), lr=args.lr)

        loss_train = []
        loss_valid = []
        run = str(dt.datetime.now())
        writer = SummaryWriter(os.path.join(cfg.CFG.RADOLAN_ROOT, 'Logs', run))

        step = 0

        for epoch in tqdm(range(args.epochs), total=args.epochs):
            epoch_loss_train = []
            epoch_loss_valid = []
            epoch_accuracy_train = 0
            total_elements_train = 0
            epoch_accuracy_valid = 0
            total_elements_valid = 0
            for phase in ["train", "valid"]:
                if phase == "train":
                    unet.train()
                else:
                    unet.eval()

                for i, data in enumerate(loaders[phase]):
                    if phase == "train":
                        step += 1

                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = unet(x)

                        loss = cross_entropy_loss(y_pred, utils.to_class_index(y_true))

                        batch_elements = 1
                        for dim_ in y_true.shape:
                            batch_elements *= dim_
                        batch_elements /= 4  # number of classes

                        if phase == "valid":
                            loss_valid.append(loss.item())
                            epoch_loss_valid.append(loss.item())
                            y_pred = torch.topk(y_pred, 1, dim=1).indices
                            correct = (y_pred == y_true).float().sum()
                            epoch_accuracy_valid += correct
                            total_elements_valid += batch_elements

                        if phase == "train":
                            loss_train.append(loss.item())
                            epoch_loss_train.append(loss.item())
                            loss.backward()
                            optimizer.step()
                            total_elements_train += batch_elements
                            y_pred = torch.topk(y_pred, 1, dim=1).indices
                            correct = (y_pred == y_true).float().sum()
                            epoch_accuracy_train += correct

                    if phase == "train" and (step + 1) % 10 == 0:
                        if verbose:
                            print(loss_train, step)
                            print('In Training.')
                        loss_train = []

                if phase == "valid":
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

        if args.save:
            saved_name_path = utils.unet_saver(
                unet,
                path=os.path.join(cfg.CFG.RADOLAN_ROOT, 'Models', run),
                timestamp=run
            )
            print('Saved Unet state_dict: {}'.format(saved_name_path))


@contextmanager
def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    weighted_rand_sampler = WeightedRandomSampler(
        weights=dataset_train.weights,
        num_samples=len(dataset_train),
        replacement=False
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        sampler=weighted_rand_sampler
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init
    )

    try:
        yield loader_train, loader_valid

    finally:
        loader_train.dataset.dataset.file_handle.close()
        loader_valid.dataset.dataset.file_handle.close()


def datasets(args):
    f = create_h5(args.filename)
    dataset = Dataset(
        h5file_handle=f,
        date_ranges_path=cfg.CFG.DATE_RANGES_FILE_PATH,
        image_size=args.image_size
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


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


def matplotlib_imshow(img, mean, std, writer, cols=6, rows=12):
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    axes = []
    plt.figure(figsize=(rows, cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace=0.025, hspace=0.025)
    for i in range(rows):
        for j in range(cols):
            ax1 = plt.subplot(gs1[i*j])
            # axes.append(fig.add_subplot(rows, cols, (i * j) + 1))
            ax1.imshow(npimg[i, j])
            writer.add_image('Batch {}, seq {}'.format(i+1, j+1), npimg[i, j], dataformats='HW')

    plt.show()


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    args = parser.parse_args()
    main(args)
