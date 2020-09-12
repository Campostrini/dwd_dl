import argparse
import json
import os
import datetime as dt

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from dwd_dl.dataset import RadolanDataset as Dataset
# from logger import Logger
# from loss import DiceLoss
# from transform import transforms
from dwd_dl.unet import UNet
# from utils import log_images, dsc
from dwd_dl import config
from dwd_dl import utils
import dwd_dl as dl


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    mse_loss = torch.nn.MSELoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adadelta(unet.parameters(), lr=args.lr)

    # logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    run = str(dt.datetime.now())
    writer = SummaryWriter(os.path.join(config.RADOLAN_PATH, 'Logs', run))

    dataiter = iter(loader_train)
    past_seq, true_next = next(dataiter)

#    matplotlib_imshow(past_seq, loader_train.dataset.mean, loader_train.dataset.std, writer)

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        epoch_loss_train = []
        epoch_loss_valid = []
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = mse_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        epoch_loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         # TODO : Need to log images.
                        #         print('Logging Images needed in validation.')

                    if phase == "train":
                        loss_train.append(loss.item())
                        epoch_loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    print(loss_train, step)
                    print('In Training.')
                    loss_train = []

            if phase == "valid":
                print(loss_valid, step)
                print('In Validation.')
                # logger.scalar_summary("val_dsc", mean_dsc, step)

                # TODO: Add saving weights
                # if mean_dsc > best_validation_dsc:
                #     best_validation_dsc = mean_dsc
                #     torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []
        writer.add_scalar('Loss/train', np.array(epoch_loss_train).mean(), epoch)
        writer.add_scalar('Loss/valid', np.array(epoch_loss_valid).mean(), epoch)
        writer.flush()
    if args.save:
        saved_name_path = utils.unet_saver(
            unet,
            path=os.path.join(os.path.abspath(config.RADOLAN_PATH), 'Models', run),
            timestamp=run
        )
        print('Saved Unet state_dict: {}'.format(saved_name_path))


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

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        date_ranges_path=config.DATE_RANGES_PATH,
        radolan_dir=args.images,
        subset="train",
        image_size=args.image_size
    )
    valid = Dataset(
        date_ranges_path=config.DATE_RANGES_PATH,
        radolan_dir=args.images,
        subset="validation",
        image_size=args.image_size
    )
    return train, valid


# def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
#     dsc_list = []
#     num_slices = np.bincount([p[0] for p in patient_slice_index])
#     index = 0
#     for p in range(len(num_slices)):
#         y_pred = np.array(validation_pred[index : index + num_slices[p]])
#         y_true = np.array(validation_true[index : index + num_slices[p]])
#         dsc_list.append(dsc(y_pred, y_true))
#         index += num_slices[p]
#     return dsc_list


# def log_loss_summary(logger, loss, step, prefix=""):
#     logger.scalar_summary(prefix + "loss", np.mean(loss), step)


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
    dl.config.config_initializer('..')
    print('Initializer Run')
#    dl.config.download_and_extract()
#    dl.config.clean_unused()
    print('Cleaning up unused.')
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="input batch size for training (default: 12)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="../Radolan", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Save the Unet at the end of training. Path is RADOLAN_PATH/Models/RUN-TIMESTAMP"
    )
    args = parser.parse_args()
    main(args)
