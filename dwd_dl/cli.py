"""This submodule handles the cli-like utilities such as the parser

"""

import argparse


class RadolanParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description="Training U-Net model for segmentation of RADOLAN precipitation images"
        )
        self.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="device for training (default: cuda:0)",
        )
        self.add_argument(
            "--workers",
            type=int,
            default=4,
            help="number of workers for data loading (default: 4)",
        )
        self.add_argument(
            "--weights", type=str, default="./weights", help="folder to save weights"
        )
        self.add_argument(
            "--logs", type=str, default="./logs", help="folder to save logs"
        )
        self.add_argument(
            "--images", type=str, default="../Radolan", help="root folder with images"
        )
        self.add_argument(
            "--save",
            type=bool,
            default=False,
            help="Save the Unet at the end of training. Path is RADOLAN_PATH/Models/RUN-TIMESTAMP"
        )
        self.add_argument(
            "--verbose",
            type=bool,
            default=False,
            help="Verbose setting. Either true or false."
        )


class DeprecatedRadolanParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description="OldArgparseArguments"
        )

        self.add_argument(
            "--batch-size",
            type=int,
            default=6,
            help="input batch size for training (default: 6)",
        )


        self.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="number of epochs to train (default: 100)",
        )

        self.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="initial learning rate (default: 0.001)",
        )
        self.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="device for training (default: cuda:0)",
        )
        self.add_argument(
            "--workers",
            type=int,
            default=4,
            help="number of workers for data loading (default: 4)",
        )
        self.add_argument(
            "--vis-images",
            type=int,
            default=200,
            help="number of visualization images to save in log file (default: 200)",
        )
        self.add_argument(
            "--vis-freq",
            type=int,
            default=10,
            help="frequency of saving images to log file (default: 10)",
        )
        self.add_argument(
            "--weights", type=str, default="./weights", help="folder to save weights"
        )
        self.add_argument(
            "--logs", type=str, default="./logs", help="folder to save logs"
        )
        self.add_argument(
            "--images", type=str, default="../Radolan", help="root folder with images"
        )
        self.add_argument(
            "--image-size",
            type=int,
            default=256,
            help="target input image size (default: 256)",
        )
        self.add_argument(
            "--aug-scale",
            type=int,
            default=0.05,
            help="scale factor range for augmentation (default: 0.05)",
        )
        self.add_argument(
            "--save",
            type=bool,
            default=False,
            help="Save the Unet at the end of training. Path is RADOLAN_PATH/Models/RUN-TIMESTAMP"
        )
        self.add_argument(
            "--cat",
            type=bool,
            default=False,
            help="Whether the skips should be implemented with torch.cat or with a simple sum. "
                 "False (default) means sum."
        )
        self.add_argument(
            "--verbose",
            type=bool,
            default=False,
            help="Verbose setting. Either true or false."
        )
        self.add_argument(
            "--filename",
            type=str,
            default='no_name.h5',
            help='Name for the h5 file.'
        )

