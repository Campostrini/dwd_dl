from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


class UNetLitModel(pl.LightningModule):

    def __init__(self, in_channels=6, out_channels=1, init_features=32, permute_output=True, softmax_output=False,
                 conv_bias=False, depth=7, cat=False, classes=4, lr=1e-3, batch_size=6, image_size=256, num_workers=4,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters(
            'in_channels',
            'out_channels',
            'init_features',
            'conv_bias',
            'depth',
            'cat',
            'lr',
            'batch_size',
            'image_size',
        )
        self.workers = num_workers
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset = None
        self.valid_dataset = None
        self.train_dataset = None
        self.init_features = init_features
        lon_lat_channels = 2
        timestamp_channel = 1
        image_channel = 1
        self._in_channels = in_channels * (image_channel + timestamp_channel + lon_lat_channels)
        self._out_channels = out_channels
        self._permute_output = permute_output
        self._softmax = softmax_output
        self._conv_bias = conv_bias
        self._depth = depth
        self._classes = classes
        self._lr = lr
        sizes = [self.init_features * 2 ** n for n in range(depth)]
        if not cat:
            sizes_in_down = sizes.copy()
            sizes_in_down.insert(0, self._in_channels)
            del sizes_in_down[-1]
            sizes_out_down = sizes.copy()

            sizes_in_up = sizes.copy()
            sizes_out_up = sizes.copy()
            sizes_out_up.insert(0, out_channels * classes)
            del sizes_out_up[-1]
        else:
            sizes_in_down = [2 * i for i in sizes]
            sizes_in_down.insert(0, self._in_channels * 2)
            del sizes_in_down[-1]
            sizes_out_down = sizes.copy()

            sizes_in_up = [3 * i for i in sizes]
            sizes_out_up = sizes.copy()
            sizes_out_up.insert(0, out_channels * classes)
            del sizes_out_up[-1]

        self._sizes_in_down = sizes_in_down
        self._sizes_out_down = sizes_out_down
        self._sizes_in_up = sizes_in_up
        self._sizes_out_up = sizes_out_up
        self._sizes = sizes
        self._cat = cat

        self.basic1 = self._basic_block(
            in_channels=self._in_channels,
            out_channels=self._in_channels,
            name="basic1",
            conv_bias=self._conv_bias,
        )

        self.encoder = nn.ModuleList([
            self._downsample_block(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_down, sizes_out_down, (f"down{i + 1}" for i in range(self._depth))
            )
        ])

        self.down_skips = nn.ModuleList([
            self._downskip(in_channels=in_channels_, out_channels=out_channels_, name=name_, conv_bias=conv_bias) for
            in_channels_, out_channels_, name_ in zip(
                sizes_in_down, sizes_out_down, (f"down_skip{i + 1}" for i in range(self._depth))
            )
        ])

        self.bottleneck = self._basic_block(
            in_channels=sizes[-1] * 2 ** self._cat,  # avoids extra logic. Doubles in_channels if cat.
            out_channels=sizes[-1],
            name="bottleneck",
            conv_bias=self._conv_bias,
        )

        self.decoder = nn.ModuleList([
            self._upsample_block(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_up, sizes_out_up, (f"up{i + 1}" for i in range(self._depth))
            )
        ])

        self.up_skips = nn.ModuleList([
            self._upskip(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_up, sizes_out_up, (f"up_skip{i + 1}" for i in range(self._depth))
            )
        ])

        self.softmax = nn.Softmax(dim=2)  # probably not right!

    @staticmethod
    def _block(in_channels, features, name, conv_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.01)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features, momentum=0.01)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _basic_block(in_channels, out_channels, name, conv_bias):
        """Basic block, in_channels should be equal to out_channels

        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=out_channels, momentum=0.01)),
                    (name + "leakyrelu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        )
                    )
                ]
            )
        )

    @staticmethod
    def _downsample_block(in_channels, out_channels, name, conv_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels, momentum=0.01)),
                    (name + "leakyrelu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (name + "maxpool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                    (name + "norm2", nn.BatchNorm2d(num_features=in_channels, momentum=0.01)),
                    (name + "leakyrelu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        )
                    )
                ]
            )
        )

    @staticmethod
    def _upsample_block(in_channels, out_channels, name, conv_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "upsample",
                        nn.Upsample(
                            scale_factor=2,
                            mode='nearest',
                        )
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels, momentum=0.01)),
                    (name + "leakyrelu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        )
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=out_channels, momentum=0.01)),
                    (name + "leakyrelu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=conv_bias,
                        )
                    ),
                ]
            )
        )

    @staticmethod
    def _downskip(in_channels, out_channels, name, conv_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=conv_bias,
                        )
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=out_channels, momentum=0.01))
                ]
            )
        )

    @staticmethod
    def _upskip(in_channels, out_channels, name, conv_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "upsample",
                        nn.Upsample(
                            scale_factor=2,
                            mode="nearest"
                        )
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=in_channels, momentum=0.01)),
                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=conv_bias
                        )
                    )
                ]
            )
        )

    def sum_or_cat(self, *args, dim=1, **kwargs):
        if self._cat:
            return torch.cat(args, dim=dim, **kwargs)
        else:
            return torch.stack(args, dim=0).sum(dim=0)

    def forward(self, x):
        basic1 = self.basic1(x)
        x = self.sum_or_cat(basic1, x)

        trace = []

        for encoding_layer, down_skip_layer in zip(self.encoder, self.down_skips):
            x_encoded = encoding_layer(x)
            trace.append(x_encoded)
            x = self.sum_or_cat(x_encoded, down_skip_layer(x))

        trace[-1] = x
        x = self.bottleneck(x)

        for decoding_layer, up_skip_layer, trace_ in zip(
                reversed(self.decoder), reversed(self.up_skips), reversed(trace)
        ):
            x = self.sum_or_cat(x, trace_)
            x = self.sum_or_cat(decoding_layer(x), up_skip_layer(x))

        # sum at the end is always needed otherwise we have too many channels.
        if self._cat:
            x_1, x_2 = torch.split(x, x.size()[1] // 2, dim=1)
            x = x_1 + x_2

        x = x.reshape(x.shape[0], self._out_channels, 4, *x.shape[-2:])

        if self._softmax:
            x = self.softmax(x)
        if self._permute_output:
            x = x.permute(0, 2, 1, 3, 4)

        return x

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true[:, ::4, ...].to(dtype=torch.long)
        y_pred = self(x)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(y_pred, y_true)

        train_acc = torch.sum(y_true == torch.argmax(y_pred, dim=1)).item() / torch.numel(y_true)

        self.log_dict({'train_loss': loss, 'train_accuracy': train_acc})

        return {'loss': loss, 'train_acc': train_acc}

    def training_epoch_end(self, outputs):
        train_loss = float(sum([batch['loss'] for batch in outputs]) / len(outputs))
        train_acc = float(sum([batch['train_acc'] for batch in outputs])) / len(outputs)
        self.log_dict({'epoch_train_loss': train_loss, 'epoch_train_acc': train_acc})
        self.logger.experiment.add_hparams(
            dict(self.hparams),
            {
                'hparam/train_loss': train_loss,
                'hparam/train_acc': train_acc,
            },
            )

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true[:, ::4, ...].to(dtype=torch.long)
        y_pred = self(x)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(y_pred, y_true)

        val_acc = torch.sum(y_true == torch.argmax(y_pred, dim=1)).item() / torch.numel(y_true)

        self.log_dict({'valid_loss': loss, 'valid_accuracy': val_acc})
        return {'loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss = float(sum([batch['loss'] for batch in outputs]) / len(outputs))
        val_acc = float(sum([batch['val_acc'] for batch in outputs])) / len(outputs)
        self.log_dict({'epoch_val_loss': val_loss, 'epoch_val_acc': val_acc})
        self.logger.experiment.add_hparams(
            dict(self.hparams),
            {
                'hparam/val_loss': val_loss,
                'hparam/val_acc': val_acc,
            }
        )

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self._lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument(
            "--in-channels",
            type=int,
            default=6,
            help="Number of input channels. (default: 6)",
        )
        parser.add_argument(
            "--out-channels",
            type=int,
            default=1,
            help="Number of output channels. (default: 1)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=6,
            help="Input batch size for training. (default: 6)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of epochs to train. (default: 100)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Initial learning rate. (default: 0.001)",
        )
        parser.add_argument(
            "--image-size",
            type=int,
            default=256,
            help="Target input image size. (default: 256)",
        )
        parser.add_argument(
            "--cat",
            type=bool,
            default=False,
            help="Whether the skips should be implemented with torch.cat or with a simple sum. "
                 "False (default) means sum."
        )
        parser.add_argument(
            "--init-features",
            type=int,
            default=32,
            help="Number of features fo the first convolutional layer. (default: 32)"
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=7,
            help="Number of layer-groups of the UNet. (default: 7)"
        )
        parser.add_argument(
            "--conv-bias",
            type=bool,
            default=False,
            help="Use bias in the convolutions. Might have huge memory footprint. (default: False)"
        )
        return parent_parser


class RadolanLiveEvaluator(UNetLitModel):
    def __init__(self, dm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = dm

    def which_dataset(self, *args):
        return self.dm.which_dataset(*args)

    @property
    def legal_timestamps(self, *args, **kwargs):
        return self.dm.legal_timestamps

    def on_timestamp(self, timestamp):
        x, y_true = self.dm.dataset.from_timestamp(timestamp)
        x, y_true = torch.unsqueeze(x, 0), torch.unsqueeze(y_true, 0)
        y = self(x)
        x, y, y_true = x[:, ::4], torch.argmax(y, dim=1), y_true[:, ::4]
        return x.cpu().numpy(), y.cpu().numpy(), y_true.cpu().numpy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(RadolanLiveEvaluator, RadolanLiveEvaluator).add_model_specific_args(parent_parser)
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="The path to the saved model."
        )
        return parent_parser

    @auto_move_data
    def forward(self, x):
        return super().forward(x)