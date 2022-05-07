import itertools
import os
import datetime as dt
from collections import OrderedDict
from itertools import product

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection

import dwd_dl.cfg as cfg
from dwd_dl.metrics import (
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
    TruePositiveRatio,
    TrueNegativeRatio,
    FalsePositiveRatio,
    FalseNegativeRatio,
    PercentCorrect,
    HitRate,
    FalseAlarmRatio,
    CriticalSuccessIndex,
    Bias,
    HeidkeSkillScore,
    ConfusionMatrix,
    Precision,
    Recall,
    F1,
    Contingency,
    ConfusionMatrixScikit
)
from dwd_dl import log


class UNetLitModel(pl.LightningModule):

    def __init__(self, in_channels=6, out_channels=1, init_features=32, permute_output=True, softmax_output=False,
                 conv_bias=False, depth=7, cat=False, classes=4, lr=1, batch_size=6, image_size=256, num_workers=4,
                 timestamp_string=None, transformation='log_sum',
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
        self.init_features = int(init_features)
        lon_lat_channels = 2
        time_of_day_channel = 1
        day_of_year_channel = 1
        image_channel = 1
        self._in_channels = in_channels * (image_channel + time_of_day_channel + day_of_year_channel + lon_lat_channels)
        self._out_channels = out_channels
        self._permute_output = permute_output
        self._softmax = softmax_output
        self._conv_bias = conv_bias
        self._depth = int(depth)
        self._classes = classes

        if transformation == 'log':
            def transform(x, **kwargs_):
                x[:, ::5, ...] = torch.log(x[:, ::5, ...], **kwargs_)
                return x
            self._transform = transform
        elif transformation == 'log_sum':
            def transform(x, **kwargs_):
                x[:, ::5, ...] = torch.log(x[:, ::5, ...] + 0.01, **kwargs_)
                return x
            self._transform = transform
        else:
            def transform(x, **kwargs_):
                return x
            self._transform = transform

        self.lr = lr
        self.cel_weights = torch.tensor([1/95, 1/4, 1/1, 1/0.7])
        sizes = [self.init_features * 2 ** n for n in range(self._depth)]

        self._metrics_to_include = [
            TruePositive,
            TrueNegative,
            FalsePositive,
            FalseNegative,
            TruePositiveRatio,
            TrueNegativeRatio,
            FalsePositiveRatio,
            FalseNegativeRatio,
            PercentCorrect,
            HitRate,
            FalseAlarmRatio,
            CriticalSuccessIndex,
            Bias,
            HeidkeSkillScore,
        ]
        self._multiclass_metrics = [
            F1, Precision, Recall
        ]
        self.metrics, self.persistence_metrics = self._initialize_metrics(
            self._metrics_to_include, multiclass_metrics=self._multiclass_metrics, confusion_matrix=True,
        )
        self.test_metrics, self.test_persistence_metrics = self._initialize_metrics(
            self._metrics_to_include, multiclass_metrics=self._multiclass_metrics, test=True, confusion_matrix=True,
        )

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

        self.timestamp_string = timestamp_string

        self.initial_transform = self._initial_transform(self._transform)

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
                self._sizes_in_down, self._sizes_out_down, (f"down{i + 1}" for i in range(self._depth))
            )
        ])

        self.down_skips = nn.ModuleList([
            self._downskip(in_channels=in_channels_, out_channels=out_channels_, name=name_, conv_bias=conv_bias) for
            in_channels_, out_channels_, name_ in zip(
                self._sizes_in_down, self._sizes_out_down, (f"down_skip{i + 1}" for i in range(self._depth))
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

        self.apply(self.initialize_weights)

        self._loss_weights = torch.tensor(cfg.CFG.WEIGHTS, device=self.device, dtype=torch.float)
        # self.log_softmax = nn.LogSoftmax()
        # self.loss = torch.nn.NLLLoss(weight=self._loss_weights)
        self.loss = torch.nn.CrossEntropyLoss(weight=self._loss_weights)

        # self._check_no_overlap_one_example = None
        # self._count_overlaps = 0

    @staticmethod
    def initialize_weights(layer: nn.Module):
        kwargs = {
            'a': 0.01,
            'nonlinearity': 'leaky_relu'
        }
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, **kwargs)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    @staticmethod
    def _initial_transform(transformation):
        return nn.Sequential(
            OrderedDict(
                [
                    ("initial_transformation", TransformModule(transformation=transformation))
                ]
            )
        )

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
                    (name + "norm1", nn.BatchNorm2d(num_features=features,)),
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
                    (name + "norm2", nn.BatchNorm2d(num_features=features,)),
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
                    (name + "norm1", nn.BatchNorm2d(num_features=out_channels,)),
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
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels,)),
                    (name + "leakyrelu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (name + "maxpool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                    (name + "norm2", nn.BatchNorm2d(num_features=in_channels,)),
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
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels,)),
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
                    (name + "norm2", nn.BatchNorm2d(num_features=out_channels,)),
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
                            kernel_size=1,
                            stride=1,
                            bias=conv_bias,
                        )
                    ),
                    (
                        name + "avgpool",
                        nn.AvgPool2d(kernel_size=2, stride=2)
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=out_channels,))
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
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=out_channels,)),
                ]
            )
        )

    def sum_or_cat(self, *args, dim=1, **kwargs):
        if self._cat:
            return torch.cat(args, dim=dim, **kwargs)
        else:
            return torch.stack(args, dim=0).sum(dim=0)

    # def forward(self, x):
    #     x = self.initial_transform(x)
    #     basic1 = self.basic1(x)
    #     x = self.sum_or_cat(basic1, x)
    #
    #     trace = []
    #
    #     for encoding_layer, down_skip_layer in zip(self.encoder, self.down_skips):
    #         x_encoded = encoding_layer(x)
    #         trace.append(x_encoded)
    #         x = self.sum_or_cat(x_encoded, down_skip_layer(x))
    #
    #     trace[-1] = x
    #     x = self.bottleneck(x)
    #
    #     for decoding_layer, up_skip_layer, trace_ in zip(
    #             reversed(self.decoder), reversed(self.up_skips), reversed(trace)
    #     ):
    #         x = self.sum_or_cat(x, trace_)
    #         x = self.sum_or_cat(decoding_layer(x), up_skip_layer(x))
    #
    #     # sum at the end is always needed otherwise we have too many channels.
    #     if self._cat:
    #         x_1, x_2 = torch.split(x, x.size()[1] // 2, dim=1)
    #         x = x_1 + x_2
    #
    #     x = torch.reshape(x, [x.shape[0], self._out_channels, self._classes, *x.shape[-2:]])
    #
    #     if self._softmax:
    #         x = self.softmax(x)
    #     if self._permute_output:
    #         x = x.permute(0, 2, 1, 3, 4)
    #
    #     return x

    def forward(self, x):
        x = self.initial_transform(x)
        basic1 = self.basic1(x)
        x = basic1 + x

        trace = []

        for encoding_layer, down_skip_layer in zip(self.encoder, self.down_skips):
            x_encoded = encoding_layer(x)
            trace.append(x_encoded)
            x = x_encoded + down_skip_layer(x)

        trace[-1] = x
        x = self.bottleneck(x)

        for decoding_layer, up_skip_layer, trace_ in zip(
                reversed(self.decoder), reversed(self.up_skips), reversed(trace)
        ):
            x = x + trace_
            x = decoding_layer(x) + up_skip_layer(x)

        # sum at the end is always needed otherwise we have too many channels.
        if self._cat:
            x_1, x_2 = torch.split(x, x.size()[1] // 2, dim=1)
            x = x_1 + x_2

        x = torch.reshape(x, [x.shape[0], self._out_channels, self._classes, *x.shape[-2:]])

        if self._softmax:
            x = self.softmax(x)
        if self._permute_output:
            x = x.permute(0, 2, 1, 3, 4)

        return x

    def training_step(self, batch, batch_idx):
        log.debug("Trainig step")
        x, y_true = batch
        y_true = y_true[:, ::5, ...].to(dtype=torch.long)
        y_pred = self(x)

        loss = self.loss(y_pred, y_true)

        train_acc = torch.sum(y_true == torch.argmax(y_pred, dim=1)).item() / torch.numel(y_true)

        self.log_dict({'train/loss': loss, 'train/accuracy': train_acc})
        self.log('lr', self.lr, True)

        # if self._check_no_overlap_one_example is not None:
        #     training_to_compare = y_true.cpu().numpy()
        #     log.info(f"{self._check_no_overlap_one_example.shape=} and {training_to_compare.shape=}")
        #     for b in self._check_no_overlap_one_example:
        #         for t in training_to_compare:
        #             sum_ = (b == t).sum()
        #             if sum_ > 256*256*0.9:
        #                 warnings.warn(f"Sum very high {sum=}")
        #             if sum_ == 256*256:
        #                 if b.sum() > 40:
        #                     self._count_overlaps += 1
        #                     log.info(f"{b.sum()=}")
        log.debug("Training Step end")
        # log.info(f"{self._count_overlaps=}")
        return {'loss': loss, 'train_acc': train_acc}

    def training_epoch_end(self, outputs):
        train_loss = float(sum([batch['loss'] for batch in outputs]) / len(outputs))
        train_acc = float(sum([batch['train_acc'] for batch in outputs])) / len(outputs)
        self.log_dict({'train/epoch_loss': train_loss, 'train/epoch_accuracy': train_acc})

    def validation_step(self, batch, batch_idx):
        log.debug("Validation Step.")
        x, y_true = batch
        y_true = y_true[:, ::5, ...].to(dtype=torch.long)
        y_pred = self(x)

        loss = self.loss(y_pred, y_true)

        val_acc = torch.sum(y_true == torch.argmax(y_pred, dim=1)).item() / torch.numel(y_true)

        self.log_dict({'val/loss': loss, 'val/accuracy': val_acc})
        self.log_dict({'hp/val_loss': loss, 'hp/val_accuracy': val_acc})

        log.debug("Validation Step End")
        return {'loss': loss, 'val_acc': val_acc, 'preds': y_pred, 'target': y_true, 'pers': x}

    def validation_step_end(self, outputs):
        self.metrics.update(*Contingency.format_input(outputs['preds'], outputs['target']))
        metrics_out = self.metrics.compute()
        self.persistence_metrics.update(*Contingency.format_input(outputs['pers'][:, -5, ...], outputs['target'], persistence=True))
        persistence_metrics_out = self.persistence_metrics.compute()
        self.log_dict(metrics_out)
        self.log_dict(persistence_metrics_out)

    def validation_epoch_end(self, outputs):
        # TODO: Move to callbacks when it's available
        # log.info(f"{outputs=}")
        val_loss = float(sum([batch['loss'] for batch in outputs]) / len(outputs))
        val_acc = float(sum([batch['val_acc'] for batch in outputs])) / len(outputs)
        self.log_dict({'val/epoch_loss': val_loss, 'val/epoch_accuracy': val_acc})
        self.last_confusion_matrix = self.metrics['ConfusionMatrixScikit'].confusion_matrix.cpu().numpy()
        self.metrics.reset()
        self.persistence_metrics.reset()
        log.info(f"{self.last_confusion_matrix=}")
        # self._reset_metrics()
        # log.info(f"{self.trainer.datamodule.dataset.tstracker._used_indices_training=}")
        # log.info(f"{self.trainer.datamodule.dataset.tstracker._used_indices_validation=}")
        # self.trainer.datamodule.dataset.tstracker.reset()
        # self.logger.experiment.add_hparams(
        #     dict(self.hparams),
        #     {
        #         'hparam/val_loss': val_loss,
        #         'hparam/val_acc': val_acc,
        #     }
        # )

    def on_validation_end(self) -> None:
        confusion_matrix_display = ConfusionMatrixDisplay(self.last_confusion_matrix)
        confusion_matrix_display.plot()
        log.info(f"{self.last_confusion_matrix=}")
        plt.savefig(os.path.join(cfg.CFG.RADOLAN_ROOT_RAW, 'confmatrix.png'))

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true[:, ::5, ...].to(dtype=torch.long)
        y_pred = self(x)

        loss = self.loss(y_pred, y_true)

        test_acc = torch.sum(y_true == torch.argmax(y_pred, dim=1)).item() / torch.numel(y_true)

        metrics_out = self.test_metrics(*Contingency.format_input(y_pred, y_true))
        persistence_metrics_out = self.test_persistence_metrics(*Contingency.format_input(x[:, -5, ...], y_true,
                                                                                          persistence=True))

        self.log_dict({'test/loss': loss, 'test/accuracy': test_acc})
        self.log_dict({'hp/test_loss': loss, 'hp/test_accuracy': test_acc})
        self.log_dict(metrics_out)
        self.log_dict(persistence_metrics_out)
        return {'loss': loss, 'test_acc': test_acc}

    def test_epoch_end(self, outputs):
        test_loss = float(sum([batch['loss'] for batch in outputs]) / len(outputs))
        test_acc = float(sum([batch['test_acc'] for batch in outputs])) / len(outputs)
        self.log_dict({'test/epoch_loss': test_loss, 'test/epoch_accuracy': test_acc})
        self._reset_metrics()

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y_true = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser, argument_group=False):
        if argument_group:
            parser = parent_parser.add_argument_group("UNet")
        else:
            parser = parent_parser
        parser.add_argument(
            "--in_channels",
            type=int,
            default=6,
            help="Number of input channels. (default: 6)",
        )
        parser.add_argument(
            "--out_channels",
            type=int,
            default=1,
            help="Number of output channels. (default: 1)"
        )
        parser.add_argument(
            "--batch_size",
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
            default=1,
            help="Initial learning rate. (default: 0.001)",
        )
        parser.add_argument(
            "--image_size",
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
            "--init_features",
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
            "--conv_bias",
            type=bool,
            default=False,
            help="Use bias in the convolutions. Might have huge memory footprint. (default: False)"
        )
        parser.add_argument(
            "--transformation",
            type=str,
            default='log_sum',
            help="The transformation applied to each input. Either log or log_sum. (default: None)"
        )
        parser.add_argument(
            "--dask",
            type=bool,
            default=False,
            help="Whether to use Dask or a pure Zarr implementation of the Radolan Datset class."
                 "It might solve multiprocessing issues."
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="The path to the saved model."
        )
        if argument_group:
            return parent_parser
        else:
            return parser

    def assign_timestamp_string_from_checkpoint_path(self, checkpoint_path):
        name = os.path.split(checkpoint_path)[-1]
        name = name.replace('.ckpt', '')
        string = name.split('_')[0]
        try:
            dt.datetime.strptime(string, cfg.CFG.TIMESTAMP_DATE_FORMAT)
            self.timestamp_string = string
        except ValueError:
            pass

    def _initialize_metrics(self, metrics_to_include, multiclass_metrics, test=False, confusion_matrix=True,):
        test_prefix = ''
        if test:
            test_prefix = 'test/'
        mc = MetricCollection({
            f'{test_prefix}{metric.__name__}/{model_}{class_number}': metric(
                class_number) for model_, class_number, metric in product(
                ('',), range(self._classes), metrics_to_include
            )
        })
        if confusion_matrix:
            mc.add_metrics({
                f'{test_prefix}{ConfusionMatrix.__name__}': ConfusionMatrix(self._classes),
            })
            mc.add_metrics({
                f'{test_prefix}{ConfusionMatrixScikit.__name__}': ConfusionMatrixScikit(),
            })
        mc.add_metrics({
            f'{test_prefix}{metric.__name__}{average}': metric(
                self._classes, average=average
            ) for metric, average in product(
                multiclass_metrics, ('micro', 'macro', 'weighted')
            )
        })

        # PersistenceMetricCollection
        pmc = MetricCollection({
            f'{test_prefix}{metric.__name__}/{model_}{class_number}': metric(
                class_number, persistence_as_metric=True) for model_, class_number, metric in product(
                ('persistence_',), range(self._classes), metrics_to_include
            )
        })
        if confusion_matrix:
            pmc.add_metrics({
                f'{test_prefix}{ConfusionMatrix.__name__}/persistence': ConfusionMatrix(self._classes),
            })
            pmc.add_metrics({
                f'{test_prefix}{ConfusionMatrixScikit.__name__}/persistence': ConfusionMatrixScikit(True),
            })
        pmc.add_metrics({
            f'{test_prefix}{metric.__name__}{average}/persistence': metric(
                self._classes, average=average
            ) for metric, average in product(
                multiclass_metrics, ('micro', 'macro', 'weighted')
            )
        })

        for metric in pmc:
            pmc[metric].persistence_as_metric = True

        return mc, pmc

    def _reset_metrics(self, test=False):
        if not test:
            to_reset = (self.metrics, self.persistence_metrics)
        else:
            to_reset = (self.test_metrics, self.test_persistence_metrics)
        for metric_dict in to_reset:
            for metric in metric_dict:
                metric_dict[metric].reset()


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
        x, y, y_true = x[:, ::5], torch.argmax(y, dim=1), y_true[:, ::5]
        return x.cpu().numpy(), y.cpu().numpy(), y_true.cpu().numpy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(RadolanLiveEvaluator, RadolanLiveEvaluator).add_model_specific_args(parent_parser)
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="The path to the saved model."
        )
        return parent_parser

    def forward(self, x):
        return super().forward(x)


class TransformModule(nn.Module):
    def __init__(self, transformation):
        super().__init__()
        self._transform = transformation

    def forward(self, x):
        x = self._transform(x)
        return x
