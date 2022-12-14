from collections import OrderedDict

import torch
import torch.nn as nn

from dwd_dl import utils


class UNet(nn.Module):

    def __init__(self, in_channels=6, out_channels=1, init_features=32, permute_output=True, softmax_output=False,
                 conv_bias=False, depth=7, cat=False, classes=4):
        super(UNet, self).__init__()

        features = init_features
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
        sizes = [init_features * 2**n for n in range(depth)]
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

        self.basic1 = UNet._basic_block(
            in_channels=self._in_channels,
            out_channels=self._in_channels,
            name="basic1",
            conv_bias=self._conv_bias,
        )

        self.encoder = nn.ModuleList([
            UNet._downsample_block(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_down, sizes_out_down, (f"down{i+1}" for i in range(self._depth))
            )
        ])

        self.down_skips = nn.ModuleList([
            UNet._downskip(in_channels=in_channels_, out_channels=out_channels_, name=name_, conv_bias=conv_bias) for in_channels_, out_channels_, name_ in zip(
                sizes_in_down, sizes_out_down, (f"down_skip{i+1}" for i in range(self._depth))
            )
        ])

        self.bottleneck = UNet._basic_block(
            in_channels=sizes[-1] * 2**self._cat,  # avoids extra logic. Doubles in_channels if cat.
            out_channels=sizes[-1],
            name="bottleneck",
            conv_bias=self._conv_bias,
        )

        self.decoder = nn.ModuleList([
            UNet._upsample_block(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_up, sizes_out_up, (f"up{i+1}" for i in range(self._depth))
            )
        ])

        self.up_skips = nn.ModuleList([
            UNet._upskip(
                in_channels=in_channels_,
                out_channels=out_channels_,
                name=name_,
                conv_bias=conv_bias,
            ) for in_channels_, out_channels_, name_ in zip(
                sizes_in_up, sizes_out_up, (f"up_skip{i+1}" for i in range(self._depth))
            )
        ])

        self.softmax = nn.Softmax(dim=2)  # probably not right!

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
            x_1, x_2 = torch.split(x, x.size()[1]//2, dim=1)
            x = x_1 + x_2
            
        x = x.reshape(x.shape[0], self._out_channels, 4, *x.shape[-2:])
        
        if self._softmax:
            x = self.softmax(x)
        if self._permute_output:
            x = x.permute(0, 2, 1, 3, 4)

        return x

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
