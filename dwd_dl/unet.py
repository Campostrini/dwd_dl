from collections import OrderedDict

import torch
import torch.nn as nn

from . import utils


class UNet(nn.Module):

    def __init__(self, in_channels=6, out_channels=1, init_features=32, permute_output=True, softmax_output=False,
                 conv_bias=False):
        super(UNet, self).__init__()

        features = init_features
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._permute_output = permute_output
        self._softmax = softmax_output
        self._conv_bias = conv_bias

        self.basic1 = UNet._basic_block(in_channels=in_channels, out_channels=in_channels, name="basic1",
                                        conv_bias=self._conv_bias)
        self.down1 = UNet._downsample_block(in_channels=in_channels, out_channels=features, name="down1",
                                            conv_bias=self._conv_bias)
        self.down_skip1 = UNet._skip(in_channels=in_channels, out_channels=features, name="down_skip1",
                                     conv_bias=self._conv_bias)
        self.down2 = UNet._downsample_block(in_channels=features, out_channels=features * 2, name="down2",
                                            conv_bias=self._conv_bias)
        self.down_skip2 = UNet._skip(in_channels=features, out_channels=features*2, name="down_skip2",
                                     conv_bias=self._conv_bias)
        self.down3 = UNet._downsample_block(in_channels=features * 2, out_channels=features * 4, name="down3",
                                            conv_bias=self._conv_bias)
        self.down_skip3 = UNet._skip(in_channels=features * 2, out_channels=features * 4, name="down_skip3",
                                     conv_bias=self._conv_bias)
        self.down4 = UNet._downsample_block(in_channels=features * 4, out_channels=features * 8, name="down4",
                                            conv_bias=self._conv_bias)
        self.down_skip4 = UNet._skip(in_channels=features * 4, out_channels=features * 8, name="down_skip4",
                                     conv_bias=self._conv_bias)
        self.down5 = UNet._downsample_block(in_channels=features * 8, out_channels=features * 16, name="down5",
                                            conv_bias=self._conv_bias)
        self.down_skip5 = UNet._skip(in_channels=features * 8, out_channels=features * 16, name="down_skip5",
                                     conv_bias=self._conv_bias)
        self.down6 = UNet._downsample_block(in_channels=features * 16, out_channels=features * 32, name="down6",
                                            conv_bias=self._conv_bias)
        self.down_skip6 = UNet._skip(in_channels=features * 16, out_channels=features * 32, name="down_skip6",
                                     conv_bias=self._conv_bias)
        self.down7 = UNet._downsample_block(in_channels=features * 32, out_channels=features * 64, name="down7",
                                            conv_bias=self._conv_bias)
        self.down_skip7 = UNet._skip(in_channels=features * 32, out_channels=features * 64, name="down_skip7",
                                     conv_bias=self._conv_bias)

        self.bneck = UNet._basic_block(in_channels=features * 64, out_channels=features * 64, name="bneck",
                                       conv_bias=self._conv_bias)

        self.up7 = UNet._upsample_block(in_channels=(features * 64) * 2, out_channels=(features * 32) * 2, name="up7",
                                        conv_bias=self._conv_bias)
        self.up_skip7 = UNet._upskip(in_channels=(features * 64) * 2, out_channels=(features*32) * 2, name="up_skip7",
                                     conv_bias=self._conv_bias)
        self.up6 = UNet._upsample_block(in_channels=(features * 32) * 3, out_channels=(features * 16) * 2, name="up6",
                                        conv_bias=self._conv_bias)
        self.up_skip6 = UNet._upskip(in_channels=(features * 32) * 3, out_channels=(features*16) * 2, name="up_skip6",
                                     conv_bias=self._conv_bias)
        self.up5 = UNet._upsample_block(in_channels=(features * 16) * 3, out_channels=(features * 8) * 2, name="up5",
                                        conv_bias=self._conv_bias)
        self.up_skip5 = UNet._upskip(in_channels=(features * 16) * 3, out_channels=(features*8) * 2, name="up_skip5",
                                     conv_bias=self._conv_bias)
        self.up4 = UNet._upsample_block(in_channels=(features * 8) * 3, out_channels=(features * 4) * 2, name="up4",
                                        conv_bias=self._conv_bias)
        self.up_skip4 = UNet._upskip(in_channels=(features * 8) * 3, out_channels=(features*4) * 2, name="up_skip4",
                                     conv_bias=self._conv_bias)
        self.up3 = UNet._upsample_block(in_channels=(features * 4) * 3, out_channels=(features * 2) * 2, name="up3",
                                        conv_bias=self._conv_bias)
        self.up_skip3 = UNet._upskip(in_channels=(features * 4) * 3, out_channels=(features*2) * 2, name="up_skip3",
                                     conv_bias=self._conv_bias)
        self.up2 = UNet._upsample_block(in_channels=(features * 2) * 3, out_channels=(features * 1) * 2, name="up2",
                                        conv_bias=self._conv_bias)
        self.up_skip2 = UNet._upskip(in_channels=(features * 2) * 3, out_channels=(features*1) * 2, name="up_skip2",
                                     conv_bias=self._conv_bias)
        self.up1 = UNet._upsample_block(in_channels=(features * 1) * 3, out_channels=out_channels * 4, name="up1",
                                        conv_bias=self._conv_bias)
        self.up_skip1 = UNet._upskip(in_channels=(features * 1) * 3, out_channels=out_channels * 4, name="up_skip1",
                                     conv_bias=self._conv_bias)

        self.softmax = nn.Softmax(dim=2)  # probably not right!

    def forward(self, x):

        x = utils.to_class_index(x, dtype=torch.float)

        basic1 = self.basic1(x)
        basic1 = basic1 + x  # Other way to implement this skip?

        down1 = self.down1(basic1)
        down_skip1 = self.down_skip1(basic1)

        down2 = self.down2(down1 + down_skip1)
        down_skip2 = self.down_skip2(down1 + down_skip1)

        down3 = self.down3(down2 + down_skip2)
        down_skip3 = self.down_skip3(down2 + down_skip2)

        down4 = self.down4(down3 + down_skip3)
        down_skip4 = self.down_skip4(down3 + down_skip3)

        down5 = self.down5(down4 + down_skip4)
        down_skip5 = self.down_skip5(down4 + down_skip4)

        down6 = self.down6(down5 + down_skip5)
        down_skip6 = self.down_skip6(down5 + down_skip5)

        down7 = self.down7(down6 + down_skip6)
        down_skip7 = self.down_skip7(down6 + down_skip6)

        bneck = self.bneck(down7 + down_skip7)
        bneck = bneck + down7 + down_skip7

        up7 = self.up7(torch.cat((bneck, down7), dim=1))
        up_skip7 = self.up_skip7(torch.cat((bneck, down7), dim=1))
        up7 += up_skip7

        up6 = self.up6(torch.cat((up7, down6), dim=1))
        up_skip6 = self.up_skip6(torch.cat((up7, down6), dim=1))
        up6 += up_skip6

        up5 = self.up5(torch.cat((up6, down5), dim=1))
        up_skip5 = self.up_skip5(torch.cat((up6, down5), dim=1))
        up5 += up_skip5

        up4 = self.up4(torch.cat((up5, down4), dim=1))
        up_skip4 = self.up_skip4(torch.cat((up5, down4), dim=1))
        up4 += up_skip4

        up3 = self.up3(torch.cat((up4, down3), dim=1))
        up_skip3 = self.up_skip3(torch.cat((up4, down3), dim=1))
        up3 += up_skip3

        up2 = self.up2(torch.cat((up3, down2), dim=1))
        up_skip2 = self.up_skip2(torch.cat((up3, down2), dim=1))
        up2 += up_skip2

        up1 = self.up1(torch.cat((up2, down1), dim=1))
        up_skip1 = self.up_skip1(torch.cat((up2, down1), dim=1))
        up1 += up_skip1

        out = up1.reshape(up1.shape[0], self._out_channels, 4, *up1.shape[-2:])

        if self._softmax:
            out = self.softmax(out)
        if self._permute_output:
            out = out.permute(0, 2, 1, 3, 4)
        return out

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
    def _skip(in_channels, out_channels, name, conv_bias):
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
