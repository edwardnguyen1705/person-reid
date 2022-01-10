import copy
import torch
import torch.nn as nn
from typing import Callable


def create_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    bias=False,
    groups=1,
    activation: Callable = nn.LeakyReLU(inplace=True),
    dropblock: Callable = nn.Identity(),
):
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )
    bn = nn.BatchNorm2d(num_features=out_channels)
    activation = copy.deepcopy(activation)
    dropblock = copy.deepcopy(dropblock)

    return conv, bn, dropblock, activation


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        activation: Callable = nn.LeakyReLU(inplace=True),
        dropblock: Callable = nn.Identity(),
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.dropblock = copy.deepcopy(dropblock)
        self.activation = copy.deepcopy(activation)

    def forward(self, x):
        return self.activation(self.dropblock(self.norm(self.conv(x))))
