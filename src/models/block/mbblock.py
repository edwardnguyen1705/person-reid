import os
from re import I
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
from typing import Callable

from models.util import autopad, make_divisible
from models.block.semodule import SEModule
from models.block.global_context import GlobalContext
from models.layers.dropblock import drop_path


def get_att(att_name: str):
    if att_name == "SEModule":
        return SEModule
    elif att_name == "GlobalContext":
        return GlobalContext
    else:
        raise ValueError("att_name not support")


class ConvBnAct(nn.Module):
    r"""Conv + Norm Layer + Activation w/ optional skip connection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 3,
        activation: Callable = nn.ReLU(inplace=True),
        drop_path_rate: float = 0.0,
        *args,
        **kwargs
    ):
        super(ConvBnAct, self).__init__()
        self.drop_path_rate = drop_path_rate
        self.has_residual = in_channels == out_channels and stride == 1

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = copy.deepcopy(activation)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DSConv(nn.Module):
    r"""Same as MBConv but exp_ratio is 1.0. Is is first block in Efficientnet and MobileNetV3"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        exp_ratio: float = 1.0,
        se_layer: bool = True,
        activation: Callable = nn.ReLU(inplace=True),
        rd_ratio: float = 1 / 16.0,
        rd_divisor: int = 4,
        gate_fn: Callable = nn.Sigmoid(),
        drop_path_rate: float = 0.0,
        round_limit: float = 0.0,
        att_name: str = "SEModule",
    ):
        super(DSConv, self).__init__()
        self.drop_path_rate = drop_path_rate
        self.has_residual = in_channels == out_channels and stride == 1

        self.conv_dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.act1 = copy.deepcopy(activation)

        self.se = (
            get_att(att_name)(
                channels=in_channels,
                rd_ratio=rd_ratio,
                rd_divisor=rd_divisor,
                act_layer=activation,
                gate_fn=gate_fn,
                round_limit=round_limit,
            )
            if se_layer
            else nn.Identity()
        )

        self.conv_pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = copy.deepcopy(activation)

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        exp_ratio: float = 4,
        se_layer: bool = True,
        activation: Callable = nn.ReLU(inplace=True),
        rd_ratio: float = 1 / 16.0,
        rd_divisor: int = 4,
        gate_fn: Callable = nn.Sigmoid(),
        drop_path_rate: float = 0.0,
        round_limit: float = 0.0,
        att_name: str = "SEModule",
    ):
        super(MBConv, self).__init__()
        self.drop_path_rate = drop_path_rate
        self.has_residual = in_channels == out_channels and stride == 1

        hidden_channels = make_divisible(in_channels * exp_ratio)

        self.conv_pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.act1 = copy.deepcopy(activation)

        self.conv_dw = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channels)
        self.act2 = copy.deepcopy(activation)

        self.se = (
            get_att(att_name)(
                channels=hidden_channels,
                rd_ratio=rd_ratio,
                rd_divisor=rd_divisor,
                act_layer=activation,
                gate_fn=gate_fn,
                round_limit=round_limit,
            )
            if se_layer
            else nn.Identity()
        )

        self.conv_pwl = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        shortcut = x

        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.se(x)

        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        exp_ratio: float = 4,
        se_layer: bool = False,
        activation: Callable = nn.ReLU(inplace=True),
        rd_ratio: float = 1 / 16.0,
        rd_divisor: int = 4,
        gate_fn: Callable = nn.Sigmoid(),
        drop_path_rate: float = 0.0,
        round_limit: float = 0.0,
        att_name: str = "SEModule",
        *args,
        **kwargs
    ):
        super(FusedMBConv, self).__init__()
        self.drop_path_rate = drop_path_rate
        self.has_residual = in_channels == out_channels and stride == 1

        hidden_channels = make_divisible(in_channels * exp_ratio)

        self.conv_exp = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.act1 = copy.deepcopy(activation)

        self.se = (
            get_att(att_name)(
                channels=hidden_channels,
                rd_ratio=rd_ratio,
                rd_divisor=rd_divisor,
                act_layer=activation,
                gate_fn=gate_fn,
                round_limit=round_limit,
            )
            if se_layer
            else nn.Identity()
        )

        self.conv_pwl = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        shortcut = x

        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x
