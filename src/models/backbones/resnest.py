import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import re
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Type, Callable, Union, List, Optional

from models.layers.common import create_conv
from models.block.ibn import IBN
from models.block.semodule import SEModule
from models.block.build_attention import build_attention
from models.util import remove_layers_from_state_dict, autopad, make_divisible
from models.activations import cfg_to_activation


from models.backbones.resnet import ResNet


model_urls = {
    "resnest14d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth",
    "resnest26d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth",
    "resnest50d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",
    "resnest101e": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",
    "resnest200e": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",
    "resnest269e": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",
}


class RadixSoftmax(nn.Module):
    def __init__(self, radix: int, cardinality: int):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        radix: int = 2,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        activation: Callable = nn.LeakyReLU(inplace=True),
    ):
        super(SplitAttn, self).__init__()
        self.radix = radix

        out_channels = out_channels or in_channels
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor
            )
        else:
            attn_chs = rd_channels * radix

        # Radix-Conv
        self.conv, self.bn0, self.drop_block, self.act0 = create_conv(
            in_channels=in_channels,
            out_channels=mid_chs,
            kernel_size=3,
            stride=stride,
            padding=autopad(3, padding),
            groups=groups * radix,
            activation=activation,
        )
        # Attention
        self.fc1 = nn.Conv2d(out_channels, attn_chs, kernel_size=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(attn_chs)
        self.act1 = copy.deepcopy(activation)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, kernel_size=1, groups=groups)
        # Softmax
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape

        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)

        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)

        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(
                dim=1
            )
        else:
            out = x * x_attn

        return out.contiguous()


class ResNestBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ibn: Optional[str] = None,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        activation: Callable = nn.ReLU(inplace=True),
        with_se: bool = False,
        avd=False,
        avd_first: bool = False,
        is_first: bool = False,
        radix: int = 1,
    ):
        assert not with_se, "not support"
        assert ibn is None, "not support"

        super(ResNestBottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1, self.bn1, _, self.act1 = create_conv(
            in_channels=inplanes,
            out_channels=width,
            kernel_size=1,
            activation=activation,
        )

        self.avd_first = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and avd_first
            else nn.Identity()
        )

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                in_channels=width,
                out_channels=width,
                stride=stride,
                padding=1,
                groups=groups,
                radix=self.radix,
                rd_ratio=0.25,
                rd_channels=None,
                rd_divisor=8,
            )
            self.bn2 = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            )
            self.bn2 = nn.BatchNorm2d(width)
            self.act2 = copy.deepcopy(activation)

        self.avd_last = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and not avd_first
            else nn.Identity()
        )

        self.conv3, self.bn3, _, self.act3 = create_conv(
            in_channels=width,
            out_channels=width * self.expansion,
            kernel_size=1,
            activation=activation,
        )

        self.downsample = downsample if downsample is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.downsample(shortcut)

        out += shortcut

        out = self.act3(out)

        return out


def get_resnest_name(
    num_layer: int,
    stem_width: int,
) -> str:
    resnest_name = "resnest" + str(num_layer)

    if stem_width == 32:
        resnest_name += "d"
    elif stem_width == 64:
        resnest_name += "e"
    else:
        raise ValueError("stem_width")

    return resnest_name


def build_resnest(cfg: dict) -> ResNet:
    num_layer = cfg["num_layer"]
    last_stride = cfg["last_stride"]
    activation = cfg_to_activation(cfg["activation"])
    pretrained = cfg["pretrained"]
    progress = cfg["progress"]

    blocks_cfg = {
        "14": ResNestBottleneck,
        "26": ResNestBottleneck,
        "50": ResNestBottleneck,
        "101": ResNestBottleneck,
        "200": ResNestBottleneck,
    }

    layers_cfg = {
        "14": [1, 1, 1, 1],
        "26": [2, 2, 2, 2],
        "50": [3, 4, 6, 3],
        "101": [3, 4, 23, 3],
        "200": [3, 24, 36, 3],
    }

    stem_width = {
        "14": 32,
        "26": 32,
        "50": 32,
        "101": 64,
        "200": 64,
    }

    model = ResNet(
        block=blocks_cfg[str(num_layer)],
        layers=layers_cfg[str(num_layer)],
        non_layers=None,
        ibn_cfg=(None, None, None, None),
        activation=activation,
        last_stride=last_stride,
        groups=1,
        width_per_group=64,
        with_se=False,
        stem_width=stem_width[str(num_layer)],
        stem_type="deep",
        avg_down=True,
        attention_cfg=None,
        block_args=dict(radix=2, avd=True, avd_first=False),
    )

    resnest_name = get_resnest_name(num_layer, stem_width[str(num_layer)])

    if pretrained:
        url = model_urls[resnest_name]

        state_dict = remove_layers_from_state_dict(
            load_state_dict_from_url(
                url,
                progress=progress,
                map_location="cpu" if not torch.cuda.is_available() else None,
            ),
            ["classifier", "fc"],
        )

        model.load_state_dict(state_dict, strict=True)

    return model


if __name__ == "__main__":
    resnet = build_resnest(
        dict(
            num_layer=200,
            last_stride=2,
            activation=dict(name="ReLU", inplace=True),
            pretrained=True,
            progress=True,
        )
    )

    print(resnet)

    input = torch.randn(4, 3, 256, 128)

    output = resnet(input)

    print(output.shape)
