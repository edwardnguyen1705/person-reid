import os
import re
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import math
import torch
import torch.nn as nn

from typing import Callable
from torch.hub import load_state_dict_from_url

from models.util import autopad, make_divisible
from models.block.mbblock import MBConv, DSConv
from models.activations import cfg_to_activation


URL = {
    "b0": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth",
    "b1": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth",
    "b2": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth",
    "b3": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth",
    "b4": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth",
}


BASE_MODEL = [
    # module, expand_ratio, channels, repeats, stride, kernel_size
    [DSConv, 1, 16, 1, 1, 3],
    [MBConv, 6, 24, 2, 2, 3],
    [MBConv, 6, 40, 2, 2, 5],
    [MBConv, 6, 80, 3, 2, 3],
    [MBConv, 6, 112, 3, 1, 5],
    [MBConv, 6, 192, 4, 2, 5],
    [MBConv, 6, 320, 1, 1, 3],
]


BASE_MODEL_RLS = [
    # module, expand_ratio, channels, repeats, stride, kernel_size
    [DSConv, 1, 16, 1, 1, 3],
    [MBConv, 6, 24, 2, 2, 3],
    [MBConv, 6, 40, 2, 2, 5],
    [MBConv, 6, 80, 3, 2, 3],
    [MBConv, 6, 112, 3, 1, 5],
    [MBConv, 6, 192, 4, 1, 5],
    [MBConv, 6, 320, 1, 1, 3],
]

SETTINGS = {
    # From timm
    # (width_factor, depth_factor, resolution, drop_rate)
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
    "b8": (2.2, 3.6, 672, 0.5),
    # "l2": (4.3, 5.3, 800, 0.5),
}


class Efficientnet(nn.Module):
    def __init__(
        self,
        depth_factor: float,
        width_factor: float,
        resolution: int,
        activation: Callable = nn.SiLU(inplace=True),
        drop_path_rate: float = 0.2,
        base_model=BASE_MODEL,
    ):
        super(Efficientnet, self).__init__()

        self.last_channels = math.ceil(1280 * width_factor)

        channels = make_divisible(32 * width_factor, divisor=8)

        self.conv_stem = nn.Conv2d(
            in_channels=3,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.act1 = copy.deepcopy(activation)

        self.blocks = []
        pre_channels = channels
        for module, expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = make_divisible(channels * width_factor, divisor=8)

            num_layer = math.ceil(repeats * depth_factor)

            blocks = []

            for layer_id in range(num_layer):
                blocks.append(
                    module(
                        in_channels=pre_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if layer_id == 0 else 1,
                        padding=autopad(kernel_size),
                        exp_ratio=expand_ratio,
                        se_layer=True,
                        activation=activation,
                        rd_ratio=1 / (4.0 * expand_ratio),
                        rd_divisor=1.0,
                        drop_path_rate=drop_path_rate,
                        round_limit=0.9,
                    )
                )
                pre_channels = out_channels
            self.blocks.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*self.blocks)

        self.conv_head = nn.Conv2d(
            in_channels=pre_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.last_channels)
        self.act2 = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x

    def get_feature_dim(self):
        return self.last_channels


def efficientnet(
    version: str,
    pretrained=True,
    progress=True,
    activation: Callable = nn.SiLU(inplace=True),
    remove_last_stride=False,
):
    assert version in SETTINGS, f"version must in {list(SETTINGS.keys())}"

    base_model = BASE_MODEL

    if remove_last_stride:
        base_model = BASE_MODEL_RLS

    width_factor, depth_factor, res, _ = SETTINGS[version]

    model = Efficientnet(
        depth_factor, width_factor, res, activation, base_model=base_model
    )

    if pretrained and version in URL:
        model.load_state_dict(
            {
                key: value
                for key, value in load_state_dict_from_url(
                    URL[version], progress=progress
                ).items()
                if not key.startswith("classifier")
            }
        )

    return model


def build_efficientnet(cfg):
    version = cfg["version"]
    remove_last_stride = cfg["remove_last_stride"]
    activation = cfg_to_activation(cfg["activation"])
    pretrained = cfg["pretrained"]
    progress = cfg["progress"]

    return efficientnet(
        version=version,
        pretrained=pretrained,
        progress=progress,
        activation=activation,
        remove_last_stride=remove_last_stride,
    )
