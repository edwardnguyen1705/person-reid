import os
import sys
import copy

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch
import torch.nn as nn
from typing import Callable
from torch.hub import load_state_dict_from_url


from models.util import autopad
from models.activations import cfg_to_activation
from models.block.mbblock import MBConv, DSConv, ConvBnAct


URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth"


class MobileNetV3Large(nn.Module):
    def __init__(
        self,
        activation1: Callable = nn.ReLU(inplace=True),
        activation2: Callable = nn.Hardswish(inplace=True),
        gate_fn: Callable = nn.Hardsigmoid(inplace=True),
    ):
        super(MobileNetV3Large, self).__init__()

        settings = [
            # module, in_channels, out_channels, kernel_size, stride, exp, SELayer, activation
            [
                (DSConv, 16, 16, 3, 1, 16, False, activation1),
            ],
            [
                (MBConv, 16, 24, 3, 2, 64, False, activation1),
                (MBConv, 24, 24, 3, 1, 72, False, activation1),
            ],
            [
                (MBConv, 24, 40, 5, 2, 72, True, activation1),
                (MBConv, 40, 40, 5, 1, 120, True, activation1),
                (MBConv, 40, 40, 5, 1, 120, True, activation1),
            ],
            [
                (MBConv, 40, 80, 3, 2, 240, False, activation2),
                (MBConv, 80, 80, 3, 1, 200, False, activation2),
                (MBConv, 80, 80, 3, 1, 184, False, activation2),
                (MBConv, 80, 80, 3, 1, 184, False, activation2),
            ],
            [
                (MBConv, 80, 112, 3, 1, 480, True, activation2),
                (MBConv, 112, 112, 3, 1, 672, True, activation2),
            ],
            [
                (MBConv, 112, 160, 5, 2, 672, True, activation2),
                (MBConv, 160, 160, 5, 1, 960, True, activation2),
                (MBConv, 160, 160, 5, 1, 960, True, activation2),
            ],
            [
                (ConvBnAct, 160, 960, 1, 1, 672, False, activation2),
            ],
        ]

        self.conv_stem = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.act1 = copy.deepcopy(activation2)

        self.blocks = []
        for setting in settings:
            blocks = []
            for (
                module,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                exp,
                se_layer,
                activation,
            ) in setting:
                blocks.append(
                    module(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=autopad(kernel_size),
                        exp_ratio=exp / in_channels,
                        se_layer=se_layer,
                        activation=activation,
                        rd_ratio=1 / 4.0,
                        rd_divisor=8,
                        gate_fn=gate_fn,
                        drop_path_rate=0.0,
                        round_limit=0.9,
                    )
                )

            self.blocks.append(nn.Sequential(*blocks))
            self.feature_dim = out_channels

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x

    def get_feature_dim(self):
        return self.feature_dim


def build_mobilenetv3(cfg):
    activation1 = cfg_to_activation(cfg["activation1"])
    activation2 = cfg_to_activation(cfg["activation2"])
    gate_fn = cfg_to_activation(cfg["gate_fn"])
    pretrained = cfg["pretrained"]
    progress = cfg["progress"]

    model = MobileNetV3Large(
        activation1=activation1,
        activation2=activation2,
        gate_fn=gate_fn,
    )

    if pretrained:
        model.load_state_dict(
            {
                key: value
                for key, value in load_state_dict_from_url(
                    URL,
                    progress=progress,
                    map_location="cpu" if not torch.cuda.is_available() else None,
                ).items()
                if (
                    not key.startswith("conv_head") and not key.startswith("classifier")
                )
            }
        )

    return model
