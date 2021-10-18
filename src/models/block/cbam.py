import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.common import Conv
from models.util import autopad


class ChannelAttn(nn.Module):
    r"""Original CBAM channel attention module, currently avg + max pool variant only."""

    def __init__(self, channels, reduction: int = 16, activation=nn.ReLU(inplace=True)):
        super(ChannelAttn, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = copy.deepcopy(activation)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * (
                self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
                # x_avg
                + self.fc2(self.act(self.fc1(F.adaptive_max_pool2d(x, 1))))
                # x_max
            ).sigmoid()
        )


class SpatialAttn(nn.Module):
    r"""Original CBAM spatial attention module"""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttn, self).__init__()
        self.conv = Conv(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=autopad(kernel_size),
            bias=False,
            activation=nn.Identity(),
        )

    def forward(self, x):
        return (
            x
            * (
                self.conv(
                    torch.cat(
                        [
                            torch.mean(x, dim=1, keepdim=True),
                            # x_avg
                            torch.max(x, dim=1, keepdim=True)[0],
                            # x_max
                        ],
                        dim=1,
                    )
                )
            ).sigmoid()
        )


class CbamModule(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
