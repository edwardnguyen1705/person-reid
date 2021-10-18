import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
from typing import Callable, Optional

from models.util import make_divisible


class SEModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        act_layer: Callable = nn.ReLU(inplace=True),
        gate_fn: Callable = nn.Sigmoid(),
        round_limit: float = 0.0,
    ):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=round_limit
            )
        self.conv_reduce = nn.Conv2d(
            in_channels=channels, out_channels=rd_channels, kernel_size=1, bias=True
        )
        self.act1 = copy.deepcopy(act_layer)
        self.conv_expand = nn.Conv2d(
            in_channels=rd_channels, out_channels=channels, kernel_size=1, bias=True
        )
        self.gate_fn = copy.deepcopy(gate_fn)

    def forward(self, x):
        return x * self.gate_fn(
            self.conv_expand(self.act1(self.conv_reduce(x.mean((2, 3), keepdim=True))))
        )
