import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn

from typing import Callable

from models.util import get_norm
from models.weight_init import weights_init_kaiming


class ThirdPlaceLandmarksNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        activation: Callable = nn.SiLU(inplace=True),
        *args,
        **kwargs,
    ):
        super(ThirdPlaceLandmarksNeck, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.act = copy.deepcopy(activation)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))

    def reset_parameters(self) -> None:
        self.bn.apply(weights_init_kaiming)
