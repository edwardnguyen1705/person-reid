import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from models.util import get_norm
from models.weight_init import weights_init_kaiming


class BNNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_norm: str = "2d",
        bias_freeze: bool = True,
        *args,
        **kwargs,
    ):
        super(BNNeck, self).__init__()
        self.bn = get_norm(in_channels, type_norm, bias_freeze)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)

    def reset_parameters(self) -> None:
        self.bn.apply(weights_init_kaiming)
