import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch.nn import Parameter

from models.softmax.subcenter import ArcMarginSubCenter


class LinearSoftmax(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: int = 30,
        margin: float = 0.25,
        k: int = 1,
        *args,
        **kwargs
    ) -> None:
        super(LinearSoftmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.sub_center = ArcMarginSubCenter(in_features, out_features, k)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.sub_center(inputs)

        inputs = logits.clone()

        inputs = inputs.mul(self.scale)

        return inputs, logits.mul(self.scale)
