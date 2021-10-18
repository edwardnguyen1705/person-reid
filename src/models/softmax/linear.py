import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super(Linear, self).__init__()
        self.ln = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.ln(inputs)

        return inputs, inputs
