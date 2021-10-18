import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from models.block.adb_attension import PAM
from models.block.semodule import SEModule


class PLRAttention(nn.Module):
    r"""Attention module from 'Learning Diverse Features with Part-Level Resolution for Person Re-Identification'"""

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1 / 16.0,
        pam_batchnorm: bool = True,
        *args,
        **kwargs
    ):
        super(PLRAttention, self).__init__()
        self.pam = PAM(channels=channels, batchnorm=pam_batchnorm)
        self.se_module = SEModule(
            channels=channels,
            rd_ratio=rd_ratio,
            act_layer=nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se_module(self.pam(x))
