import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from models.block.adb_attension import PAM, CAM


class FPBAttention(nn.Module):
    r"""Attention module from 'FPB: Feature Pyramid Branch for Person Re-Identification'"""

    def __init__(
        self, channels: int, pam_batchnorm: bool, cam_batchnorm, *args, **kwargs
    ):
        super(FPBAttention, self).__init__()
        self.pam = PAM(channels=channels, batchnorm=pam_batchnorm)
        self.cam = CAM(channels=channels, batchnorm=cam_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cam(self.pam(x))
