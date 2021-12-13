import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.util import make_divisible


class NonLocalBlock(nn.Module):
    r"""Non local block:
    Inspired from:
        - https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf
        - https://arxiv.org/pdf/2001.04193.pdf
        - https://github.com/mangye16/ReID-Survey/blob/master/modeling/layer/non_local.py
        - https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
    """

    def __init__(self, channels, rd_ratio: float = 1.0 / 16):
        super(NonLocalBlock, self).__init__()

        self.channels = channels

        self.hidden_channel = 1 if rd_ratio == -1 else int(channels * rd_ratio)
        self.theta = nn.Conv2d(
            self.channels, self.hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            self.channels, self.hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.g = nn.Conv2d(
            self.channels, self.hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Sequential(
            nn.Conv2d(
                self.hidden_channel, self.channels, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(self.channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)  # type: ignore
        nn.init.constant_(self.W[1].bias, 0.0)  # type: ignore

    def forward(self, x):
        batch_size = x.shape[0]

        theta_out = (
            self.theta(x).view(batch_size, self.hidden_channel, -1).permute(0, 2, 1)
        )
        # theta_out.shape = (batch_size, hxw, hidden_channel)

        phi_out = self.phi(x).view(batch_size, self.hidden_channel, -1)
        # phi_out.shape = (batch_size, hidden_channel, hxw)

        g_out = self.g(x).view(batch_size, self.hidden_channel, -1).permute(0, 2, 1)
        # phi_out.shape = (batch_size, hxw, hidden_channel)

        f = torch.matmul(theta_out, phi_out)
        f = f / f.size(-1)

        y = torch.matmul(f, g_out).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.hidden_channel, x.size(2), x.size(3))

        W_out = self.W(y)

        return W_out + x


class NonLocalAttn(nn.Module):
    r"""Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(
        self,
        channels,
        use_scale=True,
        rd_ratio=1 / 8.0,
        rd_channels=None,
        rd_divisor=8,
        **kwargs
    ):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            if rd_ratio == -1:
                rd_channels = 1
            rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
        self.scale = channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv2d(channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, channels, kernel_size=1, stride=1, bias=True)
        self.norm = nn.BatchNorm2d(channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, C, H, W = t.size()
        t = t.view(B, C, -1).permute(0, 2, 1)
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).permute(0, 2, 1)

        att = torch.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)  # type: ignore
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
