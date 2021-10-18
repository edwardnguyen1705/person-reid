import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

from models.util import make_divisible


class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = copy.deepcopy(act_layer)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial BCHW tensors"""

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class GlobalContext(nn.Module):
    r"""GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
    https://blog.paperspace.com/global-context-networks-gcnet/
    """

    def __init__(
        self,
        channels,
        use_attn=True,
        fuse_add=False,
        fuse_scale=True,
        rd_ratio=1.0 / 8,
        rd_channels=None,
        rd_divisor=1,
        act_layer: Callable = nn.ReLU(inplace=True),
        gate_fn: Callable = nn.Sigmoid(),
        round_limit: float = 0.0,
    ):
        super(GlobalContext, self).__init__()
        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=round_limit
            )

        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None

        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        self.gate = gate_fn
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)

        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x
