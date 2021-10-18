import os
import sys

from models.activations import cfg_to_activation

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
from typing import Callable
from torch.hub import load_state_dict_from_url

from models.util import autopad
from models.block.mbblock import FusedMBConv, MBConv, ConvBnAct

URL = {
    "efficientnetv2_rw_t": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth",
    "gc_efficientnetv2_rw_t": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gc_efficientnetv2_rw_t_agc-927a0bde.pth",
    "efficientnetv2_rw_s": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_v2s_ra2_288-a6477665.pth",
    "efficientnetv2_rw_m": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_rw_m_agc-3d90cb1e.pth",
}


FEATURE_DIM = {
    "efficientnetv2_rw_t": 1024,
    "gc_efficientnetv2_rw_t": 1024,
    "efficientnetv2_rw_s": 1792,
    "efficientnetv2_rw_m": 2152,
}

ATT_NAME = {
    "efficientnetv2_rw_t": "SEModule",
    "gc_efficientnetv2_rw_t": "GlobalContext",
    "efficientnetv2_rw_s": "SEModule",
    "efficientnetv2_rw_m": "SEModule",
}


V2T = [
    # module, out, stride, layers, se, exp_ratio, kernel_size
    (ConvBnAct, 24, 1, 2, False, 1, 3),
    (FusedMBConv, 40, 2, 4, False, 4, 3),
    (FusedMBConv, 48, 2, 4, False, 4, 3),
    (MBConv, 104, 2, 6, True, 4, 3),
    (MBConv, 128, 1, 9, True, 6, 3),
    (MBConv, 208, 2, 14, True, 6, 3),
]

V2S = [
    # module, out, stride, layers, se, exp_ratio, kernel_size
    (FusedMBConv, 24, 1, 2, False, 1, 3),
    (FusedMBConv, 48, 2, 4, False, 4, 3),
    (FusedMBConv, 64, 2, 4, False, 4, 3),
    (MBConv, 128, 2, 6, True, 4, 3),
    (MBConv, 160, 1, 9, True, 6, 3),
    (MBConv, 272, 2, 15, True, 6, 3),
]

V2M = [
    # module, out, stride, layers, se, exp_ratio, kernel_size
    (FusedMBConv, 32, 1, 3, False, 1, 3),
    (FusedMBConv, 56, 2, 5, False, 4, 3),
    (FusedMBConv, 80, 2, 5, False, 4, 3),
    (MBConv, 152, 2, 8, True, 4, 3),
    (MBConv, 192, 1, 15, True, 6, 3),
    (MBConv, 328, 2, 24, True, 6, 3),
]

SETTINGS = {
    "efficientnetv2_rw_t": V2T,
    "gc_efficientnetv2_rw_t": V2T,
    "efficientnetv2_rw_s": V2S,
    "efficientnetv2_rw_m": V2M,
}


class EfficientnetV2_RW(nn.Module):
    r"""
    Custom version by timm
    """

    def __init__(
        self,
        settings: list,
        feature_dim: int,
        activation: Callable = nn.SiLU(inplace=True),
        drop_path_rate: float = 0.2,
        att_name: str = "SEModule",
    ):
        super(EfficientnetV2_RW, self).__init__()
        self.feature_dim = feature_dim

        stem_size = settings[0][1]

        self.conv_stem = nn.Conv2d(
            in_channels=3,
            out_channels=stem_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=24)
        self.act1 = copy.deepcopy(activation)

        self.blocks = []
        pre_channels = stem_size
        for (
            module,
            out_channel,
            stride,
            num_layer,
            se,
            exp_ratio,
            kernel_size,
        ) in settings:
            blocks = []
            for idx in range(num_layer):
                blocks.append(
                    module(
                        in_channels=pre_channels,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        stride=stride if idx == 0 else 1,
                        padding=autopad(kernel_size),
                        exp_ratio=exp_ratio,
                        se_layer=se,
                        activation=activation,
                        rd_ratio=1 / (4.0 * exp_ratio),
                        rd_divisor=1.0,
                        drop_path_rate=drop_path_rate,
                        round_limit=0.9,
                        att_name=att_name,
                    )
                )
                pre_channels = out_channel
            self.blocks.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*self.blocks)

        self.conv_head = nn.Conv2d(
            in_channels=pre_channels,
            out_channels=feature_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=feature_dim)
        self.act2 = copy.deepcopy(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


def efficientnetv2_rw(
    name,
    pretrained: bool = True,
    progress: bool = True,
    activation: Callable = nn.SiLU(inplace=True),
    *args,
    **kwargs,
):
    model = EfficientnetV2_RW(
        settings=SETTINGS[name],
        feature_dim=FEATURE_DIM[name],
        att_name=ATT_NAME[name],
        activation=activation,
        *args,
        **kwargs,
    )

    if pretrained:
        model.load_state_dict(
            {
                key: value
                for key, value in load_state_dict_from_url(
                    URL[name],
                    progress=progress,
                    map_location="cpu" if not torch.cuda.is_available() else None,
                ).items()
                if not key.startswith("classifier")
            }
        )

    return model


def build_efficientnetv2_rw(cfg):
    version = cfg["version"]
    activation = cfg_to_activation(cfg["activation"])
    pretrained = cfg["pretrained"]
    progress = cfg["progress"]

    return efficientnetv2_rw(
        name=version,
        pretrained=pretrained,
        progress=progress,
        activation=activation,
    )


if __name__ == "__main__":
    model = efficientnetv2_rw(
        name="gc_efficientnetv2_rw_t",
        progress=True,
    )

    print(model)
