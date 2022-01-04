import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import re
import copy
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Type, Callable, Union, List, Optional

from models.block.ibn import IBN
from models.block.semodule import SEModule
from models.block.build_attention import build_attention
from models.util import remove_layers_from_state_dict
from models.activations import cfg_to_activation


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "resnet18_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth",
    "resnet34_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth",
    "resnet50_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
    "resnet101_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth",
    "resnet18d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
    "resnet34d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth",
    "resnet50d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
    "resnet101d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
    "resnext50d_32x4d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth",
    # With se-module
    "seresnet50": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth",
    "seresnext50_32x4d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth",
}


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ibn: Optional[str] = None,
        groups: int = 1,
        base_width: int = 64,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: Callable = nn.ReLU(inplace=True),
        with_se: bool = False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn1 = IBN(planes) if ibn == "a" else nn.BatchNorm2d(planes)
        self.relu = copy.deepcopy(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = (
            SEModule(
                channels=planes,
                rd_ratio=1 / 16.0,
                act_layer=nn.ReLU(inplace=True),
            )
            if with_se
            else nn.Identity()
        )
        self.IN = (
            nn.InstanceNorm2d(planes, affine=True) if ibn == "b" else nn.Identity()
        )
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(
            self.IN(
                self.downsample(x)
                + self.se(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
            )
        )


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ibn: Optional[str] = None,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        activation: Callable = nn.ReLU(inplace=True),
        with_se: bool = False,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = IBN(width) if ibn == "a" else nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = (
            SEModule(
                channels=planes * self.expansion,
                rd_ratio=1 / 16.0,
                act_layer=nn.ReLU(inplace=True),
            )
            if with_se
            else nn.Identity()
        )
        self.IN = (
            nn.InstanceNorm2d(planes * self.expansion, affine=True)
            if ibn == "b"
            else nn.Identity()
        )
        self.relu = copy.deepcopy(activation)
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(
            self.IN(
                self.downsample(x)
                + self.se(
                    self.bn3(
                        self.conv3(
                            self.relu(
                                self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
                            )
                        )
                    )
                )
            )
        )


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        ibn_cfg=("a", "a", "a", None),
        activation: Callable = nn.ReLU(inplace=True),
        last_stride: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        stem_type: str = "",
        stem_width: int = 64,
        avg_down: bool = False,
        with_se: bool = False,
        non_layers: Optional[List[int]] = None,
        attention_cfg: dict = None,
    ):
        self.groups = groups
        self.base_width = width_per_group
        self.with_se = with_se
        self.block = block
        super(ResNet, self).__init__()

        deep_stem = "deep" in stem_type
        self.inplanes = stem_width * 2 if deep_stem else 64

        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_chs[0], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                activation,
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                activation,
                nn.Conv2d(
                    stem_chs[1], self.inplanes, 3, stride=1, padding=1, bias=False
                ),
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )

        if ibn_cfg[0] == "b":
            self.bn1 = nn.InstanceNorm2d(self.inplanes, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.relu = copy.deepcopy(activation)
        # TODO: Check maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(
            block,
            self.inplanes,
            layers[0],
            ibn=ibn_cfg[0],
            activation=activation,
            avg_down=avg_down,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            ibn=ibn_cfg[1],
            activation=activation,
            avg_down=avg_down,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            ibn=ibn_cfg[2],
            activation=activation,
            avg_down=avg_down,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=last_stride,
            ibn=ibn_cfg[3],
            activation=activation,
            avg_down=avg_down,
        )
        self.feature_dim = 512 * block.expansion

        self.random_init()

        # non local block

        self._make_nl(layers, non_layers, attention_cfg=attention_cfg)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        ibn: Optional[str] = None,
        activation: Callable = nn.ReLU(),
        avg_down: bool = False,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not avg_down:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                    if stride != 1
                    else nn.Identity(),
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = [
            block(
                inplanes=self.inplanes,
                planes=planes,
                ibn=None if ibn == "b" else ibn,
                groups=self.groups,
                base_width=self.base_width,
                stride=stride,
                downsample=downsample,
                activation=activation,
                with_se=self.with_se,
            )
        ]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    ibn=None if (ibn == "b" and i < blocks - 1) else ibn,
                    groups=self.groups,
                    base_width=self.base_width,
                    activation=activation,
                    with_se=self.with_se,
                )
            )

        return nn.Sequential(*layers)

    def _make_nl(
        self,
        layers: List[int],
        non_layers: Optional[List[int]] = None,
        attention_cfg: dict = None,
    ):
        if non_layers is not None and attention_cfg is not None:
            name = attention_cfg["name"]
            rd_ratio = attention_cfg["rd_ratio"]
            pam_batchnorm = attention_cfg["pam_batchnorm"]
            cam_batchnorm = attention_cfg["cam_batchnorm"]

            self.NL_1 = nn.ModuleList(
                [
                    build_attention(
                        name=name,
                        channels=256,
                        cam_batchnorm=cam_batchnorm,
                        pam_batchnorm=pam_batchnorm,
                        rd_ratio=rd_ratio,
                    )  # type: ignore
                    for _ in range(non_layers[0])
                ]
            )
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [
                    build_attention(
                        name=name,
                        channels=512,
                        cam_batchnorm=cam_batchnorm,
                        pam_batchnorm=pam_batchnorm,
                        rd_ratio=rd_ratio,
                    )  # type: ignore
                    for _ in range(non_layers[1])
                ]
            )
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [
                    build_attention(
                        name=name,
                        channels=1024,
                        cam_batchnorm=cam_batchnorm,
                        pam_batchnorm=pam_batchnorm,
                        rd_ratio=rd_ratio,
                    )  # type: ignore
                    for _ in range(non_layers[2])
                ]
            )
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [
                    build_attention(
                        name=name,
                        channels=2048,
                        cam_batchnorm=cam_batchnorm,
                        pam_batchnorm=pam_batchnorm,
                        rd_ratio=rd_ratio,
                    )  # type: ignore
                    for _ in range(non_layers[3])
                ]
            )
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
            self.NL_1 = self.NL_2 = self.NL_3 = self.NL_4 = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)  # type: ignore
                NL1_counter += 1

        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)  # type: ignore
                NL2_counter += 1

        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)  # type: ignore
                NL3_counter += 1

        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)  # type: ignore
                NL4_counter += 1

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_resnet_name(
    resnext: bool,
    with_se: bool,
    num_layer: int,
    with_d: bool,
    resnext_type: str,
    with_ibn_a: bool,
    with_nl: bool,
) -> str:
    assert (
        num_layer >= 18 and num_layer <= 101
    ), "Only support resnet has num_layer in [18, 34, 50, 101]"

    resnet_name = ""
    if with_se:
        resnet_name += "se"

    resnet_name += "resnext" if resnext else "resnet"
    resnet_name += str(num_layer)

    if with_d:
        resnet_name += "d"

    if resnext:
        resnet_name += str(resnext_type)

    if with_ibn_a:
        resnet_name += "_ibn_a"

    if with_nl:
        resnet_name += "_nl"

    return resnet_name


def convert_se_layer_from_timm(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("se.fc1", "se.conv_reduce").replace(
            "se.fc2", "se.conv_expand"
        )

        new_state_dict[key] = value
    return new_state_dict


def build_resnet(cfg: dict) -> ResNet:
    resnext = cfg["resnext"]
    with_se = cfg["with_se"]
    num_layer = cfg["num_layer"]
    with_d = cfg["with_d"]
    resnext_type = cfg["resnext_type"]
    with_ibn_a = cfg["with_ibn_a"]
    last_stride = cfg["last_stride"]
    activation = cfg_to_activation(cfg["activation"])
    pretrained = cfg["pretrained"]
    progress = cfg["progress"]

    attention_cfg = {
        "enable": cfg["attention"]["enable"],
        "name": cfg["attention"]["name"],
        "rd_ratio": cfg["attention"]["rd_ratio"],
        "cam_batchnorm": cfg["attention"]["cam_batchnorm"],
        "pam_batchnorm": cfg["attention"]["pam_batchnorm"],
    }

    resnet_name = get_resnet_name(
        resnext,
        with_se,
        num_layer,
        with_d,
        resnext_type,
        with_ibn_a,
        with_nl=False,
    )

    blocks_cfg = {
        "18": BasicBlock,
        "34": BasicBlock,
        "50": Bottleneck,
        "101": Bottleneck,
    }

    layers_cfg = {
        "18": [2, 2, 2, 2],
        "34": [3, 4, 6, 3],
        "50": [3, 4, 6, 3],
        "101": [3, 4, 23, 3],
    }

    nl_cfg = {
        "18": [0, 0, 0, 0],
        "34": [0, 0, 0, 0],
        "50": [0, 2, 3, 0],
        "101": [0, 2, 9, 0],
    }

    model = ResNet(
        block=blocks_cfg[str(num_layer)],
        layers=layers_cfg[str(num_layer)],
        non_layers=nl_cfg[str(num_layer)] if cfg["attention"]["enable"] else None,
        ibn_cfg=("a", "a", "a", None) if with_ibn_a else (None, None, None, None),
        activation=activation,
        last_stride=last_stride,
        groups=int(resnext_type.split("*")[0]) if resnext_type is not None else 1,
        width_per_group=int(resnext_type.split("*")[1][:-1])
        if resnext_type is not None
        else 64,
        with_se=with_se,
        stem_width=32 if with_d else 64,
        stem_type="deep" if with_d else "",
        avg_down=bool(with_d),
        attention_cfg=attention_cfg,
    )

    if pretrained:
        try:
            url = None
            for pretrained_name in [
                resnet_name,
                resnet_name.replace("_nl", ""),
                resnet_name.replace("_ibn_a", ""),
                resnet_name.replace("_ibn_a", "").replace("nl", ""),
            ]:
                if pretrained_name in model_urls:
                    url = model_urls[pretrained_name]
                    break

            if url:
                state_dict = remove_layers_from_state_dict(
                    load_state_dict_from_url(
                        url,
                        progress=progress,
                        map_location="cpu" if not torch.cuda.is_available() else None,
                    ),
                    ["classifier"],
                )

                if with_se:
                    state_dict = convert_se_layer_from_timm(state_dict)

                model.load_state_dict(
                    state_dict,  # type: ignore
                    strict=not cfg["attention"]["enable"],
                )
            else:
                print("Not found pretrained")
        except Exception as e:
            print(e)

    return model
