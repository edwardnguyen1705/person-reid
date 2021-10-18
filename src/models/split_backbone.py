import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from models.backbones.osnet import build_osnet
from models.backbones.osnet_ain import build_osnet_ain
from models.backbones.resnet import build_resnet


def get_split_backbone(cfg):
    if cfg["name"] == "resnet":
        model = build_resnet(cfg["resnet"])

        return (
            Backbone1Resnet(model),
            Backbone2Resnet(model),
            model.feature_dim,
            model.block,
        )

    elif cfg["name"] == "osnet":
        model = build_osnet(cfg["osnet"])

        return (
            nn.Sequential(
                model.conv1,
                model.maxpool,
                model.conv2,
                model.attention1,
                model.conv3[0],
            ),
            nn.Sequential(model.conv3[1:], model.attention2, model.conv4, model.conv5),
            model.feature_dim,
            model.block,
        )

    elif cfg["name"] == "osnet_ain":
        model = build_osnet_ain(cfg["osnet_ain"])

        return (
            nn.Sequential(
                model.conv1,
                model.maxpool,
                model.conv2,
                model.pool2,
                model.attention1,
                model.conv3[0],
            ),
            nn.Sequential(
                model.conv3[1:], model.pool3, model.attention2, model.conv4, model.conv5
            ),
            model.feature_dim,
            model.block,
        )

    raise ValueError("Not support backend_name, got unexpected: %s" % cfg["name"])


class Backbone1Resnet(nn.Module):
    def __init__(self, resnet):
        super(Backbone1Resnet, self).__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3[0]

        self.NL_1 = resnet.NL_1
        self.NL_1_idx = resnet.NL_1_idx
        self.NL_2 = resnet.NL_2
        self.NL_2_idx = resnet.NL_2_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1

        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1

        # Layer 3
        x = self.layer3(x)

        return x


class Backbone2Resnet(nn.Module):
    def __init__(self, resnet):
        super(Backbone2Resnet, self).__init__()
        self.layer3 = resnet.layer3[1:]
        self.layer4 = resnet.layer4

        self.NL_3 = resnet.NL_3
        self.NL_3_idx = resnet.NL_3_idx
        self.NL_4 = resnet.NL_4
        self.NL_4_idx = resnet.NL_4_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if (i + 1) == self.NL_3_idx[
                NL3_counter
            ]:  # because 1 layer in sequence splitted
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1

        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x
