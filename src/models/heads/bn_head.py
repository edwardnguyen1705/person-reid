import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from models.poolings import build_pooling
from models.necks import build_neck
from models.softmax import build_softmax
from models.activations import cfg_to_activation


__all__ = ["BNHead"]


class BNHead(nn.Module):
    def __init__(self, cfg, in_channels: int, num_classes: int):
        super(BNHead, self).__init__()
        self.feature_pos = cfg["feature_pos"]
        self.neck_type_norm = cfg["neck"]["type_norm"]

        neck_out_channels = cfg["neck"]["out_channels"]

        self.pool_layer = build_pooling(cfg["pooling"]["name"])

        if neck_out_channels is None:
            neck_out_channels = in_channels

        self.bottleneck = build_neck(
            name=cfg["neck"]["name"],
            in_channels=in_channels,
            out_channels=neck_out_channels,
            type_norm=cfg["neck"]["type_norm"],
            bias_freeze=cfg["neck"]["bias_freeze"],
            activation=cfg_to_activation(cfg["neck"]["activation"]),
        )

        self.softmax = build_softmax(
            name=cfg["head"]["name"],
            in_features=neck_out_channels,
            out_features=num_classes,
            scale=cfg["head"]["scale"],
            margin=cfg["head"]["margin"],
            k=cfg["head"]["k"],
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        pool_feat = self.pool_layer(x)

        if self.neck_type_norm == "1d":
            pool_feat = pool_feat.view(x.shape[0], -1)

        neck_feat = self.bottleneck(pool_feat)

        if self.neck_type_norm == "2d":
            neck_feat = neck_feat.view(x.shape[0], -1)

        if not self.training or targets is None:
            return neck_feat

        if self.feature_pos == "before":
            feat = (
                pool_feat.view(x.shape[0], -1)
                if self.neck_type_norm == "2d"
                else pool_feat
            )
        elif self.feature_pos == "after":
            feat = neck_feat
        else:
            raise KeyError(f"neck_feat: {self.neck_feat}")

        out_cls, pred_cls = self.softmax(neck_feat, targets)

        return feat, out_cls, pred_cls
