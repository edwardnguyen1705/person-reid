import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.backbones import build_backbone
from models.heads.embedding import Embedding


class Baseline(nn.Module):
    def __init__(self, cfg: dict):
        super(Baseline, self).__init__()

        self.backbone, self.feature_dim, *_ = build_backbone(cfg["backbone"])

        self.head = Embedding(
            cfg=cfg,
            backbone_feature_dim=self.feature_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(x), targets)


if __name__ == "__main__":
    from utils import read_cfg

    cfg = read_cfg("configs/light_mbn.yaml")

    from ptflops import get_model_complexity_info

    model = Baseline(cfg["model"])

    macs, params = get_model_complexity_info(
        model.eval(),
        (3, 384, 128),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
