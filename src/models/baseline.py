import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.backbones import build_backbone
from models.heads import build_head


class Baseline(nn.Module):
    def __init__(self, cfg: dict, num_classes: int):
        super(Baseline, self).__init__()

        self.backbone, self.feature_dim, *_ = build_backbone(cfg["backbone"])

        self.head = build_head(
            cfg=cfg["head"], in_channels=self.feature_dim, num_classes=num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(x), targets)


if __name__ == "__main__":
    from utils import read_cfg

    cfg = read_cfg("configs/models/fast_reid.yml")

    from ptflops import get_model_complexity_info

    model = Baseline(cfg["model"], 751)

    macs, params = get_model_complexity_info(
        model.eval(),
        (3, 384, 128),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
