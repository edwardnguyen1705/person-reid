import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List
from torch.nn import Parameter

from models.softmax.subcenter import ArcMarginSubCenter


class ArcFaceAdaptiveMargin(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        margins: torch.Tensor,
        scale: int = 30,
        k: int = 3,
        *args,
        **kwargs
    ) -> None:
        super(ArcFaceAdaptiveMargin, self).__init__()
        self.scale = scale
        self.register_buffer("margins", margins)
        self.sub_center = ArcMarginSubCenter(in_features, out_features, k)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.sub_center(inputs)

        pred_cls = inputs.clone().mul(self.scale)

        with torch.no_grad():
            ms = self.margins[targets]

            cos_m = torch.cos(ms)
            sin_m = torch.sin(ms)

            th = torch.cos(math.pi - ms)
            mm = torch.sin((math.pi - ms) * ms)

            # Remove index = -1 in market1501 datasets
            index = torch.where(targets != -1)[0]

            m_hot = torch.zeros(
                (index.shape[0], inputs.shape[1]),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            # m_hot.shape = (batch_size, feature_dim)

            m_hot.scatter_(1, targets.unsqueeze(dim=1), 1)

        cosine = inputs
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))

        output = (m_hot * phi) + ((1.0 - m_hot) * cosine)
        output *= self.scale

        return output, pred_cls


if __name__ == "__main__":
    import numpy as np

    from data.datasources import build_datasource
    from data.dataloader import create_dataloader
    from utils.config import read_config, dotter

    # datasource = build_datasource(
    #     name="market1501", root="/home/coder/project/datasets/market1501/processed/"
    # )

    config = read_config("configs/fast_reid.yml", base=True)

    print({k: str(v) for k, v in dotter(config).items()})

    # dataloader, datasource = create_dataloader(config["data"])

    # data = datasource.get_data("train")

    # label = np.array([x[1] for x in data])

    # unique, counts = np.unique(label, return_counts=True)

    # if (unique == np.arange(len(datasource.get_classes()))).all():
    #     print("Yes")
    # else:
    #     print("No")

    # tmp = np.sqrt(1 / counts)

    # margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # arc_face = ArcFaceAdaptiveMargin(
    #     512, len(datasource.get_classes()), torch.from_numpy(margins), scale=30, k=3
    # )

    # for inputs, targets, *_ in dataloader["train"]:
    #     break

    # inputs = torch.rand((64, 512))

    # out = arc_face(inputs, targets)

    pass
