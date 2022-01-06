import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import torch
import torch.nn as nn

from models.layers.common import Conv
from models.heads.embedding import Embedding
from models.split_backbone import get_split_backbone
from models.layers.top_dropblock import BatchFeatureErase_Top


class LightMBN(nn.Module):
    def __init__(self, cfg: dict):
        super(LightMBN, self).__init__()

        feature_dim = cfg["feature_dim"]

        (
            self.backbone,
            other,
            self.backbone_dim,
            self.backbone_block,
        ) = get_split_backbone(cfg["backbone"])

        self.global_branch = copy.deepcopy(other)
        self.partial_branch = copy.deepcopy(other)
        self.channel_branch = copy.deepcopy(other)

        self.batch_dropblock_top = BatchFeatureErase_Top(
            channels=self.backbone_dim, bottleneck=self.backbone_block
        )

        self.shared_channel_conv = Conv(
            in_channels=self.backbone_dim // 2,
            out_channels=feature_dim,
            kernel_size=1,
            bias=False,
            activation=nn.ReLU(inplace=True),
        )
        self.weights_init_shared(self.shared_channel_conv)

        self.glo_max_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.glo_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.cha_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.par_max_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.par_avg_pooling = nn.AdaptiveAvgPool2d((2, 1))

        reduction = nn.Conv2d(
            in_channels=self.backbone_dim,
            out_channels=feature_dim,
            kernel_size=1,
            bias=False,
        )
        self.glo_drop_reduction = copy.deepcopy(reduction)
        self.glo_reduction = copy.deepcopy(reduction)
        self.par_1_reduction = copy.deepcopy(reduction)
        self.par_2_reduction = copy.deepcopy(reduction)
        self.par_g_reduction = copy.deepcopy(reduction)

        head = Embedding(
            cfg=cfg,
            backbone_feature_dim=feature_dim,
        )

        self.glo_drop_head = copy.deepcopy(head)
        self.glo_head = copy.deepcopy(head)
        self.cha_1_head = copy.deepcopy(head)
        self.cha_2_head = copy.deepcopy(head)
        self.par_1_head = copy.deepcopy(head)
        self.par_2_head = copy.deepcopy(head)
        self.par_g_head = copy.deepcopy(head)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.backbone(x)

        # Global branch
        glo_drop, glo = self.batch_dropblock_top(self.global_branch(x))
        glo_drop = self.glo_drop_reduction(self.glo_max_pooling(glo_drop))
        glo = self.glo_reduction(self.glo_avg_pooling(glo))

        # Channel branch
        cha = self.cha_avg_pooling(self.channel_branch(x))
        c1 = self.shared_channel_conv(cha[:, : self.backbone_dim // 2, ...])
        c2 = self.shared_channel_conv(cha[:, self.backbone_dim // 2 :, ...])

        # Part branch
        par = self.partial_branch(x)
        par_max = self.par_max_pooling(par)
        par_avg = self.par_avg_pooling(par)

        p1 = self.par_1_reduction(par_avg[:, :, 0:1, :])
        p2 = self.par_2_reduction(par_avg[:, :, 1:2, :])
        pg = self.par_g_reduction(par_max)

        f_glo_drop = self.glo_drop_head(glo_drop, targets)
        f_glo = self.glo_head(glo, targets)
        f_c1 = self.cha_1_head(c1, targets)
        f_c2 = self.cha_2_head(c2, targets)
        f_p1 = self.par_1_head(p1, targets)
        f_p2 = self.par_2_head(p2, targets)
        f_pg = self.par_g_head(pg, targets)

        if not self.training or targets is None:
            return torch.cat((f_glo_drop, f_glo, f_c1, f_c2, f_p1, f_p2, f_pg), dim=1)

        return (
            [f_glo[0], f_glo_drop[0], f_pg[0]],
            [
                f_glo[1],
                f_c1[1],
                f_c2[1],
                f_p1[1],
                f_p2[1],
                f_pg[1],
            ],
            [
                f_glo[2],
                f_c1[2],
                f_c2[2],
                f_p1[2],
                f_p2[2],
                f_pg[2],
            ],
        )

    def weights_init_shared(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            nn.init.constant_(m.bias, 0.0)
        elif classname.find("Conv2d") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    from utils import read_cfg

    cfg = read_cfg("configs/light_mbn.yml")

    from ptflops import get_model_complexity_info

    model = LightMBN(cfg["model"])

    macs, params = get_model_complexity_info(
        model.eval(),
        (3, 384, 128),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
