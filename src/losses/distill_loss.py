import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from losses.wrap_loss import WrapLoss
from losses.jsdiv_loss import JSDIVLoss

__all__ = ["DistillLoss"]


class DistillLoss(nn.Module):
    def __init__(self, cfg):
        super(DistillLoss, self).__init__()
        self.wrap_loss = WrapLoss(cfg["WrapLoss"])
        self.jsdiv_loss = JSDIVLoss(cfg["JSDIVLoss"]["t"])

    def forward(self, s_feat, s_score, s_pred_cls, t_pred_cls, targets):
        loss, loss_item = self.wrap_loss(s_feat, s_score, targets)

        jsdiv_loss = self.jsdiv_loss(s_pred_cls, t_pred_cls.detach())

        loss_item = torch.cat((loss_item, jsdiv_loss.unsqueeze(dim=0)), dim=0)

        return loss + jsdiv_loss, loss_item
