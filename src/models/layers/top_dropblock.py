# Copied from: https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


from .batch_dropblock import BatchDrop


class BatchDropTop(nn.Module):
    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x ** 2).sum(1)
            act = act.view(b, h * w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]

            mask = []
            for i in range(b):
                rmask = torch.ones(h, device=x.device)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))

            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)

            if visdrop:
                return mask

            x = x * mask
        return x


class BatchFeatureErase_Top(nn.Module):
    r"""
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC
    """

    def __init__(
        self,
        channels,
        bottleneck,
        h_ratio=0.33,
        w_ratio=1.0,
        drop_top=True,
        bottleneck_features=True,
        visdrop=False,
    ):
        super(BatchFeatureErase_Top, self).__init__()

        self.drop_top = drop_top
        self.bottleneck_features = bottleneck_features
        self.visdrop = visdrop

        self.drop_batch_bottleneck = bottleneck(channels, 512)

        if self.drop_top:
            self.drop_batch_drop = BatchDropTop(h_ratio)
        else:
            self.drop_batch_drop = BatchDrop(h_ratio, w_ratio)

    def forward(self, x):
        features = self.drop_batch_bottleneck(x)

        x = self.drop_batch_drop(features, visdrop=self.visdrop)

        if self.visdrop:
            return x  # x is dropmask

        if self.bottleneck_features:
            return x, features
        else:
            return x
