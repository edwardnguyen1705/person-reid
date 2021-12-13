# Copied from: https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
import random
import torch
import torch.nn as nn


class BatchDrop(nn.Module):
    # Random block and set it value to zero for batch
    def __init__(self, h_ratio: float, w_ratio: float):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx : sx + rh, sy : sy + rw] = 0
            x = x * mask
        return x
