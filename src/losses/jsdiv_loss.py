import torch
import torch.nn as nn
import torch.nn.functional as F

from .kl_loss import KLLoss


__all__ = ["JSDIVLoss"]


class JSDIVLoss(nn.Module):
    def __init__(self, t: int = 16):
        self.t = t
        self.kldiv = KLLoss(t=t)

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        return (self.kldiv(logits_s, logits_t) + self.kldiv(logits_t, logits_s)) / 2
