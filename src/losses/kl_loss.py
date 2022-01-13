import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["KLLoss"]


class KLLoss(nn.Module):
    def __init__(self, t: int = 4):
        self.t = t

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.softmax(logits_t / self.t, dim=1)
        return F.kl_div(p_s, p_t, reduction="sum") * (self.t ** 2) / logits_s.shape[0]
