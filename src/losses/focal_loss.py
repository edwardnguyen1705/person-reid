import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def focal_loss_jit(predict, target, gamma, alpha):
    loss = F.cross_entropy(predict, target, reduction="none")

    return alpha * (1 - torch.exp(-loss)) ** gamma * loss


def focal_loss(predict, target, gamma=1.5, alpha=0.25, reduction="none"):
    gamma = torch.tensor(gamma, dtype=torch.float, device=predict.device)

    alpha = torch.tensor(alpha, dtype=torch.float, device=predict.device)

    loss = focal_loss_jit(predict, target, gamma, alpha)

    if reduction == "none":
        return loss

    loss = loss.mean(dim=0)

    return loss.mean() if reduction == "mean" else loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25, reduction="mean"):
        assert reduction in [
            "sum",
            "mean",
            "none",
        ], "reduction must be mean, sum or none"
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        return focal_loss(predict, target, self.gamma, self.alpha)
