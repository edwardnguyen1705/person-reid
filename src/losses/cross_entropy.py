import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        epsilon (float): weight.
    """

    def __init__(self, epsilon=0.1, reduction="sum"):
        assert reduction in [
            "sum",
            "mean",
            "none",
        ], "reduction must be mean, sum or none"
        super(CrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        r"""
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        num_classes = inputs.shape[1]

        log_prob = self.logsoftmax(inputs)

        with torch.no_grad():
            one_hot = F.one_hot(targets, num_classes)
            one_hot = (1 - self.epsilon) * one_hot + self.epsilon / num_classes

        loss = -one_hot * log_prob

        if self.reduction == "none":
            return loss

        with torch.no_grad():
            non_zero_cnt = max(loss.sum(dim=1).nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum(dim=0) / non_zero_cnt

        return loss.mean() if self.reduction == "mean" else loss.sum()
