import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.util import logsumexp
from metrics.distances import self_cosine_similarity


class CircleLoss(nn.Module):
    r"""
    Circle loss for pairwise labels only
    https://zhuanlan.zhihu.com/p/141484729
    https://arxiv.org/abs/2002.10857
    """

    def __init__(self, margin=0.3, gamma=128):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, feature: torch.Tensor, targets: torch.Tensor):
        r"""
        Args:
            - feature (torch.FloatTensor): feature vector of image (batch_size, feature_dim)
            - targets (torch.LongTensor): ground truth of image, it mean vector of pid (batch_size)
        """

        batch_size = feature.shape[0]

        similarity_mat = self_cosine_similarity(feature)

        targets = targets.view(batch_size, 1).expand(batch_size, batch_size)

        is_pos = targets.eq(targets.t()).float()
        is_neg = targets.ne(targets.t()).float()
        is_pos = is_pos - torch.eye(batch_size, batch_size, device=is_pos.device)

        s_p = similarity_mat * is_pos
        s_n = similarity_mat * is_neg

        alpha_p = torch.relu(1 + self.margin - s_p.detach())
        alpha_n = torch.relu(s_n.detach() + self.margin)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -self.gamma * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)

        loss = F.softplus(
            logsumexp(logit_p, keep_mask=is_pos.bool(), add_one=False, dim=1)
            + logsumexp(logit_n, keep_mask=is_neg.bool(), add_one=False, dim=1)
        )

        zero_rows = torch.where(
            (torch.sum(is_pos, dim=1) == 0) | (torch.sum(is_neg, dim=1) == 0)
        )[0]

        final_mask = torch.ones_like(loss)
        final_mask[zero_rows] = 0

        return (loss * final_mask).sum() / final_mask.sum()
