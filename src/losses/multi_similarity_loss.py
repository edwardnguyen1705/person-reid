import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from losses.util import logsumexp
from metrics.distances import self_cosine_similarity


class MultiSimilarityLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        gamma: float = 1.0,
        eps: float = 1e-6,
    ):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, feature: torch.Tensor, targets: torch.Tensor):
        r"""
        Args:
            - feature (torch.FloatTensor): feature vector of image (batch_size, feature_dim)
            - targets (torch.LongTensor): ground truth of image, it mean vector of pid (batch_size)
        """

        batch_size = feature.shape[0]

        # Calculate similarity matrix
        similarity_mat = self_cosine_similarity(feature)

        targets = targets.view(batch_size, 1).expand(batch_size, batch_size)

        is_pos = targets.eq(targets.t()).float()
        is_neg = targets.ne(targets.t()).float()
        is_pos = is_pos - torch.eye(batch_size, batch_size, device=is_pos.device)

        s_p = similarity_mat * is_pos
        s_n = similarity_mat * is_neg

        logit_p = -self.alpha * (s_p - self.gamma)
        logit_n = self.beta * (s_n - self.gamma)

        return (
            (1 / self.alpha)
            * logsumexp(logit_p, keep_mask=is_pos.bool(), add_one=True, dim=1)
            + (1 / self.beta)
            * logsumexp(logit_n, keep_mask=is_neg.bool(), add_one=True, dim=1)
        ).mean()
