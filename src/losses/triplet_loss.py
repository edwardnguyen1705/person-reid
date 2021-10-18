import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F


from metrics.distances import self_cosine_dist, self_euclidean_dist


def softmax_weights(dist, mask, eps=1e-6):
    diff = dist - torch.max(dist * mask, dim=1, keepdim=True)[0]

    return (torch.exp(diff) * mask) / (
        torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + eps
    )


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg, eps: float = 1e-6):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos, eps=eps)
    weights_an = softmax_weights(-dist_an, is_neg, eps=eps)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.3,
        distance_mode: str = "euclidean",
        hard_mining: bool = True,
        norm_feature: bool = False,
        eps: float = 1e-6,
    ):
        assert distance_mode in [
            "euclidean",
            "cosine",
        ], f"distance_mode not support, must in [euclidean, cosine]"

        if norm_feature and distance_mode == "cosine":
            raise ValueError("norm_feature only support euclidian distance mode")
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_mode = distance_mode
        self.hard_mining = hard_mining
        self.norm_feature = norm_feature
        self.eps = eps

    def forward(self, inputs, targets):
        r"""
        Args:
            - inputs (torch.FloatTensor): feature vector of image (batch_size, feature_dim)
            - targets (torch.LongTensor): ground truth of image, it mean vector of pid (batch_size)
        """

        batch_size = inputs.shape[0]
        device = inputs.device

        if self.distance_mode == "euclidean":
            if self.norm_feature:
                dist_mat = self_euclidean_dist(
                    F.normalize(inputs, p=2, dim=1), sqrt=True, clip=True
                )
            else:
                dist_mat = self_euclidean_dist(inputs, sqrt=True, clip=True)
        elif self.distance_mode == "cosine":
            dist_mat = self_cosine_dist(inputs, alpha=2)
        else:
            raise KeyError(self.distance_mode)

        targets = targets.view(batch_size, 1).expand(batch_size, batch_size)

        is_pos = targets.eq(targets.t()).float()
        is_neg = targets.ne(targets.t()).float()

        if self.hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(
                dist_mat, is_pos, is_neg, self.eps
            )

        if self.margin > 0:
            return F.margin_ranking_loss(
                dist_an,
                dist_ap,
                torch.ones_like(dist_an, device=device),
                margin=self.margin,
            )
        else:
            loss = F.soft_margin_loss(
                dist_an - dist_ap, torch.ones_like(dist_an, device=device)
            )

            if loss == float("Inf"):
                loss = F.margin_ranking_loss(
                    dist_an,
                    dist_ap,
                    torch.ones_like(dist_an, device=device),
                    margin=0.3,
                )

            return loss
