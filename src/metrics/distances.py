import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "cosine_similarity",
    "cosine_dist",
    "euclidean_dist",
    "self_cosine_similarity",
    "self_cosine_dist",
    "self_euclidean_dist",
]


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return torch.mm(x, y.t())


def cosine_dist(x: torch.Tensor, y: torch.Tensor, alpha: int = 1) -> torch.Tensor:
    # cosine_distance = 1 - cosine_similarity
    dist = alpha - alpha * cosine_similarity(x, y)
    return dist


def euclidean_dist(
    x: torch.Tensor, y: torch.Tensor, sqrt: bool = False, clip: bool = False
) -> torch.Tensor:
    m, n = x.shape[0], y.shape[0]

    dist = (
        torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    )

    dist = dist - 2 * torch.matmul(x, y.t())

    dist = dist.clamp(min=1e-12) if clip else dist

    dist = dist.sqrt() if sqrt else dist

    return dist


def self_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return torch.mm(x, x.t())


def self_cosine_dist(x: torch.Tensor, alpha: int = 1) -> torch.Tensor:
    # cosine_distance = 1 - cosine_similarity
    dist = alpha - alpha * self_cosine_similarity(x)
    return dist


def self_euclidean_dist(
    x: torch.Tensor, sqrt: bool = False, clip: bool = False
) -> torch.Tensor:
    m = x.shape[0]

    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)

    dist = xx + xx.t() - 2 * torch.matmul(x, x.t())

    dist = dist.clamp(min=1e-12) if clip else dist

    dist = dist.sqrt() if sqrt else dist

    return dist
