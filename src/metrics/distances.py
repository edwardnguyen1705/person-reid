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
    "hamming_distance",
]


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return torch.mm(x, y.t())


def cosine_dist(x: torch.Tensor, y: torch.Tensor, alpha: int = 1) -> torch.Tensor:
    return alpha - alpha * cosine_similarity(x, y)


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


def hamming_distance(x: torch.Tensor, y: torch.Tensor):
    _, dim = x.shape

    x = (x - 0.5) * 2
    y = (y - 0.5) * 2

    return dim - (torch.mm(x, y.t()) + dim) / 2


def self_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return torch.mm(x, x.t())


def self_cosine_dist(x: torch.Tensor, alpha: int = 1) -> torch.Tensor:
    return alpha - alpha * self_cosine_similarity(x)


def self_euclidean_dist(
    x: torch.Tensor, sqrt: bool = False, clip: bool = False
) -> torch.Tensor:
    m = x.shape[0]

    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)

    dist = xx + xx.t() - 2 * torch.matmul(x, x.t())

    dist = dist.clamp(min=1e-12) if clip else dist

    dist = dist.sqrt() if sqrt else dist

    return dist
