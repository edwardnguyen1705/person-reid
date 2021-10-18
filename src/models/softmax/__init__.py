import torch
import torch.nn as nn

from .arcface_softmax import ArcFaceSoftmax
from .circle_softmax import CircleSoftmax
from .cos_softmax import CosSoftmax
from .linear_softmax import LinearSoftmax
from .linear import Linear

# https://hav4ik.github.io/articles/deep-metric-learning-survey
__softmaxs = {
    "ArcFaceSoftmax": ArcFaceSoftmax,
    "CircleSoftmax": CircleSoftmax,
    "CosSoftmax": CosSoftmax,
    "LinearSoftmax": LinearSoftmax,
    "Linear": Linear,
}


def build_softmax(
    name: str,
    in_features: int,
    out_features: int,
    scale: int,
    margin: float,
    k: int = 1,
) -> nn.Module:
    if name in __softmaxs.keys():
        return __softmaxs[name](
            in_features=in_features,
            out_features=out_features,
            scale=scale,
            margin=margin,
            k=k,
        )

    else:
        raise KeyError("head name : {name} not support")
