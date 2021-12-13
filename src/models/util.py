import copy
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return int(new_v)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def get_norm(in_features, type_norm="2d", bias_freeze=False):
    assert type_norm in ["1d", "2d"], "type_norm must be 1d or 2d"
    if type_norm == "1d":
        norm = nn.BatchNorm1d(in_features)
    elif type_norm == "2d":
        norm = nn.BatchNorm2d(in_features)
    else:
        raise ValueError("type_norm not support")

    if bias_freeze:
        norm.bias.requires_grad_(False)
    return norm


def remove_layers_from_state_dict(state_dict: OrderedDict, layers: List[str]):
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not any(key.startswith(x) for x in layers)
    )
