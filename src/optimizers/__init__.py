import re
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from .params import get_params


def build_optimizers(cfg, model: nn.Module) -> optim.Optimizer:
    assert "name" in cfg
    assert "lr" in cfg

    param_groups = get_params(
        model=model,
        base_lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        bias_lr_factor=cfg["bias_lr_factor"],
        bias_weight_decay=cfg["bias_weight_decay"],
        norm_weight_decay=cfg["norm_weight_decay"],
        overrides=cfg["overrides"],
    )

    if cfg["name"] == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(cfg["adam"]["beta1"], cfg["adam"]["beta2"]),
            amsgrad=cfg["adam"]["amsgrad"],
        )

    elif cfg["name"] == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            momentum=cfg["sgd"]["momentum"],
            dampening=cfg["sgd"]["dampening"],
            nesterov=cfg["sgd"]["nesterov"],
        )

    elif cfg["name"] == "adamW":
        optimizer = optim.AdamW(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(cfg["adamW"]["beta1"], cfg["adamW"]["beta2"]),
        )

    else:
        raise KeyError(
            "config[optimizer][name] error, got unexpected: {}".format(cfg["name"])
        )

    return optimizer
