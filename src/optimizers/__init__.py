import re
import copy
import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizers(cfg, model: nn.Module) -> optim.Optimizer:
    assert "name" in cfg
    assert "lr" in cfg

    param_groups = model.parameters()

    if "specified_lr" in cfg and cfg["specified_lr"]["enable"] == True:
        assert "layers" in cfg["specified_lr"]
        assert "lr" in cfg["specified_lr"]

        base_params = []
        new_params = []
        for name, module in model.named_children():
            if name not in cfg["specified_lr"]["layers"]:
                base_params += [p for p in module.parameters()]
            else:
                new_params += [p for p in module.parameters()]
        param_groups = [  # type: ignore
            {"params": base_params},
            {"params": new_params, "lr": cfg["specified_lr"]["lr"]},
        ]
    else:
        param_groups = model.parameters()

    base_learning_rate = cfg["lr"]

    if cfg["name"] == "adam":

        return optim.Adam(
            param_groups,
            lr=base_learning_rate,
            weight_decay=cfg["adam"]["weight_decay"],
            betas=(cfg["adam"]["beta1"], cfg["adam"]["beta2"]),
            amsgrad=cfg["adam"]["amsgrad"],
        )

    elif cfg["name"] == "sgd":

        return optim.SGD(
            param_groups,
            lr=base_learning_rate,
            momentum=cfg["sgd"]["momentum"],
            weight_decay=cfg["sgd"]["weight_decay"],
            dampening=cfg["sgd"]["dampening"],
            nesterov=cfg["sgd"]["nesterov"],
        )

    elif cfg["name"] == "adamW":
        return optim.AdamW(
            param_groups,
            lr=base_learning_rate,
            betas=(cfg["adamW"]["beta1"], cfg["adamW"]["beta2"]),
            weight_decay=cfg["adamW"]["weight_decay"],
        )

    else:
        raise KeyError(
            "config[optimizer][name] error, got unexpected: {}".format(cfg["name"])
        )
