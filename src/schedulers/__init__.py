import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from typing import Dict, Any

from .warmup import WarmupMultiStepLR, WarmupCosineAnnealingLR


def build_lr_scheduler(cfg, optimizer: optim.Optimizer, total_iterations: int):
    if cfg["enable"] == False:
        return None

    if cfg["name"] == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            milestones=cfg["WarmupMultiStepLR"]["steps"],
            gamma=cfg["WarmupMultiStepLR"]["gamma"],
            warmup_factor=cfg["WarmupMultiStepLR"]["warmup_factor"],
            warmup_iters=cfg["WarmupMultiStepLR"]["warmup_iters"],
            warmup_method=cfg["WarmupMultiStepLR"]["warmup_method"],
        )

    elif cfg["name"] == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg["ReduceLROnPlateau"]["factor"],
            patience=cfg["ReduceLROnPlateau"]["patience"],
            min_lr=cfg["ReduceLROnPlateau"]["min_lr"],
        )

    elif cfg["name"] == "MultiStepLR":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["MultiStepLR"]["steps"],
            gamma=cfg["MultiStepLR"]["gamma"],
        )

    elif cfg["name"] == "WarmupCosineAnnealingLR":
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters=cfg["WarmupCosineAnnealingLR"]["max_iters"],
            delay_iters=cfg["WarmupCosineAnnealingLR"]["delay_iters"],
            eta_min_lr=cfg["WarmupCosineAnnealingLR"]["eta_min_lr"],
            warmup_factor=cfg["WarmupCosineAnnealingLR"]["warmup_factor"],
            warmup_iters=cfg["WarmupCosineAnnealingLR"]["warmup_iters"],
            warmup_method=cfg["WarmupCosineAnnealingLR"]["warmup_method"],
        )
    elif cfg["name"] == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["CosineAnnealingLR"]["max_iters"],
            eta_min=cfg["CosineAnnealingLR"]["eta_min_lr"],
        )

    # elif cfg["name"] == "OneCycleLR":
    #     return lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=cfg["OneCycleLR"]["max_lr"],
    #         epochs=cfg["OneCycleLR"]["epochs"],
    #         steps_per_epoch=total_iterations,
    #         anneal_strategy=cfg["OneCycleLR"]["anneal_strategy"],
    #         div_factor=cfg["OneCycleLR"]["div_factor"],
    #     )

    else:
        raise KeyError("cfg[lr_scheduler][name] error")
