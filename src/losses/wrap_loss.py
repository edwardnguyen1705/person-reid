import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


from losses.cross_entropy import CrossEntropy
from losses.triplet_loss import TripletLoss
from losses.cosface_loss import CosFace
from losses.multi_similarity_loss import MultiSimilarityLoss
from losses.circle_loss import CircleLoss
from losses.focal_loss import FocalLoss


LOSSES = {
    "CrossEntropy": CrossEntropy,
    "TripletLoss": TripletLoss,
    "CosFace": CosFace,
    "MultiSimilarityLoss": MultiSimilarityLoss,
    "CircleLoss": CircleLoss,
    "FocalLoss": FocalLoss,
}


class LossType(enum.Enum):
    RANKING_LOSS = 1
    IDENTITY_LOSS = 2


class WrapLoss(nn.Module):
    def __init__(self, config):
        super(WrapLoss, self).__init__()

        self.loss_func = nn.ModuleList([])
        self.loss_scale = []
        self.loss_type = []
        try:
            for loss_ele in config["name"].replace(" ", "").split("+"):
                scale, loss_name = loss_ele.split("*")
                assert (
                    loss_name in LOSSES
                ), "name not support, got unexpected: {}".format(loss_name)

                args, loss_type = self.__config_to_args(config, loss_name)
                self.loss_func.append(LOSSES[loss_name](**args))
                self.loss_scale.append(float(scale))
                self.loss_type.append(loss_type)
        except:
            raise RuntimeError("config[name] can't parse")

    def forward(self, feat, score, targets):
        losses, id_loss, ranking_loss = 0, 0, 0

        for func, scale, type in zip(self.loss_func, self.loss_scale, self.loss_type):
            if type == LossType.IDENTITY_LOSS:
                loss = (
                    sum([func(x, targets) for x in score])
                    if isinstance(score, list) or isinstance(score, tuple)
                    else func(score, targets)
                )
                id_loss += loss

            else:
                loss = (
                    sum([func(x, targets) for x in feat])
                    if isinstance(feat, list) or isinstance(feat, tuple)
                    else func(feat, targets)
                )
                ranking_loss += loss

            losses += scale * loss

        return (
            losses,
            torch.stack((id_loss, ranking_loss)).detach(),
        )

    def __config_to_args(self, config: dict, name: str) -> Tuple[dict, LossType]:
        assert name in LOSSES, "name not support, got unexpected: {}".format(name)

        if name == "CrossEntropy":
            return {
                "epsilon": config["cross_entropy"]["epsilon"],
                "reduction": config["cross_entropy"]["reduction"],
            }, LossType.IDENTITY_LOSS
        elif name == "TripletLoss":
            return {
                "margin": config["triplet"]["margin"],
                "distance_mode": config["triplet"]["distance_mode"],
                "hard_mining": config["triplet"]["hard_mining"],
                "norm_feature": config["triplet"]["norm_feature"],
                "eps": config["triplet"]["eps"],
            }, LossType.RANKING_LOSS
        elif name == "CosFace":
            return {
                "margin": config["cosface"]["margin"],
                "gamma": config["cosface"]["gamma"],
            }, LossType.RANKING_LOSS
        elif name == "MultiSimilarityLoss":
            return {
                "alpha": config["multi_similarity"]["alpha"],
                "beta": config["multi_similarity"]["beta"],
                "gamma": config["multi_similarity"]["gamma"],
                "eps": config["multi_similarity"]["eps"],
            }, LossType.RANKING_LOSS
        elif name == "CircleLoss":
            return {
                "margin": config["circle"]["margin"],
                "gamma": config["circle"]["gamma"],
            }, LossType.RANKING_LOSS
        elif name == "FocalLoss":
            return {
                "gamma": config["focal"]["gamma"],
                "alpha": config["focal"]["alpha"],
                "reduction": config["focal"]["reduction"],
            }, LossType.IDENTITY_LOSS
        else:
            raise ValueError("name not support, got unexpected: {}".format(name))
