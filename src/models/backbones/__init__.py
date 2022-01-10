import torch.nn as nn

from typing import Tuple

from .osnet import build_osnet
from .osnet_ain import build_osnet_ain
from .resnet import build_resnet
from .mobilenetv3 import build_mobilenetv3
from .efficientnet import build_efficientnet
from .efficientnetv2_rw import build_efficientnetv2_rw
from .resnest import build_resnest


def build_backbone(cfg) -> Tuple[nn.Module, int, nn.Module]:
    if cfg["name"] == "resnet":
        model = build_resnet(cfg["resnet"])
        return model, model.feature_dim, model.block

    elif cfg["name"] == "resnest":
        model = build_resnest(cfg["resnest"])
        return model, model.feature_dim

    elif cfg["name"] == "osnet":
        model = build_osnet(cfg["osnet"])
        return model, model.feature_dim, model.block

    elif cfg["name"] == "osnet_ain":
        model = build_osnet_ain(cfg["osnet_ain"])
        return model, model.feature_dim, model.block

    elif cfg["name"] == "efficientnet":
        model = build_efficientnet(cfg["efficientnet"])

        return model, model.get_feature_dim()

    elif cfg["name"] == "efficientnetv2_rw":
        model = build_efficientnetv2_rw(cfg["efficientnetv2_rw"])

        return model, model.get_feature_dim()

    elif cfg["name"] == "mobilenetv3":
        model = build_mobilenetv3(cfg["mobilenetv3"])

        return model, model.get_feature_dim()

    else:
        raise KeyError(f"backbone name : {cfg['name']} not support")
