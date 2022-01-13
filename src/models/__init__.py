import os
from typing import Type, Union

from .baseline import Baseline
from .light_mbn import LightMBN


def build_model(cfg: dict, num_classes: int) -> Type[Union[Baseline, LightMBN]]:
    if cfg["name"] == "Baseline":
        return Baseline(cfg, num_classes)

    elif cfg["name"] == "LightMBN":
        return LightMBN(cfg, num_classes)

    else:
        raise ValueError(
            "config[model][name] not support, got unexpected: {}".format(cfg["name"])
        )
