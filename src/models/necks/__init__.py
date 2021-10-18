import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch.nn as nn
from typing import Union, Callable

from models.necks.bnneck import BNNeck
from models.necks.general_landmarks_neck import GeneralLandmarksNeck
from models.necks.third_place_landmarks_neck import ThirdPlaceLandmarksNeck

__necks = {
    "BNNeck": BNNeck,
    "GeneralLandmarksNeck": GeneralLandmarksNeck,
    "ThirdPlaceLandmarksNeck": ThirdPlaceLandmarksNeck,
    "Identity": nn.Identity,
}


def build_neck(
    name: str,
    in_channels: int,
    out_channels: int,
    type_norm: str,
    bias_freeze: bool,
    activation: Callable,
) -> nn.Module:
    if name not in list(__necks.keys()):
        raise KeyError("name error, name must in %s" % (str(list(__necks.keys()))))
    return __necks[name](
        in_channels=in_channels,
        out_channels=out_channels,
        type_norm=type_norm,
        bias_freeze=bias_freeze,
        activation=activation,
    )
