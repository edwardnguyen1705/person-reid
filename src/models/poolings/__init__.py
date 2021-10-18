import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch.nn as nn

from models.poolings.gem_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from models.poolings.avg_pooling import AvgPooling2d, FastGlobalAvgPool2d

__poolings = {
    "GeneralizedMeanPooling": GeneralizedMeanPooling,
    "GeneralizedMeanPoolingP": GeneralizedMeanPoolingP,
    "AvgPooling2d": AvgPooling2d,
    "FastGlobalAvgPool2d": FastGlobalAvgPool2d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
}


def build_pooling(name, pooling_size=1) -> nn.Module:
    if name not in list(__poolings.keys()):
        raise KeyError("name error, name must in %s" % (str(list(__poolings.keys()))))
    return __poolings[name](output_size=pooling_size)
