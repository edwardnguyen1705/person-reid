import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .linear_softmax import LinearSoftmax


class ArcFaceSoftmax(LinearSoftmax):
    r"""
    margin: 28.6
    scale: 64
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: int = 30,
        margin: float = 0.25,
        k: int = 1,
        *args,
        **kwargs
    ):
        super(ArcFaceSoftmax, self).__init__(
            in_features, out_features, scale, margin, k, *args, **kwargs
        )
        self.margin = math.radians(self.margin)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.sub_center(inputs)

        inputs = logits.clone()

        # Remove index = -1 in market1501 datasets
        index = torch.where(targets != -1)[0]

        m_hot = torch.zeros(
            (index.shape[0], inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        # m_hot.shape = (batch_size, feature_dim)
        m_hot.scatter_(1, targets.unsqueeze(dim=1), self.margin)

        inputs = inputs.acos()

        inputs[index] += m_hot

        inputs = inputs.cos().mul(self.scale)

        return inputs, logits.mul(self.scale).detach()
