import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .linear_softmax import LinearSoftmax


class CosSoftmax(LinearSoftmax):
    r"""
    margin: 0.25
    scale: 64
    """

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Remove index = -1 in market1501 datasets
        logits = self.sub_center(inputs)

        inputs = logits.clone()

        index = torch.where(targets != -1)[0]

        m_hot = torch.zeros(
            (index.shape[0], inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        # m_hot.shape = (batch_size, feature_dim)

        m_hot.scatter_(1, targets.unsqueeze(dim=1), self.margin)

        inputs[index] -= m_hot

        inputs = inputs.mul(self.scale)

        return inputs, logits.mul(self.scale)
