import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .linear_softmax import LinearSoftmax


class CircleSoftmax(LinearSoftmax):
    r"""
    margin: 0.35
    scale: 64
    """

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.sub_center(inputs)

        inputs = logits.clone()

        index = torch.where(targets != -1)[0]

        m_hot = torch.zeros(
            (index.shape[0], inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        # m_hot.shape = (batch_size, feature_dim)
        m_hot.scatter_(1, targets.unsqueeze(dim=1), 1)

        alpha_p = torch.relu(-inputs.detach() + 1 + self.margin)
        alpha_n = torch.relu(inputs.detach() + self.margin)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logits_p = alpha_p * (inputs - delta_p)
        logits_n = alpha_n * (inputs - delta_n)

        inputs[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

        # Remove index == -1 in market1501 datasets
        neg_index = torch.where(targets == -1)[0]
        inputs[neg_index] = logits_n[neg_index]

        inputs = inputs.mul(self.scale)

        return inputs, logits.mul(self.scale).detach()
