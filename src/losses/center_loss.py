import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.center = nn.parameter.Parameter(
            torch.randn((self.num_classes, self.feature_dim))
        )

    def forward(self, inputs, targets):
        r"""
        Args:
            - inputs (torch.FloatTensor): feature vector of image (batch_size, feature_dim)
            - targets (torch.LongTensor): ground truth of image, it mean vector of pid (batch_size)
        """

        batch_size = inputs.shape[0]
        device = inputs.device

        return (
            (torch.cdist(inputs, self.center))
            * (
                (
                    (targets.unsqueeze(dim=1).expand(batch_size, self.num_classes)).eq(
                        (
                            torch.arange(
                                self.num_classes, device=device, dtype=torch.long
                            )
                        ).expand(batch_size, self.num_classes)
                    )
                ).float()
            )
        ).clamp(min=1e-12, max=1e12).sum() / batch_size
