import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginSubCenter(nn.Module):
    __constants__ = ["in_features", "out_features", "k"]

    in_features: int

    out_features: int

    weight: torch.Tensor

    k: int

    def __init__(self, in_features: int, out_features: int, k: int = 1) -> None:
        super(ArcMarginSubCenter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = Parameter(torch.Tensor(out_features * k, in_features))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.linear(
            input=F.normalize(inputs, p=2), weight=F.normalize(self.weight, p=2)
        )

        inputs = inputs.view(-1, self.out_features, self.k)
        inputs, _ = torch.max(inputs, dim=2)

        return inputs

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, k={}".format(
            self.in_features,
            self.out_features,
            self.k,
        )
