import torch
import torch.nn as nn


class IBN(nn.Module):
    r"""
    https://medium.com/syncedreview/facebook-ai-proposes-group-normalization-alternative-to-batch-normalization-fb0699bffae7
    Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes: int, ratio: float = 0.5):
        super(IBN, self).__init__()
        self.half_planes = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half_planes, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half_planes)

    def forward(self, x):
        split = torch.split(x, self.half_planes, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)
