import torch
import torch.nn as nn


class CAM(nn.Module):
    r"""Channel attention module from 'ABD-Net: Attentive but Diverse Person Re-Identification'
    - Same way as 'Layer Attention Module from Single Image Super-Resolution via Holistic Attention Network'
    """

    def __init__(self, channels: int, batchnorm: bool = False):
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # type: ignore
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channels) if batchnorm else nn.Identity()

    def forward(self, x):
        r"""
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        B, C, H, W = x.shape

        proj_query = x.view(B, C, -1)

        proj_key = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)

        proj_value = x.view(B, C, -1)

        out = torch.bmm(attention, proj_value)

        out = out.view(B, C, H, W)

        out = self.bn(self.gamma * out) + x

        return out


class PAM(nn.Module):
    r"""Position attention module from 'ABD-Net: Attentive but Diverse Person Re-Identification'
    - From: Self-Attention GAN
    """

    def __init__(self, channels: int, batchnorm: bool = False):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=channels, out_channels=channels // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=channels, out_channels=channels // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # type: ignore
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channels) if batchnorm else nn.Identity()

    def forward(self, x):
        r"""
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        B, C, H, W = x.shape

        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1).contiguous()

        proj_key = self.key_conv(x).view(B, -1, W * H)

        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(B, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(B, C, H, W)

        out = self.bn(self.gamma * out) + x

        return out
