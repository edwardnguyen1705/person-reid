import os
import sys


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import copy
import torch
import warnings
from torch import nn
from typing import Callable
from torch.nn import functional as F

from models.activations import cfg_to_activation
from models.block.build_attention import build_attention

pretrained_urls = {
    "osnet_ain_x1.0": "https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo",
    "osnet_ain_x0.75": "https://drive.google.com/uc?id=1apy0hpsMypqstfencdH-jKIUEFOW4xoM",
    "osnet_ain_x0.5": "https://drive.google.com/uc?id=1KusKvEYyKGDTUBVRxRiz55G31wkihB6l",
    "osnet_ain_x0.25": "https://drive.google.com/uc?id=1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt",
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bn: bool = True,
    ):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity(())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(LightConvStream, self).__init__()
        assert depth >= 1, "depth must be equal to or larger than 1, but got {}".format(
            depth
        )
        layers = []
        layers += [LightConv3x3(in_channels, out_channels, activation=activation)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels, activation=activation)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels: int,
        num_gates=None,
        return_gates=False,
        gate_activation="sigmoid",
        reduction: int = 16,
        layer_norm=False,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))  # type: ignore
        self.relu = copy.deepcopy(activation)
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0
        )
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        reduction=4,
        T=4,
        activation: Callable = nn.ReLU(inplace=True),
        **kwargs,
    ):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels, activation=activation)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [
                LightConvStream(mid_channels, mid_channels, t, activation=activation)
            ]
        self.gate = ChannelGate(mid_channels, activation=activation)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3)  # IN inside residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        blocks,
        layers,
        channels,
        feature_dim=512,
        conv1_IN=False,
        activation: Callable = nn.ReLU(inplace=True),
        attention_cfg: dict = None,
        **kwargs,
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.feature_dim = feature_dim
        self.block = OSBlock

        self.conv1 = ConvLayer(
            3,
            channels[0],
            7,
            stride=2,
            padding=3,
            IN=conv1_IN,
            activation=activation,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks=blocks[0],
            layer=layers[0],
            in_channels=channels[0],
            out_channels=channels[1],
            activation=activation,
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.attention1 = self._make_att_layer(channels[1], attention_cfg)

        self.conv3 = self._make_layer(
            blocks=blocks[1],
            layer=layers[1],
            in_channels=channels[1],
            out_channels=channels[2],
            activation=activation,
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.attention2 = self._make_att_layer(channels[2], attention_cfg)

        self.conv4 = self._make_layer(
            blocks=blocks[2],
            layer=layers[2],
            in_channels=channels[2],
            out_channels=channels[3],
            activation=activation,
        )
        self.conv5 = Conv1x1(channels[3], channels[3], activation=activation)

        self._init_params()

    def _make_att_layer(self, channels, attention_cfg):
        enable = attention_cfg["enable"]
        name = attention_cfg["name"]
        rd_ratio = attention_cfg["rd_ratio"]
        pam_batchnorm = attention_cfg["pam_batchnorm"]
        cam_batchnorm = attention_cfg["cam_batchnorm"]

        if not enable:
            return nn.Identity()

        return build_attention(
            name=name,
            channels=channels,
            cam_batchnorm=cam_batchnorm,
            pam_batchnorm=pam_batchnorm,
            rd_ratio=rd_ratio,
        )  # type: ignore

    def _make_layer(
        self,
        blocks,
        layer,
        in_channels,
        out_channels,
        activation: Callable = nn.ReLU(inplace=True),
    ):
        layers = []
        layers.append(
            blocks[0](
                in_channels,
                out_channels,
                activation=activation,
            )
        )
        for i in range(1, layer):
            layers.append(
                blocks[i](
                    out_channels,
                    out_channels,
                    activation=activation,
                )
            )

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.attention1(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.attention2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        return self.featuremaps(x)


def init_pretrained_weights(model, key=""):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = "TORCH_HOME"
        ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
        DEFAULT_CACHE_DIR = "~/.cache"
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, "checkpoints")
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + "_imagenet.pth"
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(cached_file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.format(
                cached_file
            )
        )
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".format(discarded_layers)
            )


def get_osnet_name(width):
    return "osnet_ain_" + width


def build_osnet_ain(cfg) -> OSNet:
    width = cfg["width"]
    activation = cfg_to_activation(cfg["activation"])
    pretrained = cfg["pretrained"]

    attention_cfg = {
        "enable": cfg["attention"]["enable"],
        "name": cfg["attention"]["name"],
        "rd_ratio": cfg["attention"]["rd_ratio"],
        "cam_batchnorm": cfg["attention"]["cam_batchnorm"],
        "pam_batchnorm": cfg["attention"]["pam_batchnorm"],
    }

    assert width in [
        "x1.0",
        "x0.75",
        "x0.5",
        "x0.25",
    ], "Only support OSNet has width 1.0 or 0.75"

    blocks = {
        "x1.0": [
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock],
        ],
        "x0.75": [
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock],
        ],
        "x0.5": [
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock],
        ],
        "x0.25": [
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock],
        ],
    }

    layers = {
        "x1.0": [2, 2, 2],
        "x0.75": [2, 2, 2],
        "x0.5": [2, 2, 2],
        "x0.25": [2, 2, 2],
    }

    channels = {
        "x1.0": [64, 256, 384, 512],
        "x0.75": [48, 192, 288, 384],
        "x0.5": [32, 128, 192, 256],
        "x0.25": [16, 64, 96, 128],
    }

    model = OSNet(
        blocks=blocks[width],
        layers=layers[width],
        channels=channels[width],
        activation=activation,
        attention_cfg=attention_cfg,
    )

    if pretrained:
        key = get_osnet_name(width)
        if key in pretrained_urls:
            init_pretrained_weights(model, key=key)
        else:
            print("Not found pretrained")

    return model
