import copy
import torch.nn as nn


def get_activations(name: str, *args, **kwargs) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(*args, **kwargs)
    elif name == "SiLU":
        return nn.SiLU(*args, **kwargs)
    elif name == "LeakyReLU":
        return nn.LeakyReLU(*args, **kwargs)
    elif name == "Mish":
        return nn.Mish(*args, **kwargs)
    elif name == "Sigmoid":
        return nn.Sigmoid(*args, **kwargs)
    elif name == "Tanh":
        return nn.Tanh(*args, **kwargs)
    elif name == "Hardswish":
        return nn.Hardswish(*args, **kwargs)
    elif name == "Hardsigmoid":
        return nn.Hardsigmoid(*args, **kwargs)
    elif name == "PReLU":
        return nn.PReLU(*args, **kwargs)
    elif name == "GELU":
        return nn.GELU(*args, **kwargs)
    raise KeyError(f"activation name not support, got unexpected name: {name}")


def cfg_without_name(cfg):
    return {key: value for key, value in cfg.items() if key != "name"}


def cfg_to_activation(cfg):
    return get_activations(
        cfg["name"],
        **cfg_without_name(cfg),
    )
