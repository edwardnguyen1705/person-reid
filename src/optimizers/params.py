import copy
import torch
import torch.nn as nn

from typing import Optional, Set, List, Dict, Any

__all__ = ["get_params", "NORM_MODULE_TYPES"]

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_params(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    bias_lr_factor: Optional[float] = None,
    bias_weight_decay: Optional[float] = None,
    norm_weight_decay: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    r"""
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        bias_lr_factor: multiplier of lr for bias parameters.
        bias_weight_decay: override weight decay for bias parameters
        norm_weight_decay: override weight decay for params in normalization layers
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
    For common detection models, ``norm_weight_decay`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.
    Example:
    ::
        torch.optim.SGD(
            get_params(
                model,
                norm_weight_decay=0
            ),
            lr=0.01,
            weight_decay=1e-4,
            momentum=0.9
        )
    """
    if overrides is None:
        overrides = {}

    defaults = {"lr": base_lr, "weight_decay": weight_decay}
    # bias
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if bias_weight_decay is not None:
        bias_overrides["weight_decay"] = bias_weight_decay
    if len(bias_overrides):
        overrides["bias"] = bias_overrides

    memo: Set[torch.nn.parameter.Parameter] = set()
    params: List[Dict[str, Any]] = []

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):

            if not value.requires_grad:
                continue

            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            if isinstance(module, NORM_MODULE_TYPES) and norm_weight_decay is not None:
                hyperparams["weight_decay"] = norm_weight_decay

            hyperparams.update(overrides.get(module_param_name, {}))

            params.append({"params": [value], **hyperparams})

    return params
