from .bn_head import BNHead
from .hasing_head import HashingHead


__all__ = ["build_head"]


def build_head(cfg: dict, in_channels: int, num_classes: int):
    if cfg["name"] == "BNHead":
        return BNHead(cfg["BNHead"], in_channels=in_channels, num_classes=num_classes)
    elif cfg["name"] == "HashingHead":
        return HashingHead(
            cfg["HashingHead"], in_channels=in_channels, num_classes=num_classes
        )
    else:
        raise ValueError("cfg['name'] must be either 'BNHead' or 'HashingHead'")
