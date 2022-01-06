from .market1501 import Market1501
from .msmt17 import MSMT17


__all__ = ["build_datasource"]


def build_datasource(cfg, data_root: str):
    if cfg["name"] == "market1501":
        return Market1501(data_root)
    elif cfg["name"] == "msmt17":
        return MSMT17(data_root)
    else:
        raise ValueError(
            "cfg[name] error, build_datasource got unexpected: %s" % cfg["name"]
        )
