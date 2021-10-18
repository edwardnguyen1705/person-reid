import copy
import json
import yaml
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping


__all__ = ["read_cfg", "read_json", "write_json", "dict_dotter"]


def read_cfg(path_cfg: str):
    r"""read config yml file, return dict"""

    def update(d, u):
        r"""deep update dict.
        copied from here: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        """
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    cfg = yaml.safe_load(open(path_cfg))

    if "base" not in cfg:
        return cfg

    base_cfg = yaml.safe_load(open(cfg["base"]))
    cfg = update(base_cfg, cfg)

    return cfg


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def dict_dotter(mixed, key="", dots={}):
    if isinstance(mixed, dict):
        for (k, v) in mixed.items():
            dict_dotter(mixed[k], "%s.%s" % (key, k) if key else k)
    else:
        dots[key] = mixed

    return dots
