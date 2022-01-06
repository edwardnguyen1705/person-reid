import os
import yaml
import copy
from functools import reduce
from itertools import combinations
from collections.abc import Mapping


__all__ = ["read_cfg", "merge_cfg", "check_cfg_conflict"]


def dotter(mixed, key="", dots={}):
    if isinstance(mixed, dict):
        for (k, v) in mixed.items():
            dots = dotter(mixed[k], "%s.%s" % (key, k) if key else k, dots)
    else:
        dots[key] = mixed

    return dots


def update(d, u):
    r"""deep update dict.
    copied from here: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        d[k] = update(d.get(k, {}), v) if isinstance(v, Mapping) else v
    return d


def read_cfg(path_cfg: str):
    assert os.path.exists(path_cfg), path_cfg

    r"""read config yml file, return dict"""

    cfg = yaml.safe_load(open(path_cfg))

    if "base" not in cfg:
        return cfg

    base_cfg = read_cfg(os.path.join(os.path.dirname(path_cfg), cfg["base"]))

    cfg = update(base_cfg, cfg)

    return cfg


def merge_cfg(list_cfg):
    return reduce(lambda a, b: update(a, b), list_cfg, {})


def check_cfg_conflict(list_cfg):
    for a, b in list(combinations([dotter(x, dots=dict()) for x in list_cfg], r=2)):
        for key in a.keys():
            if key == "base":
                continue

            if key in b.keys():
                raise ValueError("Conflict keys key: " + key)
