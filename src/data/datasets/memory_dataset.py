import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch

torch.multiprocessing.set_sharing_strategy("file_system")


__all__ = ["DatasetCache"]


class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError(
                "Data caching is disabled and get funciton is unavailable! Check your config."
            )
        return self._dict[str(key)]

    def cache(self, key, value):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = value
