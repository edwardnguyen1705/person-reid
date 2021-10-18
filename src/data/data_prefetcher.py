import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch

from utils import batch_to_device


class DataPrefetcher:
    r"""
    https://neuralnet-pytorch.readthedocs.io/en/latest/_modules/neuralnet_pytorch/utils/data_utils.html
    https://zhuanlan.zhihu.com/p/80695364
    """

    def __init__(self, loader, device, enable: bool = True):
        self.loader = iter(loader)
        self.device = device
        self.enable = enable

        if self.enable:
            self.stream = torch.cuda.Stream()

            self.preload()

    def preload(self):
        self.batch = next(self.loader, None)

        if self.batch is None:
            return None

        with torch.cuda.stream(self.stream):
            self.batch = batch_to_device(
                self.batch, device=self.device, non_blocking=True
            )

    def next(self):
        if self.enable:
            torch.cuda.current_stream().wait_stream(self.stream)

            batch = self.batch

            if batch is None:
                raise StopIteration

            self.preload()

            return batch
        else:
            batch = next(self.loader, None)

            if batch is None:
                raise StopIteration

            batch = batch_to_device(batch, device=self.device, non_blocking=True)

            return batch

    def __next__(self):
        return self.next()
