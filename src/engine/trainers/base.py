import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import weakref
import torch.nn as nn
import torch.distributed as dist
from typing import Callable

from utils import is_root, MetricTracker


class BaseTrainer(object):
    def __init__(self, args, device):
        self._hooks = []
        self.args = args
        self.device = device

        self.cuda = device.type != "cpu"

    def register_hooks(self, hooks):
        self.train_metrics = self.get_train_metric_dict()
        self.train_tracker = MetricTracker(*list(self.train_metrics.keys()))

        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            # To avoid circular reference
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int):
        self.before_train()

        for self.epoch in range(start_epoch, max_epoch + 1):
            self.before_epoch()

            self.before_train_epoch()
            self.batch_idx = 0

            while True:
                try:
                    self.before_step()
                    self.result_train_step = self.run_train_step()
                    self.after_step()
                    self.batch_idx += 1
                except Exception as e:
                    self.break_step()
                    if not isinstance(e, StopIteration):
                        raise Exception(e)
                    break

            self.after_train_epoch()

            if self.check_is_val_epoch(self.epoch):
                self.before_val_epoch()
                self.result_val_epoch = self.run_val_epoch()
                self.after_val_epoch()

            self.after_epoch()

        self.after_train()

        if self.check_is_test():
            self.before_test()
            self.run_test()
            self.after_test()

        self.end()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

        if self.cuda:
            torch.cuda.empty_cache()

    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch()

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def before_train_epoch(self):
        for h in self._hooks:
            h.before_train_epoch()

        self.train_tracker.reset()

    def after_train_epoch(self):
        self.result_train_epoch = self.train_tracker.result()

        for h in self._hooks:
            h.after_train_epoch()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

        for metric in self.train_metrics.keys():
            assert metric in self.result_train_step

            self.train_tracker.update(key=metric, value=self.result_train_step[metric])

    def break_step(self):
        for h in self._hooks:
            h.break_step()

    def before_val_epoch(self):
        for h in self._hooks:
            h.before_val_epoch()

    def after_val_epoch(self):
        for h in self._hooks:
            h.after_val_epoch()

    def before_test(self):
        for h in self._hooks:
            h.before_test()

    def after_test(self):
        for h in self._hooks:
            h.after_test()

    def before_test_step(self):
        for h in self._hooks:
            h.before_test_step()

    def after_test_step(self):
        for h in self._hooks:
            h.after_test_step()

    def end(self):
        for h in self._hooks:
            h.end()

    def check_is_val_epoch(self, epoch: int):
        return False

    def check_is_test(self):
        return False

    def run_test(self):
        pass

    def run_train_step(self):
        raise NotImplementedError

    def run_val_epoch(self):
        raise NotImplementedError

    def build_datasource(self):
        raise NotImplementedError

    def build_train_loader(self):
        raise NotImplementedError

    def build_test_loader(self):
        raise NotImplementedError

    def build_hooks(self):
        raise NotImplementedError

    def resume_or_load(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_optimizers(self):
        raise NotImplementedError

    def build_losses(self):
        raise NotImplementedError

    def get_train_metric_dict(self):
        raise NotImplementedError

    def get_val_metric_dict(self):
        raise NotImplementedError
