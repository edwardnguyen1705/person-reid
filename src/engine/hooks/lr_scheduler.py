import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .base import HookBase
from schedulers import build_lr_scheduler


class LrScheduler(HookBase):
    def __init__(self, cfg, optimizer, total_iterations):
        self.cfg = cfg
        self.lr_scheduler = None
        if self.cfg["enable"]:
            self.lr_scheduler = build_lr_scheduler(cfg, optimizer, total_iterations)

    def after_train_epoch(self):
        if self.lr_scheduler is not None and self.cfg["start"] <= self.trainer.epoch:
            self.lr_scheduler.step()

    def state_dict(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler.state_dict()
        return None

    def load_state_dict(self, state_dict):
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict)
