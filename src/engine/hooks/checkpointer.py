import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import shutil
import warnings

from engine.hooks.base import HookBase
from utils import get_model


class Checkpointer(HookBase):
    def __init__(self, train_metrics, val_metrics, checkpoint_dir):
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.checkpoint_dir = checkpoint_dir

        self.train_best_metrics = dict()
        for x, _ in self.train_metrics.items():
            self.train_best_metrics[x] = None

        self.val_best_metrics = {}
        for x, _ in self.val_metrics.items():
            self.val_best_metrics[x] = None

    def _resume_state_dict(self, state, only_model: bool = False):
        self.trainer.model.load_state_dict(state["model"])

        if "model_ema" in state and state["model_ema"] and self.trainer.model_ema:
            self.trainer.model_ema.load_state_dict(state["model_ema"])

        if only_model:
            return

        print(state["epoch"] + 1)

        self.trainer.start_epoch = state["epoch"] + 1
        self.trainer.criterion.load_state_dict(state["criterion"])
        self.trainer.optimizer.load_state_dict(state["optimizer"])
        self.trainer.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.trainer.scaler.load_state_dict(state["scaler"])

        for metric, _ in self.train_metrics.items():
            self.train_best_metrics[metric] = state["train_best_{}".format(metric)]

        for metric, _ in self.val_metrics.items():
            self.val_best_metrics[metric] = state["val_best_{}".format(metric)]

    def _get_state_dict(self):
        state = {
            "run_id": self.trainer.args.run_id,
            "epoch": self.trainer.epoch,
            "model": get_model(self.trainer.model).state_dict(),
            "criterion": self.trainer.criterion.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "scaler": self.trainer.scaler.state_dict(),
            "model_ema": self.trainer.model_ema.state_dict()
            if self.trainer.model_ema is not None
            else None,
        }

        for metric, value in self.train_best_metrics.items():
            state["train_best_{}".format(metric)] = value
        for metric, value in self.val_best_metrics.items():
            state["val_best_{}".format(metric)] = value
        return state

    def _get_best_metric(self, metrics, results, best_metrics):
        save_best = {}
        for metric, is_min_type in metrics.items():
            save_best[metric] = False
            if is_min_type:
                if (
                    best_metrics[metric] is None
                    or best_metrics[metric] >= results[metric]
                ):
                    best_metrics[metric] = results[metric]
                    save_best[metric] = True
            elif (
                best_metrics[metric] is None or best_metrics[metric] <= results[metric]
            ):
                best_metrics[metric] = results[metric]
                save_best[metric] = True
        return save_best, best_metrics

    def _save(self, filename, state_dict):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.checkpoint_dir, filename)
        print(f"Saving {filepath}")

        torch.save(state_dict, filepath)

        if not os.path.exists(filepath):
            warnings.warn("Save but can not found saved file")

    def after_train_epoch(self):
        self.train_save_best, self.train_best_metrics = self._get_best_metric(
            self.train_metrics, self.trainer.result_train_epoch, self.train_best_metrics
        )

    def after_val_epoch(self):
        self.val_save_best, self.val_best_metrics = self._get_best_metric(
            self.val_metrics, self.trainer.result_val_epoch, self.val_best_metrics
        )

    def after_epoch(self):
        state_dict = self._get_state_dict()

        # Save last checkpoint
        self._save("last.pth", state_dict)

        for metric, is_save in self.train_save_best.items():
            if is_save:
                self._save(f"train_best_{metric}.pth", state_dict)

        if self.trainer.check_is_val_epoch(self.trainer.epoch):
            for metric, is_save in self.val_save_best.items():
                if is_save:
                    self._save(f"val_best_{metric}.pth", state_dict)

    def _save_by_copy(self, from_filepath, to_filepath):
        print(f"Copying {to_filepath}")
        shutil.copyfile(from_filepath, to_filepath)

    def before_test(self):
        self.test_best_metrics = self.val_best_metrics
        self.current_checkpoint = os.path.join(self.checkpoint_dir, "last.pth")
        for metric in self.test_best_metrics.keys():
            self._save_by_copy(
                os.path.join(self.checkpoint_dir, f"val_best_{metric}.pth"),
                os.path.join(self.checkpoint_dir, f"best_{metric}.pth"),
            )

    def after_test_step(self):
        test_save_best, self.test_best_metrics = self._get_best_metric(
            self.val_metrics, self.trainer.result_test_step, self.test_best_metrics
        )

        for metric, is_save in test_save_best.items():
            if is_save:
                self._save_by_copy(
                    self.current_checkpoint,
                    os.path.join(self.checkpoint_dir, f"best_{metric}.pth"),
                )

    def resume_to_test(self):
        for metric in self.train_metrics.keys():
            self.current_checkpoint = os.path.join(
                self.checkpoint_dir, f"train_best_{metric}.pth"
            )
            print(f"Resume state dict to test on checkpoint: {self.current_checkpoint}")
            self._resume_state_dict(
                torch.load(self.current_checkpoint),
                only_model=True,
            )
            yield metric

    def resume_from_checkpoint(self, checkpoint_path: str, map_location):
        self._resume_state_dict(
            torch.load(checkpoint_path, map_location=map_location),
            only_model=False,
        )


def get_run_id(checkpoint_path: str):
    return torch.load(checkpoint_path, map_location=torch.device("cpu"))["run_id"]
