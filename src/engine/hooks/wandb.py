import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import wandb

from utils import Timer
from engine.hooks.base import HookBase


class Wandb(HookBase):
    def __init__(
        self,
        config: dict,
        project: str,
        run_id: str,
        entity: str,
        group: str,
        sync_tensorboard: bool = True,
    ):
        wandb.init(
            config=config,
            project=project,
            id=run_id,
            resume="allow",
            entity=entity,
            group=group,
            sync_tensorboard=sync_tensorboard,
        )
        wandb.run.name = run_id
        wandb.run.save()

    def before_train(self):
        wandb.watch(self.trainer.model, self.trainer.criterion)

        self._total_train_timer = Timer()
        self._total_train_timer.pause()

    def before_train_epoch(self):
        wandb.log(
            {
                "lr": self.trainer.optimizer.param_groups[-1]["lr"],
                "epoch": self.trainer.epoch,
            }
        )

    def after_train_epoch(self):
        wandb.log(
            {
                **dict(
                    (f"train/{key}", value)
                    for key, value in self.trainer.result_train_epoch.items()
                ),
                **{"epoch": self.trainer.epoch},
                "step_total_time": self._total_train_timer.seconds(),
                "step_avg_time": self._total_train_timer.avg_seconds(),
            }
        )

    def before_step(self):
        self._total_train_timer.resume()

    def after_step(self):
        self._total_train_timer.pause()

    def break_step(self):
        if not self._total_train_timer.is_paused():
            self._total_train_timer.pause()

    def after_val_epoch(self):
        if self.trainer.check_is_val_epoch(self.trainer.epoch):
            wandb.log(
                {
                    **dict(
                        (f"val/{key}", value)
                        for key, value in self.trainer.result_val_epoch.items()
                    ),
                    **{"epoch": self.trainer.epoch},
                }
            )

    def after_train(self):
        wandb.log(
            {
                "step_total_time": self._total_train_timer.seconds(),
                "step_avg_time": self._total_train_timer.avg_seconds(),
            }
        )

    def after_test(self):
        wandb.log(
            {
                f"hparams/{key}": value
                for key, value in self.trainer.checkpointer.test_best_metrics.items()
            }
        )

    def end(self):
        wandb.finish()
