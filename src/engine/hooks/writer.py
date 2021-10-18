from tqdm.auto import tqdm

from .base import HookBase


class Writer(HookBase):
    def __init__(self, total_iterations: int):
        self.total_iterations = total_iterations

    def before_train_epoch(self):
        print("Epoch: ", self.trainer.epoch)

        self.progress_bar = tqdm(total=self.total_iterations)
        self.progress_bar.set_description(f"Epoch: {self.trainer.epoch}, Train")

    def after_step(self):
        self.progress_bar.set_postfix(
            dict(
                (key, "%.3f" % value)
                for key, value in self.trainer.result_train_step.items()
            )
        )
        self.progress_bar.update(1)

    def break_step(self):
        self.progress_bar.close()

    def after_train_epoch(self):
        self.progress_bar.close()
        for key, value in self.trainer.result_train_epoch.items():
            print("train    {:15s}: {}".format(str(key), value))

    def after_val_epoch(self):
        if self.trainer.check_is_val_epoch(self.trainer.epoch):
            for key, value in self.trainer.result_val_epoch.items():
                print("train    {:15s}: {}".format(str(key), value))

    def after_test_step(self):
        for key, value in self.trainer.result_test_step.items():
            print("train    {:15s}: {}".format(str(key), value))
