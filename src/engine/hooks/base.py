import os
import sys


class HookBase(object):
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass

    def before_val_epoch(self):
        pass

    def after_val_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def break_step(self):
        pass

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_test_step(self):
        pass

    def after_test_step(self):
        pass

    def end(self):
        pass
