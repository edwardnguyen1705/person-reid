from torch.nn.parallel import DistributedDataParallel

from .base import HookBase


class FreezeLayers(HookBase):
    def __init__(self, model, layers, epoch):
        if isinstance(model, DistributedDataParallel):
            model = model.module

        self.model = model
        self.layers = layers
        self.epoch = epoch

    def _freeze(self):
        print(f"Freeze: {self.layers}")
        for name, module in self.model.named_children():
            if name in self.layers:
                # Change params freeze
                for param in module.parameters():
                    param.requires_grad = False
                # Change BN in freeze layers to eval mode
                module.eval()

    def _unfreeze(self):
        print(f"Un-freeze: {self.layers}")
        for name, module in self.model.named_children():
            if name in self.layers:
                for param in module.parameters():
                    param.requires_grad = True
                module.train()

    def before_epoch(self):
        epoch = self.trainer.epoch

        if epoch == 1:
            self._freeze()
        if epoch == self.epoch:
            self._unfreeze()
