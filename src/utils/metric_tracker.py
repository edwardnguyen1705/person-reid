import os
import torch
import pandas as pd
import torch.distributed as dist

__all__ = ["MetricTracker", "AverageMeter", "ProgressMeter", "DistributedMetricTracker"]


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class DistributedMetricTracker:
    def __init__(self, device, *keys):
        self.device = device
        self.keys = keys
        self.synchronized = False
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        self.synchronized = False
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n

    def __reduce_value(self, value, ReduceOp=dist.ReduceOp.SUM):
        value_tensor = torch.Tensor([value]).to(
            self.device, torch.float32, non_blocking=True
        )
        dist.all_reduce(value_tensor, ReduceOp, async_op=False)
        value = value_tensor.item()
        del value_tensor

        return value

    def avg(self, key):
        if not self.synchronized:
            dist.barrier()
            for key in self.keys:
                self._data.total[key] = self.__reduce_value(
                    self._data.total[key], dist.ReduceOp.SUM
                )
                self._data.counts[key] = self.__reduce_value(
                    self._data.counts[key], dist.ReduceOp.SUM
                )
                self._data.average[key] = self._data.total[key] / self._data.counts[key]
            self.synchronized = True
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
