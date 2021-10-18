import os
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager


__all__ = [
    "torch_distributed_zero_first",
    "is_parallel",
    "reduce_value",
    "get_model",
    "convert_state_dict",
    "is_root",
    "check_any",
    "get_local_rank",
    "batch_to_device",
    "get_num_workers",
    "get_batch_size",
]


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    r"""Decorator to make all processes in distributed training wait for each
    local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield

    if local_rank == 0:
        torch.distributed.barrier()


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def reduce_value(value, device, ReduceOp=dist.ReduceOp.SUM):
    value_tensor = torch.Tensor([value]).to(device, torch.float32, non_blocking=True)
    dist.all_reduce(value_tensor, ReduceOp, async_op=False)
    value = value_tensor.item()
    del value_tensor

    return value


def get_model(model):
    return model.module if is_parallel(model) else model


def convert_state_dict(state_dict):
    if all([x.startswith("module.") for x in state_dict.keys()]):
        return dict((key[len("module.") :], value) for key, value in state_dict.items())
    return state_dict


def is_root(multiprocessing_distributed: bool, rank: int, ngpus_per_node: int):
    if not multiprocessing_distributed or (
        multiprocessing_distributed and rank % ngpus_per_node == 0
    ):
        return True
    return False


def check_any(input):
    if isinstance(input, list) or isinstance(input, tuple):
        for x in input:
            check_any(x)
    elif isinstance(input, torch.Tensor):
        if torch.isnan(input).any():
            raise RuntimeError("NAN")
    else:
        raise RuntimeError(
            "Input must be list, tuple or Tensor, not: {}".format(type(input))
        )


def get_local_rank(rank) -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    return rank


def batch_to_device(batch, *args, **kwargs):
    r"""
    Moves a batch to the specified device.

    :param batch:
        a :class:`torch.Tensor` or an iterable of :class:`torch.Tensor`.
    :return:
        a copy of the original batch that on the specified device.
    """
    if isinstance(batch, torch.Tensor) or hasattr(batch, "to"):
        return batch.to(*args, **kwargs)

    if isinstance(batch, list) or isinstance(batch, tuple):
        return [batch_to_device(b, *args, **kwargs) for b in batch]

    if isinstance(batch, dict):
        return dict(
            (key, batch_to_device(value, *args, **kwargs))
            for key, value in batch.items()
        )

    raise RuntimeError("Type of batch not support, got type: %s" % type(batch))


def get_batch_size(batch_size: int, len_dataset: int) -> int:
    return min(batch_size, len_dataset)


def get_num_workers(num_workers: int, batch_size: int, world_size: int):
    return min(
        os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, num_workers  # type: ignore
    )
