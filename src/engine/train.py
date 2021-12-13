import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import warnings
import torch.multiprocessing as mp

from typing import Any

from engine.trainers.simple import SimpleTrainer
from engine.option import default_args
from engine.hooks.checkpointer import get_run_id
from utils import gen_run_id, get_checkpoint_dir, read_cfg


def main_worker(gpu: int, ngpus_per_node: int, args: Any):
    args, cfg = args

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    trainer = SimpleTrainer(args, cfg)
    trainer.train()


if __name__ == "__main__":
    args = default_args()

    cfg = read_cfg(args.cfg)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.resume_path == "":
        args.run_id = gen_run_id(cfg["timezone"])

    elif args.resume_run_id != "":
        args.run_id = args.resume_run_id
    else:
        args.run_id = get_run_id(args.resume_path)
    args.checkpoint_dir = get_checkpoint_dir(args.run_id, args.checkpoint_dir)

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, (args, cfg)),
        )
    else:
        args.rank = -1
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, (args, cfg))
