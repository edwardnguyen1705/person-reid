import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import warnings

from typing import Any

from engine.trainers.simple import SimpleTrainer
from engine.option import default_args
from engine.hooks.checkpointer import get_run_id
from utils import (
    gen_run_id,
    get_checkpoint_dir,
    read_cfg,
    merge_cfg,
    check_cfg_conflict,
    select_device,
)


def main_worker(gpu: int, ngpus_per_node: int, args: Any):
    args, cfg = args

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    trainer = SimpleTrainer(args, cfg)
    trainer.train()


if __name__ == "__main__":
    args = default_args()

    check_cfg_conflict(
        [
            read_cfg(args.cfg_source),
            read_cfg(args.cfg_data),
            read_cfg(args.cfg_model),
            read_cfg(args.cfg_train),
            read_cfg(args.cfg_test),
            read_cfg(args.cfg_loss),
        ]
    )

    cfg = merge_cfg(
        [
            read_cfg(args.cfg_source),
            read_cfg(args.cfg_data),
            read_cfg(args.cfg_model),
            read_cfg(args.cfg_train),
            read_cfg(args.cfg_test),
            read_cfg(args.cfg_loss),
        ]
    )

    device = select_device(args.device)

    if args.resume_path == "":
        args.run_id = gen_run_id(cfg["timezone"])
    elif args.resume_run_id != "":
        args.run_id = args.resume_run_id
    else:
        args.run_id = get_run_id(args.resume_path)

    args.checkpoint_dir = get_checkpoint_dir(args.run_id, args.checkpoint_dir)

    trainer = SimpleTrainer(args, cfg, device)
    trainer.train()


"""
python src/train.py --device 0 --cfg-source configs/sources/msmt17.yml \
    --cfg-data configs/data/fast_reid.yml  --cfg-loss configs/losses/fast_reid.yml \
    --cfg-model configs/models/fast_reid.yml --cfg-train configs/training/fast_reid.yml \
    --cfg-test configs/testing/fast_reid.yml --val --val-step 10 --test-from-checkpoint \
    --data-root /home/coder/project/datasets/msmt17/MSMT17_V1
"""
