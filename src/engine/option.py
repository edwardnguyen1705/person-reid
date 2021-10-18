import argparse


def default_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--cfg",
        default="configs/default.yaml",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Valid after train epoch",
    )
    parser.add_argument(
        "--val-step",
        default=10,
        type=int,
        help="Check val every n train epochs",
    )
    parser.add_argument(
        "--resume-path", default="", type=str, help="Path of point for resume training"
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default="",
    )
    parser.add_argument(
        "--only-model",
        default="",
        type=str,
        help="Only resume weight of model from checkpoint",
    )
    parser.add_argument(
        "--on-memory-dataset",
        action="store_true",
        help="Load all dataset onto memory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="saved/checkpoints",
        type=str,
        help="Path to saved folder",
    )
    parser.add_argument(
        "--data-root",
        default="/content/datasets/market1501",
        type=str,
        help="path to dataset folder",
    )
    parser.add_argument(
        "--background-generator",
        action="store_true",
    )
    parser.add_argument(
        "--data-prefetcher",
        action="store_true",
    )
    parser.add_argument(
        "--test-from-checkpoint",
        action="store_true",
        help="Load all checkpoint and test to find best metric",
    )

    return parser.parse_args()
