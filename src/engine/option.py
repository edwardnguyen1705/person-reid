import argparse


def default_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--cfg-source",
        default="configs/sources/peta.yml",
        type=str,
        help="path to data source config",
    )
    parser.add_argument(
        "--cfg-data",
        default="configs/data/default.yml",
        type=str,
        help="path to data hyperparameter config",
    )
    parser.add_argument(
        "--cfg-model",
        default="configs/models/default.yml",
        type=str,
        help="path to model config",
    )
    parser.add_argument(
        "--cfg-loss",
        default="configs/losses/default.yml",
        type=str,
        help="path to loss config",
    )
    parser.add_argument(
        "--cfg-train",
        default="configs/training/default.yml",
        type=str,
        help="path to training hyperparameter config",
    )
    parser.add_argument(
        "--cfg-test",
        default="configs/testing/default.yml",
        type=str,
        help="path to testing hyperparameter config",
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
    parser.add_argument("--distill", action="store_true")

    return parser.parse_args()
