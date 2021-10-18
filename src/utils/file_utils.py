import os
import pytz
from datetime import datetime

__all__ = ["gen_run_id", "get_checkpoint_dir"]


def gen_run_id(timezone: str) -> str:
    return datetime.now(pytz.timezone(timezone)).strftime(r"%m%d_%H%M%S")


def get_checkpoint_dir(run_id: str, checkpoint_dir: str = "saved/checkpoints"):
    return os.path.join(checkpoint_dir, run_id)
