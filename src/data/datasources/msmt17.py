import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

import re
from tqdm import tqdm

from base_datasource import BaseDatasource

__all__ = ["MSMT17"]


class MSMT17(BaseDatasource):
    def __init__(self, root, combineall: bool = True):
        train_dir = os.path.join(root, "train")
        test_dir = os.path.join(root, "test")

        list_train_path = os.path.join(root, "list_train.txt")
        list_val_path = os.path.join(root, "list_val.txt")
        list_query_path = os.path.join(root, "list_query.txt")
        list_gallery_path = os.path.join(root, "list_gallery.txt")

        self.pid_container = {}
        self.camid_containter = {}

        (
            self.train,
            self.pid_container["train"],
            self.camid_containter["train"],
        ) = self.process_dir(train_dir, list_train_path, relabel=True)
        (
            self.val,
            self.pid_container["val"],
            self.camid_containter["val"],
        ) = self.process_dir(train_dir, list_val_path, relabel=True)

        (
            self.query,
            self.pid_container["query"],
            self.camid_containter["query"],
        ) = self.process_dir(test_dir, list_query_path, relabel=False)
        (
            self.gallery,
            self.pid_container["gallery"],
            self.camid_containter["gallery"],
        ) = self.process_dir(test_dir, list_gallery_path, relabel=False)

        if combineall:
            self.train += self.val

            self.pid_container["train"] = (
                self.pid_container["train"] | self.pid_container["val"]
            )
            self.camid_containter["train"] = (
                self.camid_containter["train"] | self.camid_containter["val"]
            )

        self.check_exists("train")
        self.check_exists("query")
        self.check_exists("gallery")

    def get_data(self, mode="train"):
        if mode == "train":
            return self.train
        elif mode == "query":
            return self.query
        elif mode == "gallery":
            return self.gallery
        else:
            raise ValueError("mode error")

    def process_dir(self, dir_path: str, list_path: str, relabel: bool = False):
        with open(list_path, "r") as f:
            lines = f.readlines()

        pid_container = set()
        camid_containter = set()

        data = []
        for img_info in lines:
            img_path, pid = img_info.split()
            pid = int(pid)
            camid = (
                int(os.path.splitext(os.path.basename(img_path))[0].split("_")[2]) - 1
            )
            img_path = os.path.join(dir_path, img_path)
            data.append((img_path, pid, camid))

            pid_container.add(pid)
            camid_containter.add(camid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        if relabel:
            data = list(map(lambda x: (x[0], pid2label[x[1]], x[2]), data))

        return data, pid_container, camid_containter

    def get_classes(self):
        return self.pid_container["train"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data-root",
        default="/home/coder/project/datasets/msmt17/MSMT17_V1",
        type=str,
    )
    args = parser.parse_args()

    datasource = MSMT17(args.data_root)
