import os
import six
import cv2
import lmdb
import torch
import pickle
import numpy as np
import os.path as osp
from PIL import Image

from utils import deserialize


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=osp.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=2,
        )

        with self.env.begin(write=False) as txn:
            self.length = deserialize(txn.get(b"__len__"))
            self.keys = deserialize(txn.get(b"__keys__"))

    def __getitem__(self, index):
        img, target = None, None

        env = self.env

        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = deserialize(byteflow)

        return (
            cv2.imdecode(np.fromstring(unpacked[0], np.uint8), cv2.IMREAD_COLOR),
            unpacked[1],
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"
