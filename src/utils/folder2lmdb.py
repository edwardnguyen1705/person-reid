import os
import six
import cv2
import lmdb
import pickle
import numpy as np
import os.path as osp
from PIL import Image

import torch.utils.data as data

__all__ = ["raw_reader", "binary_2_cv2image", "serialize", "deserialize"]


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def binary_2_cv2image(binary):
    return cv2.cvtColor(
        cv2.imdecode(np.fromstring(binary, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )


def serialize(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    # return pa.serialize(obj).to_buffer()
    return pickle.dumps(obj)


def deserialize(bytes):
    return pickle.loads(bytes)


if __name__ == "__main__":
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from models.yolov4 import Yolov4
    from data.datasources import VOC2007
    from data.datasets import ImageDataset_Yolov4
    from utils.image import imread

    from torch.utils.data.dataloader import DataLoader

    datasource = VOC2007(
        "/home/hien/Documents/datasets", download=False, extract=False, use_tqdm=True
    )

    name = "train"

    lmdb_path = osp.join(".", "%s.lmdb" % name)

    db = lmdb.open(
        lmdb_path,
        subdir=False,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)

    write_frequency = 100

    list_key = []

    for i in range(len(datasource.get_data("train"))):
        img_path, boxes, labels, difficulty = datasource.get_data("train")[i]

        image_binary = raw_reader(img_path)

        if image_binary != b"":
            targets = np.concatenate(
                (
                    np.array(boxes),
                    np.expand_dims(np.array(labels), axis=-1),
                    np.expand_dims(np.array(difficulty), axis=-1),
                ),
                axis=-1,
            )

            key = os.path.basename(img_path)

            list_key.append(key)

            txn.put(
                u"{}".format(key).encode("ascii"),
                serialize((image_binary, targets)),
            )

            if i % write_frequency == 0:
                print("[%d/%d]" % (i, len(datasource.get_data("train"))))

                txn.commit()

                txn = db.begin(write=True)

    txn.commit()

    keys = [u"{}".format(k).encode("ascii") for k in list_key]

    with db.begin(write=True) as txn:
        txn.put(b"__keys__", serialize(keys))
        txn.put(b"__len__", serialize(len(keys)))

    db.sync()
    db.close()

    dataset = ImageFolderLMDB("train.lmdb")

    for i in range(len(dataset)):
        dataset[i]
