import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch

from utils import imread


class ReidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data=None,
        cache=None,
        transform=None,
        transform_lib="albumentations",
    ):
        assert transform_lib in [
            "albumentations",
            "torchvision",
        ], f"transform_lib must in [albumentations, torchvision], not {transform_lib}"

        assert data != None

        self.data = data
        self.cache = cache
        self.transform = transform
        self.transform_lib = transform_lib

    def reset_memory(self):
        if self.cache != None:
            self.cache.reset()
        else:
            raise RuntimeError("not using memory dataset")

    def __len__(self):
        return len(self.data)

    def __getimage__(self, index):
        path, person_id, camera_id, *_ = self.data[index]

        if self.cache != None:
            if self.cache.is_cached(path):
                return self.cache.get(path), person_id, camera_id

            image = imread(
                path, "cv2" if self.transform_lib == "albumentations" else "pillow"
            )
            self.cache.cache(path, image)
        else:
            image = imread(
                path, "cv2" if self.transform_lib == "albumentations" else "pillow"
            )

        return image, person_id, camera_id

    def __getitem__(self, index):
        image, person_id, camera_id = self.__getimage__(index)

        if self.transform is not None:
            if self.transform_lib == "albumentations":
                return self.transform(image=image)["image"], person_id, camera_id
            elif self.transform_lib == "torchvision":
                return self.transform(image), person_id, camera_id

        return image, person_id, camera_id
