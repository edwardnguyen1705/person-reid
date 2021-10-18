import os
import cv2
from PIL import Image
import albumentations as A
from matplotlib import pyplot as plt

__all__ = ["imread"]


def imread(path, lib="cv2"):
    assert lib in ["cv2", "pillow"]

    if not os.path.exists(path):
        raise RuntimeError(f"path don't extsis:  {path}")

    if lib == "cv2":
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    return Image.open(path).convert("RGB")
