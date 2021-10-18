import torch

from typing import Tuple
import torchvision.transforms as T
from torchvision.transforms.transforms import RandomGrayscale

from .to_tensor import ToTensor
from .auto_augmentation import AutoAugment
from .random_2d_translate import Random2DTranslation
from .random_erasing import RandomErasing
from .cutout import Cutout
from .lgt import LGT
from .denormalize import Denormalize


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_transform(
    image_size: Tuple[int, int],
    is_training: bool = True,
    use_autoaugmentation: bool = False,
    use_random_erasing: bool = True,
    use_cutout: bool = False,
    use_random2translate: bool = False,
    use_lgt: bool = False,
    use_random_grayscale: bool = False,
) -> T.Compose:
    if not is_training and use_autoaugmentation:
        raise RuntimeError("AutoAugment only working on training phase")

    res = []

    if is_training:
        # Auto augmentation
        if use_autoaugmentation:
            res.append(T.RandomApply([AutoAugment()], p=0.1))

        if use_random_grayscale:
            res.append(RandomGrayscale(p=0.1))

        # Resize
        res.append(
            T.Resize(
                size=(
                    image_size[0],
                    image_size[1],
                )
            )
        )

        # Random flip
        res.append(T.RandomHorizontalFlip(p=0.5))

        # Random crop
        if use_random2translate:
            res.append(Random2DTranslation(image_size[0], image_size[1]))
        else:
            res.extend(
                [
                    T.Pad(padding=10, fill=0, padding_mode="constant"),
                    T.RandomCrop(
                        size=(
                            image_size[0],
                            image_size[1],
                        )
                    ),
                ]
            )

        # To tensor of range [0, 1]
        res.append(T.ToTensor())
        res.append(T.Normalize(NORM_MEAN, NORM_STD))

        if use_random_erasing:
            res.append(RandomErasing())

        if use_cutout:
            res.append(Cutout())

        if use_lgt:
            res.append(LGT())
    else:
        res.append(
            T.Resize(
                size=(
                    image_size[0],
                    image_size[1],
                )
            )
        )
        res.append(T.ToTensor())
        res.append(T.Normalize(NORM_MEAN, NORM_STD))

    return T.Compose(res)
