import math
import random

import torchvision.transforms.functional as F


class LGT(object):
    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        img_gray = F.rgb_to_grayscale(img=img, num_output_channels=3)

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                img[0, x1 : x1 + h, y1 : y1 + w] = img_gray[0, x1 : x1 + h, y1 : y1 + w]
                img[1, x1 : x1 + h, y1 : y1 + w] = img_gray[1, x1 : x1 + h, y1 : y1 + w]
                img[2, x1 : x1 + h, y1 : y1 + w] = img_gray[2, x1 : x1 + h, y1 : y1 + w]

                return img
        return img
