import random


class Cutout(object):
    def __init__(self, probability=0.5, size=64, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.size = size

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        h = self.size
        w = self.size

        for _ in range(100):
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img
        return img
