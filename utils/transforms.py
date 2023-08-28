import torchvision.transforms.functional as F
import numpy as np


class SquarePad:
    def __call__(self, image):
        s = image.size
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")
