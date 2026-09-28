from __future__ import absolute_import

#from torchvision.transforms import *

#from PIL import Image
import random
import math
#import numpy as np

class RandomErasing(object):

    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class RandomGrayscaleErasing(object):

    def __init__(self, probability: float = 0.2, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        height, width = img.size()[-2], img.size()[-1]
        area = height * width

        for _ in range(100):

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # height / width

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # tl
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                # unbind channel dim
                r, g, b = img.unbind(dim=-3)
                # Weighted average method -> grayscale patch
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # rebind channel
                # erasing
                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]

                return img

        return img