# coding: utf-8
import random
import numpy as np

class RandomMirror(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, image, target):
        _, width, _ = image.shape
        if random.random() < self.p:
            image = image[:, ::-1]
            target = target.copy()
            target[:, [0, 2]] = width - target[:, [2, -2]]
        return image, target


class ToPercentCoords(object):
    def __call__(self, image, target):
        height, width, channels = image.shape
        target[:, 0] /= width
        target[:, 2] /= width
        target[:, 1] /= height
        target[:, 3] /= height

        return image, target

class SubtractFloatMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, jpg, label):
        for t in self.transforms:
            jpg, label = t(jpg, label)
        return jpg, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string