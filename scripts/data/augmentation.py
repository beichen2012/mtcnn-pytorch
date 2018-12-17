# coding: utf-8
import random
import numpy as np

def draw_line(img, line_num, line_width):
    h, w, _ = img.shape
    vlines = []
    for line in range(line_num):
        x0 = int(random.uniform(0, w - line_width))
        y0 = int(random.uniform(0, h - line_width))
        x1 = int(random.uniform(0, w - line_width))
        y1 = int(random.uniform(0, h - line_width))
        vline = []
        if x0 == x1:
            ymin = np.min((y0, y1))
            ymax = np.max((y0, y1))
            for y in range(ymin, ymax):
                vline += [(int(x0), int(y))]
        else:
            k = float(y1 - y0) / float(x1 - x0)
            b = y1 - k * x1
            xmin = np.min((x0, x1))
            xmax = np.max((x0, x1))
            for x in range(xmin, xmax):
                y = k * x + b
                vline += [(int(x), int(y))]
        vlines += [vline]
    # draw line
    image = img.copy()
    for vline in vlines:
        pixel = int(random.uniform(0, 255))
        for pts in vline:
            for i in range(0, line_width):
                image[pts[1], pts[0] + i] = pixel
    return image

class RandomMirror(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, image, target):
        _, width, _ = image.shape
        if random.random() < self.p:
            image = image[:, ::-1]
            target = target.copy()
            target[[0, 2]] = width - target[[2, 0]]
        return image, target


class ToPercentCoords(object):
    def __call__(self, image, target):
        height, width, channels = image.shape
        # to x,y,w,h
        target[2] = target[2] - target[0]
        target[3] = target[3] - target[1]
        # div size
        target[0] /= width
        target[2] /= width
        target[1] /= height
        target[3] /= height
        # w,h to log
        target[2] = np.log(target[2])
        target[3] = np.log(target[3])

        return image, target

class SubtractFloatMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, targets):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), targets

class PermuteCHW(object):
    def __call__(self, image, targets):
        image = image.swapaxes(1,2).swapaxes(0,1)
        return image, targets

class DrawRandomLine(object):
    def __init__(self, p=0.5, line_num=4, line_width=1):
        self.p = p
        self.line_num = line_num
        self.line_width = line_width

    def __call__(self, img, label):
        if random.random() < self.p:
            img = draw_line(img, self.line_num, self.line_width)
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, line_num={}, line_width={})'.format(
            self.p, self.line_num, self.line_width)

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