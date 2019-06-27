# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class MultiScaleResize(object):
    def __init__(self, min_sizes, max_size):
        self.resizers = []
        for min_size in min_sizes:
            self.resizers.append(Resize(min_size, max_size))

    def __call__(self, image, target):
        resizer = random.choice(self.resizers)
        image, target = resizer(image, target)
        return image, target


class RandomCrop(object):
    """
    Pad if image is smaller than the crop size.
    Discard if target has no bbox"""
    def __init__(self, crop_size, discard_prob=0.6):
        self.crop_size = crop_size
        self.discard_prob = discard_prob
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, image, target):
        w, h = image.size
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        while True:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            box = (left, top, left + new_w, top + new_h)
            # should make sure target crop method does not modify itself
            new_target = target.crop(box, remove_empty=True)
            # Attention: If Densebox does not support empty targets, random crop
            # should not provide empty targets
            # if len(new_target) > 0 or random.random() > self.discard_prob:
            if len(new_target) > 0:
                target = new_target
                break
        image = F.crop(image, top, left, new_h, new_w)
        if new_h < self.crop_size or new_w < self.crop_size:
            padding = (0, 0, (self.crop_size - new_w),
                       (self.crop_size - new_h))
            image = F.pad(image, padding=padding)
            target = target.pad(padding)
        return image, target


class Pad(object):
    """Currently using padd provided by structures.image_list.
    This leads to mismatch in output/mask
    Should change for segmentation evaluation"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
