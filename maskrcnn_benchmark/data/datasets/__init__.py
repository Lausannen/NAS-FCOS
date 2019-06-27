# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .concat_dataset import ConcatDataset

"""
For now, Pascal VOC and COCO Dataset are be supported by 2 dataload mode

|   Mode    |   Special_Deal    |   Post Branch     |   Annotation
|    0      |       False       |  retina/densebox  |  origin dataload
|    0      |       True        |  retina/densebox  |  precompute GT target in dataloader
|    1      |       False       |  retina/densebox  |  precompute image feature
|    1      |       True        |  retina/densebox  |  precompute image feature and target
"""

__all__ = ["COCODataset", "ConcatDataset"]
