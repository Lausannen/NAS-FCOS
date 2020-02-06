# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import time
import torch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator_retinanet
from maskrcnn_benchmark.modeling.rpn.densebox.loss import RetinaNetLossComputation as DenseBoxLossComputation
from maskrcnn_benchmark.modeling.rpn.retinanet.loss import RetinaNetLossComputation
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.utils.comm import get_rank


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, cfg, crop_size, mode, post_branch, size_divisible=0):
        self.size_divisible = size_divisible
        self.mode = mode
        self.crop_size = crop_size
        self.special_deal = cfg.SEARCH.PREFIX_ANCHOR
        self.post_branch = post_branch
        
        if self.mode == 0:
            if self.post_branch == "retina":
                self.anchor_generator = make_anchor_generator_retinanet(cfg)
                self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))
                self.matcher = Matcher(
                    cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
                    cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
                    allow_low_quality_matches=True
                )
                self.loss_evaluator = RetinaNetLossComputation(
                    cfg, self.matcher, self.box_coder
                )
            elif self.post_branch == "densebox":
                self.loss_evaluator = DenseBoxLossComputation(cfg)
            else:
                raise ValueError("Post {} do not support now".format(self.post_branch))


    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if self.mode == 0:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            if self.special_deal:
                if self.post_branch == "retina":
                    grid_sizes = [(math.ceil(self.crop_size / r),
                                math.ceil(self.crop_size / r))
                                for r in (8, 16, 32, 64, 128)]
                    mini_batch_size = len(targets)
                    anchors = self.anchor_generator.get_anchors(mini_batch_size, self.crop_size, grid_sizes)
                    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
                    labels, regression_targets = self.loss_evaluator.prepare_targets(anchors, targets)
                    # cat labels(list) and regression_targets(list) into one single tensor seperately
                    labels = torch.cat(labels, dim=0)
                    regression_targets = torch.cat(regression_targets, dim=0)
                    targets = { 'labels': labels, 'regression_targets': regression_targets }
                else:
                    strides = [8, 16, 32, 64, 128]
                    feature_sizes = [(math.ceil(self.crop_size / r),
                                    math.ceil(self.crop_size / r))
                                    for r in (8, 16, 32, 64, 128)]
                    points = []
                    for level, size in enumerate(feature_sizes):
                        h, w = size
                        points_per_level = self.generate_points_per_level(
                            h, w, strides[level],
                            torch.device("cpu")
                        )
                        points.append(points_per_level)
                    cls_targets, reg_targets = self.loss_evaluator.prepare_targets(points, targets)
                    cls_targets_flatten = []
                    reg_targets_flatten = []
                    for l in range(len(cls_targets)):
                        cls_targets_flatten.append(cls_targets[l].reshape(-1))
                        reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
                    cls_targets_flatten = torch.cat(cls_targets_flatten, dim=0)
                    reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
                    targets = { 
                        'cls_targets_flatten': cls_targets_flatten,
                        'reg_targets_flatten': reg_targets_flatten
                    }
                    
        elif self.mode == 1:
            feature_list = transposed_batch[0]
            feature_list_zip = zip(*(feature_list))
            feature_list_flatten = []
            for feature_per_level in feature_list_zip:
                feature_per_level = [torch.unsqueeze(xaf, dim=0) for xaf in feature_per_level]
                feature_per_level_batch = torch.cat(feature_per_level, dim=0)
                feature_list_flatten.append(feature_per_level_batch)
            images = feature_list_flatten

            if self.special_deal:
                if self.post_branch == "retina":
                    labels = transposed_batch[1]
                    regression_targets = transposed_batch[2]
                    # cat labels(list) and regression_targets(list) into one single tensor seperately
                    labels = torch.cat(labels, dim=0)
                    regression_targets = torch.cat(regression_targets, dim=0)
                    targets = { 'labels': labels, 'regression_targets': regression_targets }
                else:
                    densebox_labels = list(transposed_batch[1])
                    densebox_regs = list(transposed_batch[2])
                    #TODO: Automatically change num_points_per_level according to crop_size
                    # num_points_per_level = [4096, 1024, 256, 64, 16]
                    num_points_per_level = [2304, 576, 144, 36, 9]
                    for xi in range(len(densebox_labels)):
                        densebox_labels[xi] = torch.split(
                            densebox_labels[xi],
                            num_points_per_level,
                            dim = 0
                        )
                        densebox_regs[xi] = torch.split(
                            densebox_regs[xi],
                            num_points_per_level,
                            dim = 0
                        )
                    densebox_labels_level_first = []
                    densebox_regs_level_first = []
                    for level in range(len(num_points_per_level)):
                        densebox_labels_level_first.append(
                            torch.cat([densebox_labels_per_im[level] for densebox_labels_per_im in densebox_labels]
                                    , dim = 0)
                        )
                        densebox_regs_level_first.append(
                            torch.cat([densebox_regs_per_im[level] for densebox_regs_per_im in densebox_regs]
                                    , dim = 0)
                        )
                    cls_targets_flatten = []
                    reg_targets_flatten = []
                    for xl in range(len(densebox_labels_level_first)):
                        cls_targets_flatten.append(densebox_labels_level_first[xl].reshape(-1))
                        reg_targets_flatten.append(densebox_regs_level_first[xl].reshape(-1, 4))
                    cls_targets_flatten = torch.cat(cls_targets_flatten, dim=0)
                    reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
                    targets = {
                        "cls_targets_flatten": cls_targets_flatten,
                        "reg_targets_flatten": reg_targets_flatten 
                    }
                img_ids = transposed_batch[3]
            else:
                targets = transposed_batch[1]
                img_ids = transposed_batch[2]            
        else:
            raise ValueError("No mode {} for data batch collect_fn".format(self.mode))
        return images, targets, img_ids


    def generate_points_per_level(self, h, w, stride, device):
        start = stride // 2
        shifts_x = torch.arange(
            0, w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        points = torch.stack((shift_x, shift_y), dim=1) + start
        return points
