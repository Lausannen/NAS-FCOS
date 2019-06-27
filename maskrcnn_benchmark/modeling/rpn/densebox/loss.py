"""
This file contains specific functions for computing losses on the FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


INF = 100000000


class RetinaNetLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.RETINANET.LOSS_GAMMA,
            cfg.MODEL.RETINANET.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.bbox_reg_loss_func = IOULoss()
        self.reg_weights_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_points(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_points(self, points, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = points[:, 0], points[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                       (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            points_to_gt_area = area[None].repeat(len(points), 1)
            points_to_gt_area[is_in_boxes == 0] = INF
            points_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            points_to_min_aera, points_to_gt_inds = points_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(points)), points_to_gt_inds]
            labels_per_im = labels_per_im[points_to_gt_inds]
            labels_per_im[points_to_min_aera == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_mask_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        reg_weights = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                        (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(reg_weights)

    def __call__(self, points, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            regression_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)

        if isinstance(targets, dict):
            cls_targets_flatten = targets["cls_targets_flatten"]
            reg_targets_flatten = targets["reg_targets_flatten"]
        else:
            cls_targets, reg_targets = self.prepare_targets(points, targets)
            cls_targets_flatten = []
            reg_targets_flatten = []
            for l in range(len(cls_targets)):
                cls_targets_flatten.append(cls_targets[l].reshape(-1))
                reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            cls_targets_flatten = torch.cat(cls_targets_flatten, dim=0)
            reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)   

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        for l in range(len(box_cls)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)

        pos_inds = torch.nonzero(cls_targets_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            cls_targets_flatten.int()
        ) / (pos_inds.numel() + N)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_mask_targets(reg_targets_flatten)            
            reg_loss = self.bbox_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            reg_weights_loss = self.reg_weights_loss_func(
                centerness_flatten, 
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            reg_weights_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, reg_weights_loss


def make_retinanet_loss_evaluator(cfg):
    loss_evaluator = RetinaNetLossComputation(cfg)
    return loss_evaluator
