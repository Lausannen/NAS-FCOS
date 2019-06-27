import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class DenseBoxHead(torch.nn.Module):

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DenseBoxHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        # assert cfg.MODEL.BACKBONE.OUT_CHANNELS == cfg.SEARCH.DECODER.AGG_SIZE
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
                )))
        return logits, bbox_reg, centerness


class DenseBoxModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(DenseBoxModule, self).__init__()

        head = DenseBoxHead(cfg)

        box_selector_test = make_retinanet_postprocessor(cfg)

        loss_evaluator = make_retinanet_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.strides = cfg.MODEL.RETINANET.ANCHOR_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)

        if not isinstance(targets, dict):
            points = self.generate_points(features)
        else:
            points = None

        if self.training:
            return self._forward_train(
                points, box_cls,
                box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                points, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, points, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_reg_weights = self.loss_evaluator(
            points, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_densebox_cls": loss_box_cls,
            "loss_densebox_reg": loss_box_reg,
            "loss_reg_weights": loss_reg_weights
        }
        return None, losses

    def _forward_test(self, points, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            points, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def generate_points(self, features):
        points = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            points_per_level = self.generate_points_per_level(
                h, w, self.strides[level],
                feature.device
            )
            points.append(points_per_level)
        return points

    def generate_points_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        points = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return points

def build_densebox(cfg):
    return DenseBoxModule(cfg)
