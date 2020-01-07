# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone.backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..rpn.retinanet.retinanet import build_retinanet
from ..rpn.densebox.densebox import build_densebox
from ..rpn.nas_head.nas_head import build_nas_head
from maskrcnn_benchmark.nas.modeling.micro_decoders import build_decoder


class Head(nn.Module):
    def __init__(self, fpn, rpn):
        super(Head, self).__init__()
        self.fpn = fpn
        self.rpn = rpn

    def forward(self, images, features, targets):
        features = self.fpn(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            return losses
        return proposals


class SingleStageDetector(nn.Module):
    """
    Main class for Single Stage Detector. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - fpn
    = rpn
    """

    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()

        self.encoder = build_backbone(cfg)
        if cfg.MODEL.RETINANET_ON:
            rpn = build_retinanet(cfg)
            self.rpn_name = 'retinanet'
        elif cfg.MODEL.DENSEBOX_ON:
            if cfg.SEARCH.NAS_HEAD_ON:
                rpn = build_nas_head(cfg)
                self.rpn_name = 'nas_head'
            else:
                rpn = build_densebox(cfg)
                self.rpn_name = 'densebox'
        else:
            rpn = build_rpn(cfg)
        fpn = build_decoder(cfg)
        self.decoder = Head(fpn, rpn)

    def generate_anchors(self, mini_batch_size, img_size):
        if self.rpn_name == 'retinanet':
            self.decoder.rpn.generate_anchors(mini_batch_size,
                                              img_size)

    def reset_anchors(self):
        self.decoder.rpn.anchors = None

    def forward(self, images, targets=None, features=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            features (list[Tensor]): encoder output features (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if features is None:
            images = to_image_list(images)
            features = self.encoder(images.tensors)
        return self.decoder(images, features, targets)
