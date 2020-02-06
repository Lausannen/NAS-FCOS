import math
import torch
import logging
import torch.nn.functional as F

from torch import nn
from ..retinanet.loss import make_retinanet_loss_evaluator
from ..retinanet.inference import make_retinanet_postprocessor
from ..anchor_generator import make_anchor_generator_retinanet

from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.nas.modeling.micro_heads import MicroHead_v2_retinanet

class RetinaNet_NasHeadModule(torch.nn.Module):
    """
    Module for NAS Head computation based on retina-net. Take features 
    from the backbone and Decoder proposals and losses
    """

    def __init__(self, cfg):
        super(RetinaNet_NasHeadModule, self).__init__()
        config_arch = cfg.SEARCH.DECODER.CONFIG
        decoder_layer_num = cfg.SEARCH.DECODER.NUM_CELLS
        head_repeats = cfg.SEARCH.HEAD.REPEATS
        share_weight_search = cfg.SEARCH.HEAD.WEIGHT_SHARE_SEARCH
        nas_decoder_on = cfg.SEARCH.NAS_DECODER_ON
        logger = logging.getLogger("maskrcnn_benchmark.nas_head")

        if nas_decoder_on:
            if share_weight_search:
                sample_weight_layer = config_arch[decoder_layer_num]
                sample_head_arch = config_arch[decoder_layer_num + 1]
                assert len(config_arch) == decoder_layer_num + 2
            else:
                sample_weight_layer = [0]
                sample_head_arch = config_arch[decoder_layer_num]
                assert len(config_arch) == decoder_layer_num + 1
        else:
            if share_weight_search:
                sample_weight_layer = config_arch[0]
                sample_head_arch = config_arch[1]
                assert len(config_arch) == 2
            else:
                sample_weight_layer = [0]
                sample_head_arch = config_arch[0]
                assert len(config_arch) == 1

        assert len(sample_weight_layer) == 1

        logger.info("Share Weight Level: {}".format(sample_weight_layer[0]))
        logger.info("Sample Head Arch : {}".format(sample_head_arch))

        head = MicroHead_v2_retinanet(sample_weight_layer[0], sample_head_arch, head_repeats, cfg)

        anchor_generator = make_anchor_generator_retinanet(cfg)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.anchor_generator = anchor_generator
        self.anchors = None
        

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are 
                used for computing the predictions. Each tensor in the list 
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During 
                testing, it is an empty dict.
        """

        box_cls, box_regression = self.head(features)
        if self.anchors is None and not isinstance(targets, dict):
            anchors = self.anchor_generator(images, features)
        elif self.anchors is None and isinstance(targets, dict):
            anchors = None          # do not pass anchors since targets have been prepared
        else:         
            anchors = self.anchors  # change back to this with distributed launch

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)        

    def generate_anchors(self, mini_batch_size, crop_size):
        # feature map size of FPN outputs from high resolution to low
        grid_sizes = [(math.ceil(crop_size / r),
                       math.ceil(crop_size / r))
                      for r in (8, 16, 32, 64, 128)]
        self.anchors = self.anchor_generator.get_anchors(mini_batch_size, crop_size, grid_sizes)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}

def build_retinanet_nas_head(cfg):
    return RetinaNet_NasHeadModule(cfg)