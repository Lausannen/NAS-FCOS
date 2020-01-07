import math
import torch
import logging
import torch.nn.functional as F

from torch import nn
from ..densebox.loss import make_retinanet_loss_evaluator
from ..densebox.inference import make_retinanet_postprocessor

from maskrcnn_benchmark.layers import Scale
from maskrcnn_benchmark.nas.modeling.micro_heads import MicroHead_v2

class NasHeadModule(torch.nn.Module):
    """
    Module for NAS Head computation. Take features from the backbone and 
    Decoder proposals and losses
    """

    def __init__(self, cfg):
        super(NasHeadModule, self).__init__()
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

        head = MicroHead_v2(sample_weight_layer[0], sample_head_arch, head_repeats, cfg)

        box_selector_test = make_retinanet_postprocessor(cfg)
        loss_evaluator = make_retinanet_loss_evaluator(cfg)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.strides = cfg.MODEL.RETINANET.ANCHOR_STRIDES
        self.dense_points = 1

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
            "loss_reg_weights" : loss_reg_weights
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
        points = self.get_dense_locations(points, stride, device)
        return points

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device)
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            print("dense points only support 1, 4, 5")
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations

def build_nas_head(cfg):
    return NasHeadModule(cfg)