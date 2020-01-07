import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .layer_factory import HEAD_OPS, AGG_OPS
from ..rl.genotypes import HEAD_OP_NAMES, HEAD_AGG_NAMES

from maskrcnn_benchmark.layers import Scale

class MicroHead_v2(nn.Module):
    """
    Simplified head arch which is used to search and construct single-stage
    detector head part
    """

    def __init__(self, share_weights_layer, head_config, repeats, cfg):
        """
        Arguments:
            head_config: head arch sampled by controller
            cfg: global setting info
        """
        super(MicroHead_v2, self).__init__()
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.num_head_layers = cfg.SEARCH.HEAD.NUM_HEAD_LAYERS
        self.num_head = 5
        self.output_concat = cfg.SEARCH.HEAD.OUTPUT_CONCAT
        self.fpn_strides   = cfg.MODEL.RETINANET.ANCHOR_STRIDES
        self.dense_points  = 1
        self.norm_reg_targets  = False
        self.centerness_on_reg = False

        self.share_weights_layer = share_weights_layer

        assert self.share_weights_layer >= 0
        assert self.share_weights_layer <= self.num_head_layers

        # judge whether to have split_weights
        if self.share_weights_layer == 0:
            self.has_split_weights = False
        else:
            self.has_split_weights = True

        # judge whether to have shared_weights
        if self.share_weights_layer == self.num_head_layers:
            self.has_shared_weights = False
        else:
            self.has_shared_weights = True

        if self.has_split_weights:
            self._cls_head_split_ops = nn.ModuleList()
            self._reg_head_split_ops = nn.ModuleList()
            for ind in range(self.num_head):
                cls_empty_head_layer = nn.ModuleList()
                reg_empty_head_layer = nn.ModuleList()
                self._cls_head_split_ops.append(cls_empty_head_layer)
                self._reg_head_split_ops.append(reg_empty_head_layer)

        if self.has_shared_weights:
            self._cls_head_global_ops = nn.ModuleList()
            self._reg_head_global_ops = nn.ModuleList()
        
        agg_size = self.in_channels

        for ind, cell in enumerate(head_config):
            op_index = cell
            op_name = HEAD_OP_NAMES[op_index]
            _cls_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)
            _reg_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)                

            if ind < self.share_weights_layer:
                # do not share weights
                for ind2 in range(self.num_head):
                    self._cls_head_split_ops[ind2].append(copy.deepcopy(_cls_ops))
                    self._reg_head_split_ops[ind2].append(copy.deepcopy(_reg_ops))
            else:
                # share weights
                self._cls_head_global_ops.append(_cls_ops)
                self._reg_head_global_ops.append(_reg_ops)

        final_channel = self.in_channels

        self.cls_logits = nn.Conv2d(
            final_channel, self.num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            final_channel, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            final_channel, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        # initialization
        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                
        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)


    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []

        for l, feature in enumerate(x):
            cls_out = feature
            reg_out = feature

            # obtain initial pools
            if self.has_split_weights:                
                # compute cls_split_feats
                for ops in self._cls_head_split_ops[l]:
                    cls_out = ops(cls_out)

                # compute reg_split_feats
                for ops in self._reg_head_split_ops[l]:
                    reg_out = ops(reg_out)

            if self.has_shared_weights:
                # compute cls_global_feats
                for ops in self._cls_head_global_ops:
                    cls_out = ops(cls_out)

                # compute reg_global_feats
                for ops in self._reg_head_global_ops:
                    reg_out = ops(reg_out)
            
            logits.append(self.cls_logits(cls_out))

            if self.centerness_on_reg:
                centerness.append(self.centerness(reg_out))
            else:
                centerness.append(self.centerness(cls_out))

            bbox_pred = self.scales[l](self.bbox_pred(reg_out))
            if self.norm_reg_targets:
                if self.training:
                    bbox_pred = F.relu(bbox_pred)
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness