import operator
import six
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_factory import conv_bn_relu, conv3x3, OPS, AGG_OPS
from ..rl.genotypes import OP_NAMES, AGG_NAMES
from maskrcnn_benchmark.utils.comm import get_rank

from maskrcnn_benchmark.modeling.backbone.fpn import LastLevelP6P7


class MicroDecoder_v2(nn.Module):

    def __init__(
            self,
            inp_sizes,
            config,
            agg_size=48,
            num_pools=4,
            repeats=1,
            top_blocks=None,
            inter=False):
        super(MicroDecoder_v2, self).__init__()

        inp_sizes = list(inp_sizes)
        ## NOTE: bring all outputs to the same size
        for out_idx, size in enumerate(inp_sizes):
            setattr(self,
                    'adapt{}'.format(out_idx + 1),
                    conv_bn_relu(size, agg_size, 1, 1, 0, affine=True))
            inp_sizes[out_idx] = agg_size

        init_pool = len(inp_sizes)
        self.inter = inter
        mult = 3 if self.inter else 1

        self._ops = nn.ModuleList()
        self._pos = []                # at which position to apply op
        self._pos_stride = []
        self._collect_inds = []       # noting to collect
        self._collect_inds_stride = []
        self._pools = ['l1', 'l2', 'l3', 'l4']
        self._pools_stride = [4, 8, 16, 32]
        for ind, cell in enumerate(config):
            _ops = nn.ModuleList()
            _pos = []
            l1, l2, op0, op1, op_agg = cell[0]
            for pos, op_id in zip([l1, l2], [op0, op1]):
                if pos in self._collect_inds:
                    pos_index = self._collect_inds.index(pos)
                    self._collect_inds.remove(pos)
                    self._collect_inds_stride.pop(pos_index)
                op_name = OP_NAMES[op_id]
                _ops.append(OPS[op_name](agg_size, 1, True, repeats=repeats))
                _pos.append(pos)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                self._pools_stride.append(self._pools_stride[pos])
            # summation
            op_name = AGG_NAMES[op_agg]
            _ops.append(AGG_OPS[op_name](agg_size, 1, True, repeats=repeats))
            self._ops.append(_ops)
            self._pos.append(_pos)
            self._collect_inds.append((init_pool - 1) + (ind + 1) * mult)
            self._collect_inds_stride.append(
                min(self._pools_stride[l1], self._pools_stride[l2])
            )
            self._pools.append('TODO')
            self._pools_stride.append(
                min(self._pools_stride[l1], self._pools_stride[l2])                
            )
        self.info = ' + '.join(self._pools[i] for i in self._collect_inds)
        self.top_blocks = top_blocks

    def judge(self):
        """
        Judge whether the sampled arch can be used to train
        """
        metric = len(self._collect_inds)
        if metric < 3:
            return False
        else:
            return True

    def prettify(self, n_params):
        """ Encoder config: None
            Dec Config:
              ctx: (index, op) x 4
              conn: [index_1, index_2] x 3
        """
        header = '#PARAMS\n\n {:3.2f}M'.format(n_params / 1e6)
        conn_desc = '#Connections:\n' + self.info
        return header + '\n\n' + conn_desc

    def forward(self, x):
        results = []
        feats = []
        for idx, xx in enumerate(x):
            feats.append(getattr(self,
                    'adapt{}'.format(idx + 1),
                    )(xx))
        for pos, ops in zip(self._pos, self._ops):
            assert isinstance(pos, list), "Must be list of pos"
            out0 = ops[0](feats[pos[0]])
            out1 = ops[1](feats[pos[1]])
            if self.inter:
                feats.append(out0)
                feats.append(out1)
            out2 = ops[2](out0, out1)
            feats.append(out2)

        # Get unused layer feature
        unused_collect_inds = self._collect_inds[:-3]
        unused_collect_inds_stride = self._collect_inds_stride[:-3]

        for block_idx, i in enumerate(self._collect_inds[-3:]):
            feats_mid = feats[i]
            for unused_index in unused_collect_inds:
                feats_unused = feats[unused_index]
                feats_resize = F.interpolate(feats_unused, size=feats_mid.size()[2:],
                                mode='bilinear')
                feats_mid = feats_mid + feats_resize
            cell_out = F.interpolate(feats_mid, size=x[3-block_idx].size()[2:],
                                mode='bilinear')
            results.insert(0, cell_out)
        if self.top_blocks is not None:
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        return results


def build_decoder(cfg):
    if cfg.SEARCH.DECODER.VERSION == 2:
        Decoder = MicroDecoder_v2

    # USE P5 to extract P6, P7 features
    top_blocks = LastLevelP6P7(cfg.SEARCH.DECODER.AGG_SIZE, cfg.SEARCH.DECODER.AGG_SIZE)

    return Decoder(cfg.MODEL.BACKBONE.ENCODER_OUT_CHANNELS,
                   cfg.SEARCH.DECODER.CONFIG,
                   agg_size=cfg.SEARCH.DECODER.AGG_SIZE,
                   repeats=cfg.SEARCH.DECODER.REPEATS,
                   top_blocks=top_blocks)
