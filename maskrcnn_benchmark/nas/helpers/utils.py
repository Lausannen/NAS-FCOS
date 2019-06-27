import os
import cv2
import json
import logging
import numpy as np

from heapq import heappushpop, heappop
from shutil import rmtree

import torch

from ..rl.genotypes import OP_NAMES, AGG_NAMES


def load_ckpt(ckpt_path, ckpt_dict):
    logger = logging.getLogger("maskrcnn_benchmark.nas.helpers")
    best_val = epoch_start = 0
    top2dict = dict()
    best_enc = dict()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch', 0)
        top2dict = ckpt.get('top2dict', dict())
        best_enc = ckpt.get('best_enc', dict())
        logger.info(" Found checkpoint at {} with best_val {:.4f}, top2dict {}, #enc {} at epoch {} ".format(
            ckpt_path, best_val, top2dict, len(best_enc), epoch_start
        ))
    return best_val, epoch_start, top2dict, best_enc


class TaskPerformer(object):
    def __init__(self, maxval, delta=0.3):
        """
        Args:
          maxval (float) : initial maximum value
          delta (float) : how large difference (in %) is allowable between curval and maxval

        """
        self.maxval = maxval
        self.delta = delta
        self.scheduler = {100: 0.9, # after k steps, multiply by v
                          200: 0.8,
                          300: 0.7,
                          400: 0.6}
        self.n_steps = 0
        self.decay = 0.99

    def _update_delta(self):
        mult = self.scheduler.get(self.n_steps, 1.)
        self.delta *= mult
        self.n_steps += 1

    def _update_maxval(self, newval):
        self.maxval = self.decay * self.maxval + (1. - self.decay) * newval

    def step(self, newval):
        self._update_delta()
        self._update_maxval(newval)
        if newval > self.maxval:
            #self.maxval = newval
            # self.n_steps += 1
            return True
        else:
            prct = 1. - np.random.uniform(0.0, high=self.delta)
            if newval > (self.maxval * prct):
                return True
            else:
                return False


def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    n_aux_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        if ('aux' in name):
            n_aux_params += n_elem
        n_total_params += n_elem
    return n_total_params, n_total_params - n_aux_params


def deprocess_img(inp, img_scale, img_mean, img_std):
    return (inp * img_std + img_mean) / img_scale


def apply_cmap(inp, cmap):
    return cmap[inp]


def prettify_enc_dec(enc_config, dec_info, dec_config, n_params):
    """Encoder config: [stride2-layer2, stride2-layer3, stride2-layer4] - binary / boolean vector
       Dec Info: skip connections
       Dec Config: [index_1, index_2, op_1, op_2, agg]x3
    """
    header = '#PARAMS\n\n {:3.2f}M'.format(n_params / 1e6)
    enc_desc = '#Encoder: '
    dec_desc = '#Decoder: '


    if enc_config:
        enc_desc += '\n\n'
        info = []
        val_map = {0 : '1',
                   1 : '2'}
        for idx, val in enumerate(enc_config):
            info.append('layer{}-stride={}'.format(str(idx + 2), val_map[val]))
        enc_desc += '\n\n'.join(info)
    for skip, config in zip(dec_info, dec_config):
        skip1, skip2 = skip
        _, _, op1, op2, agg_op = config
        op1 = OP_NAMES[op1]
        op2 = OP_NAMES[op2]
        agg_op = AGG_NAMES[agg_op]
        skip1 = '{{{} ==> {}}}'.format(skip1, op1)
        skip2 = '{{{} ==> {}}}'.format(skip2, op2)
        agg = '{{{}}}'.format(agg_op)
        dec_desc += '\n\n'
        dec_desc += '{} ==> {} <== {}'.format(skip1, agg, skip2)
    return header + '\n\n' + enc_desc + '\n\n' + dec_desc


def config2action(config):
    enc_config, dec_config = config
    """flatten config into list"""
    action = enc_config.copy()
    for block in dec_config:
        action += block
    return action


def action2config(action, enc_end=3, dec_block=3, dec_len=5):
    """reconstruct action"""
    enc_config = action[:enc_end]
    dec_config = []
    start = enc_end
    for i in range(dec_block):
        dec_config.append(action[start:start+dec_len])
        start = start+dec_len
    return enc_config, dec_config


def parse_geno_log(record):
    """parse one line of genotype log."""
    reward_pattern = 'reward: '
    reward_start = record.find(reward_pattern) + len(reward_pattern)
    reward_end = record.find(',')
    reward = float(record[reward_start:reward_end])
    config_pattern = 'genotype: '
    config_start = record.find(config_pattern) + len(config_pattern)
    config = eval(record[config_start:])
    action = config2action(config)
    return reward, action


def prettify_enc(enc_config):
    """Encoder config: [stride2-layer2, stride2-layer3, stride2-layer4] - binary / boolean vector"""
    if enc_config:
        info = []
        val_map = {0 : '1',
                1 : '2'}
        for idx, val in enumerate(enc_config):
            info.append('layer{}-stride={}'.format(str(idx + 2), val_map[val]))
        return '\n'.join(info)
    else:
        return str(enc_config)


def prettify_dec(dec_config):
    """Decoder config: [index_1, index_2, op_1, op_2, agg]

    index_1: """
    #info = []
    #val_map = {0 : '1',
    #           1 : '2'}
    #for idx, val in enumerate(enc_config):
    #    info.append('layer{}-stride={}'.format(str(idx + 2), val_map[val]))
    #return '\n'.join(info)
    return str(dec_config)


def try_except(func):
    """Try / except wrapper

    Args:
      func (lambda) : function to execute
      
    Returns fun output or 0 otherwise
    """
    def wrapper_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError:
            return 0
    return wrapper_func


class Saver():
    """Saver class for managing parameters"""
    def __init__(self, ckpt_dir, best_val=0, condition=lambda x,y: x > y, save_interval=50):
        """
        Args:
            args (dict): dictionary with arguments.
            ckpt_dir (str): path to directory in which to store the checkpoint.
            best_val (float): initial best value.
            condition (function): how to decide whether to save the new checkpoint
                                    by comparing best value and new value (x,y).

        """
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt_dir = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self._counter = 0
        self._save_interval = save_interval

    def _do_save(self, new_val):
        """Check whether need to save"""
        return self.condition(new_val, self.best_val)

    def save(self, new_val, dict_to_save, logger):
        """Save new checkpoint"""
        self._counter += 1
        if self._do_save(new_val):
            logger.info(" New best value {:.4f}, was {:.4f}".format(new_val, self.best_val))
            self.best_val = new_val
            dict_to_save['best_val'] = new_val
            torch.save(dict_to_save, '{}/checkpoint.pth.tar'.format(self.ckpt_dir))
            return True
        elif self._counter % self._save_interval == 0:
            logger.info(" Saving at architecture {}.".format(dict_to_save['arch_id']))
            dict_to_save['best_val'] = self.best_val
            torch.save(dict_to_save, '{}/checkpoint.pth.tar'.format(self.ckpt_dir))
            return False
        return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TopKSummary(object):
    """keep the top k runs"""

    def __init__(self, k, summary_dir):
        self.runs = []
        for i in range(k):
            self.runs.append((-1, None))
        self.summary_dir = summary_dir

    def pushpop(self, run):
        """Update top K runs and remove summary.

        run is a triple of (num_params, epoch, genotype)
        """
        res = heappushpop(self.runs, run)
        if res[0] > -1:
            # remove directories
            epoch = str(res[1][1])
            # val
            try:
                rmtree(os.path.join(self.summary_dir, 'val', epoch))
            except FileNotFoundError:
                pass
            # train loss curve
            try:
                rmtree(os.path.join(self.summary_dir, 'train_loss',
                                'arch', epoch))
            except FileNotFoundError:
                pass
            try:
                # val_miou
                rmtree(os.path.join(self.summary_dir, 'val_miou',
                                'arch', epoch))
            except FileNotFoundError:
                pass


    def state_dict(self):
        return self.runs

    def load_state_dict(self, state):
        self.runs = state
