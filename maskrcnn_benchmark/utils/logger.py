# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import sys
from tensorboardX import SummaryWriter

def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_writer(output_dir, distributed_rank, exp_name=None):
    """Create writer and arch_writer to record infomation for searching
    
    Arguments:
        output_dir (str): path for store file
        distributed_rank (int): only return writer for main process
        exp_name (str): experience file name, if None, create by time

    Returns:
        if main_process: 
            writer (SummaryWriter): tensorboardX writer object
            arch_writer (file_writer) : Record arch
        else:
            None, None
    """
    if distributed_rank != 0:
        return None, None
    else:
        if exp_name == None:
            exp_name = time.strftime('%m_%d_%H_%M_%S')
        writer = SummaryWriter('{}/{}'.format(output_dir, exp_name))
        arch_writer = open('{}/{}/genotypes.out'.format(output_dir, exp_name), 'w')
        return writer, arch_writer

