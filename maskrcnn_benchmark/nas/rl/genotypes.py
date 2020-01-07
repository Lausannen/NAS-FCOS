from collections import namedtuple

Genotype = namedtuple('Genotype', 'backbone rpn')

OP_NAMES = [
    'sep_conv_3x3',
    'sep_conv_3x3_dil3',
    'sep_conv_5x5_dil6',
    'skip_connect',
    'def_conv_3x3',
]

AGG_NAMES = [
    'psum',
    'cat'
]

HEAD_OP_NAMES = [
    'conv1x1',
    'conv3x3',
    'sep_conv_3x3',
    'sep_conv_3x3_dil3',
    'skip_connect',
    'def_conv_3x3',
]

HEAD_AGG_NAMES = [
    'psum',
    'cat'
]
