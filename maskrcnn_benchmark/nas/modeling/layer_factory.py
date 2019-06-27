"""Different custom layers"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.layers.dcn_v2 import DCN

def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     groups=groups, padding=dilation, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


OPS = {
    'skip_connect': lambda C, stride, affine, repeats=1: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 3, 
            affine=affine, dilation=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 12, 
            affine=affine, dilation=6, repeats=repeats),
    'def_conv_3x3': lambda C, stride, affine, repeats=1: DefConv(C, C, 3),
}


AGG_OPS = {
    'psum' : lambda C, stride, affine, repeats=1: ParamSum(C),
    'cat'  : lambda C, stride, affine, repeats=1: ConcatReduce(C, affine=affine, repeats=repeats),
    }


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
    )
    
def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    )

def conv_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1"""
    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        super(SepConv, self).__init__()
        if C_in != C_out:
            assert repeats == 1, "SepConv with C_in != C_out must have only 1 repeat"
        basic_op = lambda: nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), 
                basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DefConv(nn.Module):

    def __init__(self, C_in, C_out, ksize):
        super(DefConv, self).__init__()
        self.dcn = nn.Sequential(DCN(C_in, C_out, ksize, stride=1, padding=ksize // 2, deformable_groups=2),
                                 nn.BatchNorm2d(C_out),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.dcn(x)


def resize(x1, x2, largest=True):
    if largest:
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1, x2
    else:
        raise NotImplementedError


class ParamSum(nn.Module):

    def __init__(self, C):
        super(ParamSum, self).__init__()
        self.a = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.ones(C))

    def forward(self, x, y):
        bsize = x.size(0)
        x, y = resize(x, y)
        return (self.a.expand(bsize, -1)[:, :, None, None] * x +
                self.b.expand(bsize, -1)[:, :, None, None] * y)


class ConcatReduce(nn.Module):
    
    def __init__(self, C, affine=True, repeats=1):
        super(ConcatReduce, self).__init__()
        self.conv1x1 = nn.Sequential(
                            nn.BatchNorm2d(2 * C, affine=affine),
                            nn.ReLU(inplace=False),
                            nn.Conv2d(2 * C, C, 1, stride=1, groups=C, padding=0, bias=False)
                        )

    def forward(self, x, y):
        x, y = resize(x, y)
        z = torch.cat([x, y], 1)
        return self.conv1x1(z)