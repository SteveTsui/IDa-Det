# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class Binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale):

        scale = torch.abs(scale)
        
        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * scale

        ctx.save_for_backward(weight, scale)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, scale = ctx.saved_tensors
        
        para_loss = 0.0001
        bin = 0.02
        weight_bin = torch.sign(weight) * bin
        
        gradweight = para_loss * (weight - weight_bin * scale) + (gradOutput * scale)
        
        #pdb.set_trace()
        grad_scale_1 = torch.sum(torch.sum(torch.sum(gradOutput * weight,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)
        
        grad_scale_2 = torch.sum(torch.sum(torch.sum((weight - weight_bin * scale) * weight_bin,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)

        gradMFilter = grad_scale_1 - para_loss * grad_scale_2
        #pdb.set_trace()
        return gradweight, gradMFilter

class BiConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_MFilters(kernel_size)
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        
    def generate_scale(self, kernel_size):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):

        x = self.binfunc(x)

        new_weight = self.binarization(self.weight, self.scale)

        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

