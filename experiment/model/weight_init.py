#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-07-22 22:28:07 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-07-22 22:28:07 
'''
import torch
from torch.nn import init
import numpy as np

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 and classname.find('ConvTranspose') == -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvTranspose') == -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose') != -1:
        m.weight.data.copy_(get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0]))
