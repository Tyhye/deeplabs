#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-07-31 18:40:25 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-07-31 18:40:25 
'''
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):

    def __init__(self, input_channels=2048, feature_channels=256, rates=[6, 12, 18, 24], pretrained=True):
        super(ASPP, self).__init__()
        # tmp =
        self.convs = nn.ModuleList([])
        self.convs.append(nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)))

        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(input_channels, feature_channels, kernel_size=3,
                          padding=rate, dilation=rate),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)))

    def forward(self, x):
        afs = [conv(x) for conv in self.convs]
        return afs
