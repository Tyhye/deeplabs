#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-07-24 00:12:19 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-07-24 00:12:19 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnetdeeplabv3 import resnet18, resnet34, resnet50, resnet101, resnet152
from .aspp import ASPP
from .weight_init import weights_init_kaiming, weights_init_classifier


def freeze_bn(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('BatchNorm') != -1:
        m.eval()


class DeepLabV3Model(nn.Module):

    def __init__(self, classes_num=2, feature_channels=256,
                 rates=[6, 12, 18], mg=[1, 2, 4], output_stride=8,
                 basenet=resnet50, pretrained=True):

        super(DeepLabV3Model, self).__init__()
        self.conv = basenet(pretrained=pretrained, mg=mg,
                            output_stride=output_stride)

        self.aspp = ASPP(input_channels=self.conv.outchannels,
                         feature_channels=feature_channels,
                         rates=rates)
        self.global_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.conv.outchannels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True))

        self.feature = nn.Sequential(
            nn.Conv2d(feature_channels*(2+len(rates)),
                      feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            # nn.ReLU()
        )
        self.cls_layer = nn.Conv2d(
            feature_channels, classes_num, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        afs = self.aspp(x)
        gf = self.global_feature(x)
        gf = F.upsample(gf, size=x.size()[2:],
                        mode="bilinear", align_corners=True)
        afs.append(gf)
        x = torch.cat(afs, dim=1)
        x = self.feature(x)
        x = self.cls_layer(x)
        return x

    def freeze_bn(self):
        self.apply(freeze_bn)

# class Resnet_FCN(nn.Module):

#     def __init__(self, classes_num=1000, feature_channels=512, resnettype=resnet18):
#         super(Resnet_FCN, self).__init__()

#         self.base_conv = resnettype(pretrained=True)

#         self.feature_conv = nn.Sequential(
#             nn.Conv2d(self.base_conv.expansion * 512, feature_channels,
#                       kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm2d(feature_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.score_conv = nn.Conv2d(
#             feature_channels, classes_num, kernel_size=1, stride=1)
#         self.upscore = nn.ConvTranspose2d(
#             classes_num, classes_num, kernel_size=64, stride=32, padding=16, bias=False)

#         self.feature_conv.apply(weights_init_kaiming)
#         self.score_conv.apply(weights_init_classifier)
#         self.upscore.apply(weights_init_classifier)

#     def forward(self, x):
#         feature = self.base_conv(x)
#         feature = self.feature_conv(feature)
#         score = self.score_conv(feature)
#         score = self.upscore(score)
#         return score
