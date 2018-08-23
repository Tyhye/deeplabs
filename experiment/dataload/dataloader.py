#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-07-23 01:30:31 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-07-23 01:30:31 
'''
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SS_train_val_data_loader(object):
    '''
    resize_size = (h, w)
    '''

    def __init__(self, resize_size=(512, 512),
                 data_root=None, data_extension=".jpg",
                 label_root=None, label_extension=".png"):
        self.data_root = data_root
        self.label_root = label_root
        self.data_extension = data_extension
        self.label_extension = label_extension
        image_transform_list = [
            transforms.Resize(resize_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        label_transform_list = [
            transforms.Resize(resize_size, interpolation=Image.NEAREST)
        ]
        self.image_transformer = transforms.Compose(image_transform_list)
        self.label_transformer = transforms.Compose(label_transform_list)

    def __call__(self, filename):
        if self.data_root is not None:
            datapath = os.path.join(self.data_root, filename.strip())
        else:
            datapath = filename
        if self.label_root is not None:
            labelpath = os.path.join(self.label_root, filename.strip())
        else:
            labelpath = filename
        datapath = datapath + self.data_extension
        labelpath = labelpath + self.label_extension
        img = Image.open(datapath).convert('RGB')
        img = self.image_transformer(img)
        label = Image.open(labelpath)
        label = self.label_transformer(label)
        label = torch.LongTensor(np.asarray(label, dtype=int))
        return img, label


class SS_eval_data_loader(object):
    '''
    resize_size = (h, w)
    '''

    def __init__(self, resize_size=(512, 512),
                 data_root=None, data_extension=".jpg"):
        self.data_root = data_root
        self.data_extension = data_extension
        image_transform_list = [
            transforms.Resize(resize_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        self.image_transformer = transforms.Compose(image_transform_list)

    def __call__(self, filename):
        if self.data_root is not None:
            datapath = os.path.join(self.data_root, filename.strip())
        else:
            datapath = filename
        datapath = datapath + self.data_extension
        img = Image.open(datapath).convert('RGB')
        img = self.image_transformer(img)
        return img, filename
