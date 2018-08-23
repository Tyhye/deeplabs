#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-01 11:23:02 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-01 11:23:02 
'''
import cv2
import torch
import numpy as np
from PIL import Image
from .mypalette import Palette

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST,
                  "bilinear": cv2.INTER_LINEAR,
                  "area": cv2.INTER_AREA,
                  "cubic": cv2.INTER_CUBIC}

def image_tensor_unnormalize(tensor, mean=MEAN, std=STD):
    if tensor.dim() == 4:
        for ten in tensor:
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
    elif tensor.dim() == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    return tensor

def resize_tensor(tensor, resizew, resizeh, mode="nearest"):
    if tensor.dim() == 4:
        b, c, h, w = tensor.size()
        tmparray = np.zeros((b, c, resizeh, resizew))
        tensor_array = tensor.numpy().transpose(0, 2, 3, 1)
    elif tensor.dim() == 3:
        b, h, w = tensor.size()
        tmparray = np.zeros((b, resizeh, resizew))
        tensor_array = tensor.numpy()
    for idx, src in enumerate(tensor_array):
        tmp = cv2.resize(src, (resizew, resizeh),
                         interpolation=INTERPOLATIONS[mode])
        if tensor.dim() == 4:
            tmp = tmp.transpose(2,0,1)
        tmparray[idx, ...] = tmp
    return torch.from_numpy(tmparray).type_as(tensor)

def np2labelimg(arr):
    img = Image.fromarray(arr)
    img.putpalette(Palette)
    return img
