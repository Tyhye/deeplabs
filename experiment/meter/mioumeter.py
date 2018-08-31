#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-31 09:18:31 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-31 09:18:31 
'''

import numpy as np
import math
import torch


class mIOUMeter(object):

    def __init__(self):
        self.mious = []
        self.reset()

    def reset(self):
        '''Resets the meter to default settings.'''
        self.mious.clear()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        output = np.argmax(output, axis=1)
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        assert output.shape[0] == target.shape[0], "pred and target do not match"
        classes_num = output.shape[1]
        ious = []
        for idx in range(classes_num):
            TP = np.sum((output==idx)[target==idx])
            AT = np.sum(output==idx) + np.sum(target==idx) - TP
            ious.append(iou)
        ious = np.mean(np.stack(ious, axis=1), axis=1)
        self.mious.extend(list(ious))

    def value(self):
        '''Get the value of the meter in the current state.'''
        return sum(self.mious) / len(self.mious)
