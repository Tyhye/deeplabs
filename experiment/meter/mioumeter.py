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

    def __init__(self, classes_num):
        self.classes_num = classes_num
        self.reset()

    def reset(self):
        '''Resets the meter to default settings.'''
        self.tps = [[]]*self.classes_num
        self.ats = [[]]*self.classes_num
        
    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        classes_num = output.shape[1]
        output = np.argmax(output, axis=1)
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        assert output.shape == target.shape, "pred and target do not match %s %s"%(output.shape, target.shape)
        for idx in range(classes_num):
            TP = np.sum((output==idx)[target==idx])
            AT = np.sum(output==idx) + np.sum(target==idx) - TP
            self.tps[idx].append(TP)
            self.ats[idx].append(AT)
        
    def value(self):
        '''Get the value of the meter in the current state.'''
        ious = []
        for idx in range(self.classes_num):
            TP = sum(self.tps[idx])
            AT = sum(self.ats[idx])
            if AT != 0:
                ious.append(TP/AT)
            return sum(ious) / len(ious)
