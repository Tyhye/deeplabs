#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 01:32:37 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 01:32:37 
'''

from __future__ import division

from tqdm import tqdm
import numpy as np
import random

import torch
import torchnet as tnt
import torch.nn as nn
import torch.nn.functional as F

from .model.model import DeepLabV3Model
from .model.resnetdeeplabv3 import resnet18, resnet34, resnet50
from .model.resnetdeeplabv3 import resnet101, resnet152
from .process.iterprocessor import IterProcessor
from .dataload.dataloader import SS_train_val_data_loader, SS_eval_data_loader
from .dataload.datafunc import resize_tensor
from .meter.mioumeter import mIOUMeter


def get_params(Net, cfg):
    params = [{'params': Net.parameters(), 'lr': cfg.learning_rate}]
    return params


def train_deeplabv3(cfg, logprint=print):

    # ==========================================================================
    # define model and trainer list, lr_scheduler
    # ==========================================================================
    Net = DeepLabV3Model(classes_num=cfg.classes_num,
                         feature_channels=cfg.feature_channels,
                         basenet=eval(cfg.basenet),
                         pretrained=cfg.pretrained,
                         mg=cfg.mg,
                         output_stride=cfg.output_stride)
    logprint(Net)
    if cfg.pretrain_path is not None:
        cfg.Net.load_state_dict(torch.load(
            cfg.pretrain_path, map_location="cpu"), strict=True)
    Net = Net.cuda(cfg.device)
    # ==========================================================================
    # define optimizers and lr_scheduler
    # ==========================================================================
    params = get_params(Net, cfg)
    from torch.optim import SGD, Adam
    from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
    if cfg.optim == 'Adam':
        optimizer = Adam(params, lr=cfg.learning_rate, betas=(
            0.5, 0.999), eps=1e-12, weight_decay=cfg.weight_decay)
    if cfg.optim == 'SGD':
        optimizer = SGD(params, momentum=0.9,
                        weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.LRpolice == "poly":
        def lambda1(t): return (1 - float(t)/cfg.max_iters)**cfg.poly_power
        lr_scheduler = LambdaLR(optimizer, lambda1)
    else:
        raise "Unsupport Learning police"
    # ==========================================================================
    # ==========================================================================
    # metric, loss and other define
    # ==========================================================================
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_miou = mIOUMeter(cfg.classes_num)

    save_name = "deeplabv3_%s_%d" % (cfg.basenet, cfg.output_stride)
    # ==========================================================================
    # ==========================================================================
    # visdom logger
    # ==========================================================================
    if cfg.withvisdom:
        from visdom import Visdom
        from torchnet.logger import VisdomPlotLogger
        visdom = Visdom()
        port = 8097
        train_loss_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': 'Train Loss'})
        train_acc_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': 'Train Accuracy'})
        test_loss_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': 'Test Loss'})
        test_acc_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': 'Test Accuracy'})
        train_lr_logger = VisdomPlotLogger(
            'line', port=port, opts={'title': 'Learning Rate'})
    # ==========================================================================
    # define dataset and iterators
    # ==========================================================================

    def get_iterator(mode):
        if mode:
            train_load = SS_train_val_data_loader(resize_size=cfg.resize_size,
                                                  data_root=cfg.train_data_root,
                                                  data_extension=cfg.data_extension,
                                                  label_root=cfg.train_label_root,
                                                  label_extension=cfg.label_extension)
            tensor_dataset = tnt.dataset.ListDataset(cfg.train_list,
                                                     load=train_load)
        else:
            val_load = SS_train_val_data_loader(resize_size=cfg.resize_size,
                                                data_root=cfg.val_data_root,
                                                data_extension=cfg.data_extension,
                                                label_root=cfg.val_label_root,
                                                label_extension=cfg.label_extension)
            tensor_dataset = tnt.dataset.ListDataset(cfg.val_list,
                                                     load=val_load)
        return tensor_dataset.parallel(batch_size=cfg.batch_size, num_workers=1,
                                       shuffle=mode, drop_last=mode)
    # ==========================================================================
    # ==========================================================================
    # reset_metric function
    # ==========================================================================

    def reset_meters():
        meter_loss.reset()
        meter_accuracy.reset()
        meter_miou.reset()

    # ==========================================================================

    # ==========================================================================
    # process functions
    # ==========================================================================
    def on_start(state):
        pass

    def on_start_iter(state):
        lr_scheduler.step()

    def on_sample(state):
        pass

    def train_process(sample):
        img, label = sample
        scalerate = random.uniform(*cfg.scale_range)
        _, _, h, w = img.size()
        img = resize_tensor(img, resizew=int(scalerate*w), resizeh=int(scalerate*h),
                            mode="bilinear")
        img = img.cuda(cfg.device)
        score = Net(img)
        # ======================================================================
        if cfg.loss_mode == "label2score":
            # mode label2score: label resize to score
            _, _, sh, sw = score.size()
            label = resize_tensor(
                label, resizew=sw, resizeh=sh, mode="nearest")
        elif cfg.loss_mode == "score2label":
            # mode score2label: score resize to label
            score = F.upsample(score, size=label.size()[1:],
                               mode="bilinear", align_corners=True)
        else:
            raise "Wrong loss mode!"
        # ======================================================================
        sample[1] = label
        label = label.cuda(cfg.device)
        loss = criterion(score, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, score

    def test_process(sample):
        img, label = sample
        img = img.cuda(cfg.device)
        score = Net(img)
        if cfg.loss_mode == "label2score":
            _, _, sh, sw = score.size()
            label = resize_tensor(
                label, resizew=sw, resizeh=sh, mode="nearest")
        elif cfg.loss_mode == "score2label":
            score = F.upsample(score, size=label.size()[1:],
                               mode="bilinear", align_corners=True)
        else:
            raise "Wrong loss mode!"
        sample[1] = label
        label = label.cuda(cfg.device)
        loss = criterion(score, label)
        return loss, score

    def on_forward(state):
        score = state['output'].detach()
        label = state['sample'][1]
        meter_miou.add(score, label)
        score = score.permute(0, 2, 3, 1).contiguous()
        score = score.view(-1, score.size(3))
        label = label.view(-1)
        meter_accuracy.add(score, label)
        meter_loss.add(state['loss'].data)

    def on_end_iter(state):
        if not state['train']:
            return
        if (state['t']) % cfg.log_iters == 0:
            train_miou = meter_miou.value()
            train_loss = meter_loss.value()[0]
            train_accuracy = meter_accuracy.value()[0]
            learning_rate = optimizer.param_groups[-1]['lr']
            logprint('[iter %d] Training Loss: %.4f LR: %f Accuracy: %.2f%% mIOU: %.2f%%' %
                     (state['t'], train_loss, learning_rate, train_accuracy, train_miou*100))
            if cfg.withvisdom:
                train_loss_logger.log(state['t'], train_loss)
                train_acc_logger.log(state['t'], train_accuracy)
                train_lr_logger.log(state['t'], learning_rate)
                visdom.save(["main"])
            reset_meters()

        if (state['t']) % cfg.snap_iters == 0:
            torch.save(Net.state_dict(), '%s/%s_iter_%d.pt' %
                       (cfg.snap_dir, save_name, state['t']))

        if state['t'] % cfg.val_iters == 0:
            Net.eval()
            processor.test(test_process, get_iterator(False))
            val_miou = meter_miou.value()
            val_loss = meter_loss.value()[0]
            val_accuracy = meter_accuracy.value()[0]
            logprint('[iter %d] Val Loss: %.4f Accuracy: %.2f%% mIOU: %.2f%%' %
                     (state['t'], val_loss, val_accuracy, val_miou*100))
            Net.train(freeze_bn_able=cfg.freeze_bn)

    def on_end(state):
        pass

    processor = IterProcessor()
    processor.hooks['on_start'] = on_start
    processor.hooks['on_start_iter'] = on_start_iter
    processor.hooks['on_sample'] = on_sample
    processor.hooks['on_forward'] = on_forward
    processor.hooks['on_end_iter'] = on_end_iter
    processor.hooks['on_end'] = on_end

    Net.train(freeze_bn_able=cfg.freeze_bn)
    processor.train(train_process, get_iterator(True), cfg.max_iters)
