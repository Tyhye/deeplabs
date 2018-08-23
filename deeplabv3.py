#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-07-23 02:44:23 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-07-23 02:44:23 
'''

import os
import logging
from docopt import docopt
from easydict import EasyDict as edict

from experiment.deeplabv3_process import train_deeplabv3


docstr = """ DeepLab V3 Train Script. Could train deeplabv3 use this script and 
setting configures.

Usage: 
    deeplabv3.py [options]

Options:
    -h, --help                  Print this message
    --logfile=<str>             File path for saving log message. 
    --device=<str>              Device for runnint the model [default: cuda:0]
    --withvisdom                If show the result with visdom

    --basenet=<str>             BaseNet type for the Model [default: resnet18]
    --classes_num=<int>         Output classes number of the network [default: 21]
    --feature_channels=<int>    Feature channels of the network [default: 256]
    --multi_grid_rates=<list>   Multi Grid Rates for the final block [default: [1,2]]
    --output_stride=<int>       Output stride of the network [default: 8]
    
    --resize_size=<tuple>       Image resize size tuple (height, width) [default: (288, 512)]
    --scale_range=<tuple>       Image scale range fot training [default: (0.7, 1.5)]
    --result_mode=<str>         Result upsample mode [default: bilinear]
    --loss_mode=<str>           Loss mode when training [default: score2label]
    
    --learning_rate=<float>     Learning Rate [default: 0.007]
    --weight_decay=<float>      Weight Decay [default: 0.9997]
    --optim=<str>               Optimizer Type [default: SGD]
    --LRpolicy=<str>            Learning rate policy [default: poly]
    --poly_power=<float>        Power for the poly policy [default: 0.9]
    --batch_size=<int>          Batchsize [default: 8]
    
    --max_iters=<int>           Max Train iters [default: 10000]
    --log_iters=<int>           Log iter stone [default: 10]
    --val_iters=<int>           Val iter stone [default: 1000]
    --snap_iters=<int>          Snap iter stone [default: 1000]

    --train_list=<str>          Train files list txt
                                [default: datas/VOC2012/ImageSets/Segmentation/train.txt]
    --train_data_root=<str>     Train sketch images path prefix
                                [default: datas/VOC2012/JPEGImages/]
    --train_label_root=<str>    Train label map path prefix 
                                [default: datas/VOC2012/SegmentationClass/]
    --val_list=<str>            Val files list txt 
                                [default: datas/VOC2012/ImageSets/Segmentation/val.txt]
    --val_data_root=<str>       Val sketch images path prefix 
                                [default: datas/VOC2012/JPEGImages/]
    --val_label_root=<str>      Val label map path prefix
                                [default: datas/VOC2012/SegmentationClass/]
    --data_extension=<str>      Data image extension. [default: .jpg]
    --label_extension=<str>     Label image extension [default: .png]

    --snap_dir=<str>            Model state dict file path [default: saved/]
    --base_not_pretrained       If the net work using Resnet do not pretrained on ImageNet.
    --pretrain_path=<str>       Path to pretrained model.
    --freeze_bn                 if the batchnorm are freezed

"""

def main():
    args = docopt(docstr, version='v0.1')

    # -------set logger --------------------------------------------------------
    log_level = logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level=log_level)
    formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(levelname)s\t-%(message)s')
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)
    consolehandler.setFormatter(formatter)
    logger.addHandler(consolehandler)
    if args['--logfile'] is not None:
        filehandler = logging.FileHandler(args['--logfile'], mode='w')
        filehandler.setLevel(log_level)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logprint = logger.info
    logprint(args)

    # ----------------- process configure --------------------------------------
    cfg = edict()
    cfg.device = args["--device"]
    cfg.withvisdom = args["--withvisdom"]
    cfg.loss_mode = args["--loss_mode"]

    # path config
    cfg.train_list = args["--train_list"]
    cfg.val_list = args["--val_list"]
    cfg.train_data_root = args["--train_data_root"]
    cfg.val_data_root = args["--val_data_root"]
    cfg.train_label_root = args["--train_label_root"]
    cfg.val_label_root = args["--val_label_root"]
    cfg.data_extension = args["--data_extension"]
    cfg.label_extension = args["--label_extension"]

    # solver config
    cfg.batch_size = int(args["--batch_size"])
    cfg.learning_rate = float(args["--learning_rate"])
    cfg.weight_decay = float(args["--weight_decay"])
    cfg.optim = args["--optim"]
    cfg.LRpolice = args["--LRpolicy"]
    cfg.poly_power = float(args["--poly_power"])
    
    # data loader config 
    cfg.resize_size = eval(args["--resize_size"])
    cfg.scale_range = eval(args["--scale_range"])

    # network config
    cfg.basenet = args['--basenet']
    cfg.classes_num = int(args['--classes_num'])
    cfg.feature_channels = int(args['--feature_channels'])
    cfg.mg = eval(args['--multi_grid_rates'])
    cfg.output_stride = int(args['--output_stride'])
    cfg.pretrain_path = args['--pretrain_path']
    cfg.pretrained = (cfg.pretrain_path is None) and (not args['--base_not_pretrained'])
    cfg.freeze_bn = args['--freeze_bn']
    
    # iters config and snap dir
    cfg.max_iters = int(args["--max_iters"])
    cfg.log_iters = int(args["--log_iters"])
    cfg.val_iters = int(args["--val_iters"])
    cfg.snap_iters = int(args["--snap_iters"])
    cfg.snap_dir = args["--snap_dir"]
    if not os.path.exists(cfg.snap_dir):
        os.makedirs(cfg.snap_dir)
    
    train_deeplabv3(cfg, logprint=logprint)

if __name__ == "__main__":
    main()
