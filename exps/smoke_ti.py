#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (320, 320)
        self.test_size = (320, 320)
        self.random_size = (10, 15)

        self.basic_lr_per_img = 0.01 / (64*2)

        self.mosaic_scale = (0.8, 1.1)
        self.mosaic_prob = 0.0
        self.flip_prob = 0.5
        self.translate = 0.3
        self.shear = 2.0
        self.mixup_scale = (1.0, 1.1)
        self.enable_mixup = False
        self.no_aug_epochs = 50

        self.eval_interval = 1
        self.act = "relu"
        self.data_dir = os.path.join(os.getcwd(),'datasets') # "/home/jaemu/Desktop/vaping_detection/datasets"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.test_ann = "test.json"
        self.num_classes = 2
        self.train_img_dir = "train"
        self.val_img_dir = "valid"
        self.test_img_dir = "test"
        self.max_epoch = 700
        self.save_train_img = False
        self.ema = True
        self.data_num_workers = 4
        self.grayscale = True

        self.save_history_ckpt = False

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.

            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=False, grayscale=self.grayscale
            )

            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=False
            )

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

if __name__ == "__main__":

    import torch

    model = Exp().get_model()

    random_input = torch.zeros([1,3,320,320])
    model.eval()
    output = model(random_input)
    print(output)