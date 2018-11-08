# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

from models import pose_resnet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class PoseResNetSam(pose_resnet.PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        super().__init__(block, layers, cfg, **kwargs)

        self.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        self.JOINTS = cfg.MODEL.NUM_JOINTS
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        W,H = self.heatmap_size
        hm_size = int(W*H)
        self.sm = torch.nn.Softmax(dim=-1)
        self.indices = torch.arange(hm_size).double().unsqueeze(0).unsqueeze(0).expand((self.BATCH_SIZE,self.JOINTS, hm_size))

    def _sam(self, x):
        weights = self.sm(x.view(self.BATCH_SIZE, self.JOINTS,-1).double())
        sam = weights * self.indices
        res = sam.sum(dim=-1).int().unsqueeze(-1)
        indices_x = res / int(self.heatmap_size[1])
        indices_y = res % int(self.heatmap_size[0])
        return torch.cat((indices_x,indices_y),2)

    def forward(self, x):
        # x = super().forward(x)
        x = self._sam(x)

        return x


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = pose_resnet.resnet_spec[num_layers]

    model = PoseResNetSam(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
