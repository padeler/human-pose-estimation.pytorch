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

logger = logging.getLogger(__name__)

class SAM(nn.Module):

    def __init__(self, beta):
        super(SAM, self).__init__()
        self.beta = beta        
        self.sm = torch.nn.Softmax(dim=-1)
        

    def forward(self, x):
        bs, joints, H, W = [int(v) for v in x.shape]

        iy = torch.arange(start=0,end=H,step=1).float().view((1,1,H,1))
        ix = torch.arange(start=0,end=W,step=1).float().view((1,1,1,W))
        if x.is_cuda:
            ix = ix.cuda()
            iy = iy.cuda()

        indices_y = iy.expand((bs, joints, H, W))
        indices_x = ix.expand((bs, joints, H, W))

        weights = self.sm(self.beta*x.view(bs, joints,-1)).view((bs, joints, H, W))
        sam_x = (weights * indices_x).view((bs, joints, -1))
        sam_y = (weights * indices_y).view((bs, joints, -1))
        pred_x = sam_x.sum(dim=-1).unsqueeze(-1)
        pred_y = sam_y.sum(dim=-1).unsqueeze(-1)

        return torch.cat((pred_x,pred_y),2)


class PoseResNetSam(pose_resnet.PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        super().__init__(block, layers, cfg, **kwargs)
        W,H  = [int(x) for x in cfg.MODEL.EXTRA.HEATMAP_SIZE]
        self._sam = SAM(float(cfg.MODEL.EXTRA.SAM_BETA))
        

    def forward(self, x):
        hm = super().forward(x)
        coords = self._sam(hm)

        return coords, hm


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = pose_resnet.resnet_spec[num_layers]

    model = PoseResNetSam(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
