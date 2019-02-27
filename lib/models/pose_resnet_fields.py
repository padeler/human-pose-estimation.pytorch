# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn

from models import pose_resnet

logger = logging.getLogger(__name__)

class PoseResNetFields(pose_resnet.PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        super().__init__(block, layers, cfg, **kwargs)

        extra = cfg.MODEL.EXTRA

        self.fields_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS*2,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        last_deconv = self.deconv_layers(x)
        joints = self.final_layer(last_deconv)
        fields = self.fields_layer(last_deconv)

        return joints, fields


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = pose_resnet.resnet_spec[num_layers]

    model = PoseResNetFields(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
