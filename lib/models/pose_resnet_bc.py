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

class PoseResNetBC(pose_resnet.PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        super().__init__(block, layers, cfg, **kwargs)

        extra = cfg.MODEL.EXTRA
        
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.fields_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS*2,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )        

        self.bc_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=1,
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

        x = self.deconv_layers(x)

        joints = self.final_layer(x)
        fields = self.fields_layer(x)
        bc = self.bc_layer(x)

        return joints, fields, bc


    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.fields_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=>fields init .weight as normal(0, 0.001)')
                    logger.info('=>fields init .bias as 0')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            for m in self.bc_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=>bc init .weight as normal(0, 0.001)')
                    logger.info('=>bc init .bias as 0')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=>final init .weight as normal(0, 0.001)')
                    logger.info('=>final init .bias as 0')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = pose_resnet.resnet_spec[num_layers]

    model = PoseResNetBC(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
