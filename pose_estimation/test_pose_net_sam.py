# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

import numpy as np
import cv2

def main():

    update_config("experiments/coco/resnet50/256x192_d256x3_sam_adam_lr1e-3.yaml")

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose_resnet_sam.get_pose_net( config, is_train=True )

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))

    # dump_out = model(dump_input)
    # print("Model out shape: ", dump_out.shape)

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    # ).cuda()

    # Data loading code
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # trans = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])

    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False, 
        None # trans
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0

    for i, (input, target, target_weight, meta) in enumerate(valid_loader):
        print("%d frame. Target Shape %s" % (i, target.shape), input.shape)
        
        
        print()
        img = input.cpu().numpy()[0,...]

        hm = target.cpu().numpy()[0,...].transpose(1,2,0)
        hm_sum = np.sum(hm,axis=-1)
        hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        print("Nose min/max %f %f"%(hm[...,0].min(),hm[...,0].max()))
        cv2.imshow("Hm sum",hm_res)

        # give heatmaps to model
        hm_in = torch.Tensor(target)
        print("hm_in shape",hm_in.shape)
        model_out = model(hm_in)

        out = model_out.numpy()
        print("Out shape: ",out.shape)
        print("Out[0,...]\n", out[0])

        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused






if __name__ == '__main__':
    main()
