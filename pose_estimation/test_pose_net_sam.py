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
import matplotlib

import numpy as np
import cv2

from torch.nn import MSELoss

# COCO parts
coco_part_str = [u'nose', u'leye', u'reye', u'lear', u'rear', u'lsho', 
                 u'rsho', u'lelb', u'relb', u'lwr', u'rwr', u'lhip', u'rhip', 
                 u'lknee', u'rknee', u'lankle', u'rankle', u'bg']

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def draw_peaks(canvas, pred, gt, part_str):
    # pred = np.round(pred,)
    for i in range(len(part_str)):
        x, y, v = pred[i]
        if v>0.5:
            cv2.circle(canvas, (int(x),int(y)), 4, colors[i], thickness=-1)
            cv2.putText(canvas, part_str[i], (int(x),int(y)), 0, 0.5, colors[i])

        gx, gy, gv = gt[i]
        print(i,v,"Pred ",x,y,"gt",gx,gy,gv)
        if v>0.5:
            cv2.circle(canvas, (int(gx),int(gy)), 2, [0,0,0], thickness=-1)

    return canvas



def calc_loss(criterion, output, target, target_weight):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
    loss = 0

    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx].squeeze()
        heatmap_gt = heatmaps_gt[idx].squeeze()

        loss += 0.5 * criterion(
            heatmap_pred.mul(target_weight[:, idx]),
            heatmap_gt.mul(target_weight[:, idx])
        )

    return loss / num_joints



def main():

    update_config("experiments/coco/resnet50/256x192_d256x3_sam_adam_test.yaml")

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    from dataset import coco,coco_sam 

    valid_dataset = coco_sam(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False, 
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        # num_workers=config.WORKERS,
        pin_memory=True
    )

    # train_dataset = coco(
    #     config,
    #     config.DATASET.ROOT,
    #     config.DATASET.TRAIN_SET,
    #     True,
    #     None
    # )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config.TRAIN.BATCH_SIZE,
    #     shuffle=config.TRAIN.SHUFFLE,
    #     # num_workers=config.WORKERS,
    #     pin_memory=True
    # )

    criterion = MSELoss(size_average=True)

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    cv2.namedWindow("foo")

    for i, (input, target, target_weight, meta) in enumerate(valid_loader):
        print("%d frame. Target Shape %s" % (i, target.shape), input.shape)
                
        img = input.cpu().numpy()[0,...]

        # hm = target.cpu().numpy()[0,...].transpose(1,2,0)
        # hm_sum = np.sum(hm,axis=-1)
        # hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        # print("Nose min/max %f %f"%(hm[...,0].min(),hm[...,0].max()))
        # cv2.imshow("Hm sum",hm_res)

        weight_tensor = target_weight[...,:1]

        # give heatmaps to model
        # hm_in = torch.Tensor(target)
        # print("hm_in shape",hm_in.shape)
        model_out = model(input)
        

        gt = meta['joints'][...,:2].float()
        gt_sc = gt# / 4.0

        pred = model_out.detach().numpy()
        print("Pred:",pred.shape,"\n",pred[0])
        y = gt_sc.numpy()
        w = target_weight[...,:2].numpy()

        nploss = np.mean(((pred-y)*w)**2) / 2.0
        print("NP loss: ",nploss)
        loss = calc_loss(criterion, model_out, gt_sc, weight_tensor)
        print("BATCH LOSS: ",loss)

        # visible = target_weight.numpy()
        # out = model_out.numpy() #* 4.0
        # out = np.dstack((out,visible))
        # print("Out shape: ",out.shape)
        # print("Out[0,...]\n", out[0])
        
        
        # gt_vis = np.dstack((gt.numpy(),visible))
        # img = draw_peaks(np.copy(img),out[0],gt_vis[0],coco_part_str[:-1])
        # cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused






if __name__ == '__main__':
    main()
