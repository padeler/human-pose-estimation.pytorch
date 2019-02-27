
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

from utils.vis import draw_coords, coco_part_str

import dataset
import models
import matplotlib

import numpy as np
import cv2

from torch.nn import MSELoss




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

    update_config("experiments/coco_fields/resnet50/256x256_d256x3_adam_lr1e-3.yaml")

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED


    model = models.pose_resnet_fields.get_pose_net( config, is_train=True)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))

    joints, fields = model(dump_input)
    print("Model out shape (joints, fields): ", joints.shape, fields.shape)

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    # ).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


    from dataset import coco_fields

    valid_dataset = coco_fields(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False, 
        None
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=config.WORKERS,
        pin_memory=True
    )

    criterion = MSELoss(size_average=True)

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    cv2.namedWindow("foo")

    for i, (input, target, target_fields, target_weight, target_weight_fields, meta) in enumerate(valid_loader):
        print("%d frame. Target Shape %s" % (i, target.shape), input.shape, meta["image"])
        

        img = input.cpu().numpy()[0,...]

        # hm = target.cpu().numpy()[0,...].transpose(1,2,0)
        # hm_sum = np.sum(hm,axis=-1)
        # hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        # print("Nose min/max %f %f"%(hm[...,0].min(),hm[...,0].max()))
        # cv2.imshow("Hm sum",hm_res)

        # weight_tensor = target_weight[...,:1]

        # give heatmaps to model
        # hm_in = torch.Tensor(target)
        # print("hm_in shape",hm_in.shape)
                    
        joints, fields = model(transform(img).view(1, 3, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))        

        # gt = target.cpu().numpy()
        # pred = joints.detach().cpu().numpy()
        # print("Pred:",pred.shape,"\n","GT\n",gt.shape)
        # y = gt
        # w = target_weight[...,:2].cpu().numpy()

        # nploss = np.mean(((pred-y)*w)**2) / 2.0
        # print("NP loss: ",nploss)
        loss = calc_loss(criterion, joints.cpu(), target.cpu(), target_weight[...,:2])
        print("BATCH LOSS: ",loss)

        # visible = target_weight.cpu().numpy()        
        # gt_vis = np.hstack((gt[0]*4.,visible[0]))
        # pred_vis = np.hstack((pred[0]*4., visible[0]))
        
        # img = draw_coords(np.copy(img),pred_vis,gt_vis,coco_part_str[:-1])
        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused






if __name__ == '__main__':
    main()
