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

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import AverageMeter
from utils.utils import create_logger

import dataset
import models

import time
import numpy as np
import cv2 
import glob

import cv2
from valid import parse_args, reset_config

def validate_demo(config, model, preproc):
    # switch to evaluate mode
    model.eval()

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    print("Validation demo loop.")
    with torch.no_grad():

        # dataset_path = "/media/storage/home/padeler/work/datasets/coco_2017/val2017"
        # val_set = glob.glob(dataset_path+os.sep+"*.jpg")
        # val_set.sort()
        # for fname in val_set:
        #     if k&0xFF ==ord('q'):
        #         break
        #     image_id = int(os.path.os.path.basename(fname)[:-4])
        #     frame = cv2.imread(fname)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)
        while k&0xFF != ord('q'):
            ret, frame = cap.read()
            if not ret:
                raise Exception("VideoCapture.read() returned False")
            

            boxsize = 512
            im = np.zeros((boxsize,boxsize,3),dtype=np.uint8)
            sc = boxsize/max(frame.shape[0:2])
            frame_sc = cv2.resize(frame,(0,0),fx=sc,fy=sc)
            h,w = frame_sc.shape[:2]
            im[:h,:w] = frame_sc

            before = time.time()
            
            # compute output
            tensor = preproc(im)
            batch = tensor.reshape([1,]+list(tensor.shape))
            output = model(batch)
            result = output.cpu().numpy()

            dt = time.time()-before
            print("Time to result ",dt,"FPS",(1./dt))

            heatmaps = np.squeeze(result).transpose(1,2,0)

            print("Heatmaps ===>",heatmaps.shape,heatmaps.dtype)
            hm = np.sum(heatmaps, axis=-1)
            cv2.imshow("Source", im)
            cv2.imshow("Heatmaps",hm)


            k = cv2.waitKey(delay[paused])
            if k&0xFF==ord('q'):
                break
            if k&0xFF==ord('p'):
                paused = not paused


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger( config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    model = models.pose_resnet.get_pose_net(config, is_train=False)
    # model = eval('models.'+config.MODEL.NAME+'.get_pose_net')( config, is_train=False )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))



    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    preproc = transforms.Compose([ transforms.ToTensor(), normalize, ])

    # evaluate on validation set
    validate_demo(config, model, preproc)


if __name__ == '__main__':
    main()
