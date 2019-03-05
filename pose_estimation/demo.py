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
from utils import vis

import dataset
import models

import time
import numpy as np
import cv2 
import glob

import cv2
from valid import parse_args, reset_config

from utils.skeleton_tools import fields2skeletons, visualize_skeletons


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [
              0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def validate_demo(config, model, preproc):
    # switch to evaluate mode
    model.eval()

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    print("Validation demo loop.")
    with torch.no_grad():

        dataset_path = "/media/storage/home/padeler/work/datasets/coco_2017/val2017"
        val_set = glob.glob(dataset_path+os.sep+"*.jpg")
        val_set.sort()
        for fname in val_set:
            if k&0xFF ==ord('q'):
                break
            image_id = int(os.path.os.path.basename(fname)[:-4])
            frame = cv2.imread(fname)

        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.0)
        # idx = 0 
        # while k&0xFF != ord('q'):
        #     ret, frame = cap.read()
        #     if not ret:
        #         raise Exception("VideoCapture.read() returned False")
            
            # boxsize = list(config.MODEL.IMAGE_SIZE[::-1]) # height,width
            boxsize = 512
            im = np.zeros([boxsize,boxsize,3,],dtype=np.uint8)

            sc = boxsize/np.max(frame.shape[:2])
            frame_sc = cv2.resize(frame,(0,0),fx=sc,fy=sc)
            h,w = frame_sc.shape[:2]
            h = min(h,boxsize)
            w = min(w,boxsize)
            # print("IM ",im.shape," to ",frame_sc.shape)
            im[:h,:w] = frame_sc[:h,:w]

            before = time.time()
            
            # compute output
            tensor = preproc(im)
            batch = tensor.reshape([1,]+list(tensor.shape))
            pred = model(batch)
            pred = np.squeeze(pred.cpu().numpy())

            hm = pred[:17,...]
            fields = pred[17:-1,...]
            bc = pred[-1,...]

            mask = np.zeros_like(bc)

            dt = time.time()-before
            print("Time to result ",dt,"FPS",(1./dt))

            heatmaps = np.squeeze(hm).transpose(1,2,0)
            print("Heatmaps, bc ===>",heatmaps.shape,bc.shape)

            fields = np.squeeze(fields).transpose(1,2,0)

            joints, centers, joints_bc, skeletons = fields2skeletons(heatmaps, fields, bc)

            for i,jg in enumerate(joints):
                if jg is not None:
                    for j, cand in enumerate(jg):
                        x, y, score, _ = cand
                        j_bc = joints_bc[i][j]
                        pos_x = int(j_bc[0])
                        pos_y = int(j_bc[1])
                        if pos_x>=0 and pos_y>=0 and pos_x<bc.shape[1] and pos_y<bc.shape[0]:
                            mask[pos_y, pos_x] += 1.0


            if centers is not None:
                for p,j in enumerate(centers):
                    print(p,",",j)
                    x, y, score, _ = j
                    c = (int(x*4.0),int(y*4.0))
                    cv2.circle(im, c, 3, [100,50,255], -1)
                    cv2.putText(im, str(p)+" (%0.2f)"%score, c, 0, 0.3, [200,100,50],1)

            im = visualize_skeletons(im, skeletons)

            cv2.imshow("Source", im)

            sel_hm = heatmaps[...,idx]
 
            print("HM ",idx, "min, max, mean, median:", np.min(sel_hm),np.max(sel_hm),np.mean(sel_hm),np.median(sel_hm))
            cv2.imshow("Heatmap idx",cv2.normalize(sel_hm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
            
            sum_hm = np.sum(heatmaps,axis=-1)
            cv2.imshow("Heatmap sum",cv2.normalize(sum_hm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

            bc_res = cv2.resize(bc,(0,0),fx=4.0,fy=4.0)
            cv2.imshow("Barycenters",cv2.normalize(bc_res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

            mask_res = cv2.resize(mask,(0,0),fx=4.0,fy=4.0)
            cv2.imshow("Instance Mask",cv2.normalize(mask_res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

            k = cv2.waitKey(delay[paused])
            if k&0xFF==ord('q'):
                break
            if k&0xFF==ord('p'):
                paused = not paused
            if k&0xFF==ord('n'):
                idx += 1
                idx = idx%heatmaps.shape[2]
                print("Showing heatmap idx",idx)

            if k&0xFF==ord('s'):
                data = {}
                print("Saving frame data")
                np.savez("data.npz", image=im, fields=fields, heatmaps=heatmaps, bc=bc)



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

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')( config, is_train=False )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['state_dict'])
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
