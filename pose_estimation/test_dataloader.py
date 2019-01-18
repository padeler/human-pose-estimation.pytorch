
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import cv2 
import numpy as np

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
from core.function import validate
from utils.utils import create_logger

import dataset


from valid import parse_args, reset_config

# visualize
colors = [[255,255,255], 
          [255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0],
          [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0],
          [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180],
          [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
          [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255],]


def viz_joints(canvas, joints, txt_size=0.5):
    for i, p in enumerate(joints):
        x,y,v = int(p[0]), int(p[1]), int(p[2])
        if v==1:    
            cv2.circle(canvas, (x,y), 4, colors[i%len(colors)], thickness=1)
            if txt_size>0:
                cv2.putText(canvas, str(i), (x+5,y), 0, txt_size, colors[i%len(colors)])

    return canvas

def run():
    
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger( config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))


    # Data loading code
    valid_dataset = dataset.coco_sam(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET, True, None)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )


    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    print("Validation demo loop.")

    for i, (input, target, target_weight, meta) in enumerate(valid_loader):
        logger.info("%d frame. Meta: %s", i, meta)
        
        img = np.squeeze(input.cpu().numpy())
        logger.info("Img Shape %s",img.shape)

        # hm = np.squeeze(target.cpu().numpy()).transpose(1,2,0)
        # hm_sum = np.sum(hm,axis=-1)
        # hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        # logger.info("Nose min/max %f %f", hm[...,0].min(),hm[...,0].max())
        # cv2.imshow("Hm sum",hm_res)

        logger.info("Target shape: %s",target.shape)
        logger.info("Target shape: %s",target)
        tw = target_weight.cpu().numpy()
        logger.info("Target Weights shape: %s",tw.shape)
        
        joints = np.hstack((target.cpu().numpy().squeeze()*4.0,tw[0]))
        print("Joints: \n",joints)
        img = viz_joints(img, joints)
        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused




if __name__ == "__main__":
    run()
