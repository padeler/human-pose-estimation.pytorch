
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
        config.DATASET.TEST_SET,
        False, None)

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

        hm = np.squeeze(target.cpu().numpy()).transpose(1,2,0)
        hm_sum = np.sum(hm,axis=-1)
        hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        logger.info("Nose min/max %f %f", hm[...,0].min(),hm[...,0].max())
        cv2.imshow("Hm sum",hm_res)

        logger.info("Target shape: %s",target.shape)

        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused




if __name__ == "__main__":
    run()
