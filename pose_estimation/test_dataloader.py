
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

import sys
sys.path.append("../../HumanTracker/build/py_tracker_tools")
import PyTrackerTools as tt


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
    valid_dataset = dataset.coco_bc(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET, False, None)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=config.WORKERS,
        pin_memory=True
    )


    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    print("Validation demo loop.")

    for i, (input, target, target_weight, meta) in enumerate(valid_loader):
        # logger.info("%d frame. Meta: %s", i, meta)
        logger.info("Image file %s",meta["image"])
        logger.info("Center %s Scale %s",meta['center'], meta['scale'])
        img = np.squeeze(input.cpu().numpy())
        logger.info("Img %s %s",img.shape, img.dtype)

        hm = np.squeeze(target.cpu().numpy()).transpose(1,2,0)
        hm_sum = np.sum(hm[...,:-3],axis=-1)
        hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        # logger.info("Nose min/max %f %f", hm[...,0].min(),hm[...,0].max())
        cv2.imshow("Hm sum",hm_res)
        bc = hm[...,-3]
        logger.info("BC shape %s",bc.shape)

        dx = hm[...,-2]
        dy = hm[...,-1]

        deltas = cv2.resize(np.dstack((dx,dy,np.zeros_like(dx))),(0,0),fx=4.,fy=4.)
        cv2.imshow("Deltas", cv2.normalize(deltas, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC3))
        
        bc_res = cv2.resize(bc, (0,0),fx=4.,fy=4.)

        thre1 = 0.1
        centers = tt.FindPeaks(bc, thre1, 1.0, 1.0)[0]
        if centers is not None:
            for p,j in enumerate(centers):
                x, y, score, _ = j
                w = dx[int(y),int(x)] * bc.shape[1] * 4.0
                h = dy[int(y),int(x)] * bc.shape[0] * 4.0
                print("Center ",p , " ==> ", j, w, h)
                c = (int(x*4.0),int(y*4.0))
                cv2.circle(img, c, 3, [100,50,255], -1)
                cv2.putText(img, str(p)+" (%0.2f)"%score, c, 0, 0.3, [200,100,50],1)
                pt1 = (int(c[0]-w/2),int(c[1]-h/2))
                pt2 = (int(c[0]+w/2),int(c[1]+h/2))
                cv2.rectangle(img,pt1,pt2,[255,255,255], 1)


        bc_norm = cv2.normalize(bc_res, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        bc_norm = cv2.cvtColor(bc_norm, cv2.COLOR_GRAY2BGR)
        for idx, a in enumerate(meta['annotations']):
            bc = np.array(a['barycenter'])
            print("Anno ",idx," bc ", bc)
            if bc[2]>0:
                c = (int(bc[0]*4.0),int(bc[1]*4.0))
                cv2.circle(bc_norm, c, 3, [100,50,255], -1)
                cv2.putText(bc_norm, str(idx), c, 0, 0.5, [200,100,50],2)
                dx,dy = bc[-2]*4.0, bc[-1]*4.0
                pt0 = (int(c[0]-dx/2),int(c[1]-dy/2))
                pt1 = (int(c[0]+dx/2),int(c[1]+dy/2))
                cv2.rectangle(bc_norm, pt0, pt1, [255,255,255],1)

        cv2.imshow("BC", bc_norm)

        overlay = np.copy(img).astype(np.float32)/255. + hm_res[...,np.newaxis]
        overlay = cv2.normalize(overlay, None, 0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("Overlay" , overlay)

        # logger.info("Target shape: %s",target.shape)
        # logger.info("Target shape: %s",target)
        tw = target_weight.cpu().numpy()
        logger.info("Target Weights shape: %s, %d",tw.shape, tw[0,-1])
        if tw[0,-1]==0:
            paused = True
        # print("TW",tw)
        
        # joints = np.hstack((target.cpu().numpy().squeeze()*4.0,tw[0]))
        # print("Joints: \n",joints)
        # img = viz_joints(img, joints)
        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused




if __name__ == "__main__":
    run()
