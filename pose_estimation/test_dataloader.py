
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

from utils.skeleton_tools import visualize_skeletons, get_batch_predictions

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



COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [
              0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]



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

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        batch_images = meta['image']
        # score = meta['score'].numpy()

        batch_predictions = get_batch_predictions(config, target.clone().cpu().numpy(), c, s, batch_images)
        full_img = cv2.imread(batch_images[0])
        sk_viz = visualize_skeletons(full_img, batch_predictions)
        print("Skeletons found: ",len(batch_predictions)," original canvas ", full_img.shape)
        cv2.imshow("SKel Viz",sk_viz)



        target = np.squeeze(target.cpu().numpy()).transpose(1,2,0)
        bc = target[...,-1]
        fields = target[...,17:-1]
        hm = target[...,:17]
        
        hm_sum = np.sum(hm[...,:-3],axis=-1)
        hm_res = cv2.resize(hm_sum, (0,0),fx=4.,fy=4.)
        # logger.info("Nose min/max %f %f", hm[...,0].min(),hm[...,0].max())
        cv2.imshow("Hm sum",hm_res)

        overlay = np.copy(img).astype(np.float32)/255. + hm_res[...,np.newaxis]
        overlay = cv2.normalize(overlay, None, 0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("Overlay" , overlay)

        fields_dx = np.sum(fields[...,::2],axis=-1)
        fields_dy = np.sum(fields[...,1::2],axis=-1)
        fields_rgb = np.dstack((fields_dx,fields_dy,np.zeros_like(fields_dx)))
        fields_rgb = cv2.normalize(fields_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
        fields_rgb[...,2] = cv2.normalize(bc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        cv2.imshow("Fields RGB",cv2.resize(fields_rgb, (0,0),fx=4.,fy=4.))


        heatmaps = hm
        print("Heatmaps, bc ===>",heatmaps.shape,bc.shape)


        # thre1 = 0.1
        # joints = tt.FindPeaks(heatmaps, thre1, 1.0, 1.0)
        # mask = np.zeros_like(bc)

        # for i, jg in enumerate(joints):
        #     if jg is not None:
        #         for j, cand in enumerate(jg):
        #             x, y, score, _ = cand

        #             dx =  fields[int(y), int(x), i*2+1] * bc.shape[1]
        #             dy =  fields[int(y), int(x), i*2]   * bc.shape[0]
        #             pos_y = int(y-dy)
        #             pos_x = int(x-dx)
        #             print(i,",",j," ==> ",x, y, dx, dy, "===>",pos_x,pos_y)

        #             if pos_x>=0 and pos_y>=0 and pos_x<bc.shape[1] and pos_y<bc.shape[0]:
        #                 mask[pos_y, pos_x] += 1.0


        # mask_res = cv2.resize(mask,(0,0),fx=4.0,fy=4.0)
        # cv2.imshow("Instance Mask",cv2.normalize(mask_res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))


        for idx, a in enumerate(meta['annotations']):
            bc = np.array(a['barycenter'])
            print("Anno ",idx," bc ", bc)
            if bc[2]>0:
                c = (int(bc[0]*4.0),int(bc[1]*4.0))
                cv2.circle(img, c, 3, [100,50,255], -1)
                cv2.putText(img, str(idx), c, 0, 0.5, [200,100,50],2)


        # tw = target_weight.cpu().numpy()
        # logger.info("Target Weights shape: %s, %d",tw.shape, tw[0,-1])
        

        cv2.imshow("Input", img)
        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q'):
            break
        if k&0xFF==ord('p'):
            paused = not paused




if __name__ == "__main__":
    run()
