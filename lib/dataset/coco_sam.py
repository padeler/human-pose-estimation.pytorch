# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.JointsDataset import JointsDataset
from dataset.coco import COCODataset
from nms.nms import oks_nms

import copy
import cv2
import random

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
import torch 


logger = logging.getLogger(__name__)

class COCOSAMDataset(COCODataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)



    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target(nump_joints,2), target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        
        target = np.zeros((self.num_joints, 2), dtype=np.float32)
        # target[:,:] = joints[:,:2]

        feat_stride = self.image_size / self.heatmap_size
        for joint_id in range(self.num_joints):
            mu_x = joints[joint_id][0]# / feat_stride[0]
            mu_y = joints[joint_id][1]# / feat_stride[1]
            
            # XXX This check was for the gaussian. 
            # Maybe better handling if needed for almost out of view joint 
            # in the softargmax mode.
            ul = [int(mu_x/feat_stride[0]), int(mu_y/feat_stride[1])] 
            br = [int(mu_x/feat_stride[0]), int(mu_y/feat_stride[1])]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # joint outside of image. set weight to 0
                target_weight[joint_id] = 0
                mu_x = mu_y = 0

            target[joint_id, :] = [mu_x, mu_y]

        return target, target_weight