# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2 

import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.coco import COCODataset
from nms.nms import oks_nms


from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class COCOFieldsDataset(COCODataset):
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

        # Repackage DB to contain multiple annotations per image
        if not (self.is_train or self.use_gt_bbox):
            raise NotImplementedError("Non coco annotation not supported")
 
        # use ground truth bbox
        self.db = self._repackage_coco_img_anns(self.db)
        logger.info('=> Repackaged db to {} samples'.format(len(self.db)))

        


    def _repackage_coco_img_anns(self, db):
        db_index = 0
        new_db = []
        for index in self.image_set_index:
            pos = db_index
            
            # same img_index annotations are sequential in the db
            while pos<len(db) and db[pos]['img_index']==index:
                pos+=1  
            
            if (pos-db_index)>0: # add entry to new_db
                new_db.append({
                    "image": db[db_index]['image'],
                    "img_index": index,
                    "annotations": db[db_index:pos]
                })

            db_index = pos

        return new_db



    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        flip = self.is_train and self.flip and random.random() <= 0.5

        if flip:
            data_numpy = data_numpy[:, ::-1, :]
            for p in db_rec['annotations']:
                
                joints, joints_vis = fliplr_joints(
                    p['joints_3d'], p['joints_3d_vis'], 
                    data_numpy.shape[1], self.flip_pairs)
                center = p['center']
                center[0] = data_numpy.shape[1] - center[0] - 1
                p['joints_3d'] = joints
                p['joints_3d_vis'] = joints_vis
                p['center'] = center


        # each entry has multiple annotations. 
        # XXX Test this: Pick one randomly for augmentation center
        # pos = random.random(0,len(db_rec['annotations'])-1)
        # sel_ann = db_rec['annotations'][pos]
        height, width = data_numpy.shape[:2]
        c = np.array([width/2.,height/2.], dtype=np.float32)
        s = np.array([1., 1.], dtype=np.float32) # TODO Check if this is the correct scale for full image?
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0


        
        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        annotations_meta = []
        for p in db_rec['annotations']:
            joints_vis = p['joints_3d_vis']
            joints = p['joints_3d']

            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            annotations_meta.append({
                'joints':joints,
                'joints_vis':joints_vis,
            })

        target, target_weight = self.generate_target(annotations_meta)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'annotations': annotations_meta,
            'center': c,
            'scale': s,
            'rotation': r,
            'flip': flip,
        }

        return input, target, target_weight, meta


    def generate_target(self, annotations):
        '''
        :param annotations: list of annotations. Each annotation is a dict containing joints and joints_vs.
        joints:  [num_joints, 3]
        joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'


        target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        feat_stride = self.image_size / self.heatmap_size
        tmp_size = self.sigma * 3

        for p in annotations:
            joints = p['joints']
            joints_vis = p['joints_vis']
            jw = joints_vis[:,0]

            for joint_id in range(self.num_joints):
                if jw[joint_id] > 0: # if joint is annotated
                    mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                    mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        jw[joint_id] = 0
                        continue

                    # # Generate gaussian
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    # TODO handle overlapping gaussians 
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            
            target_weight[:] += jw[:, np.newaxis]

        target_weight[target_weight>0] = 1
        return target, target_weight
