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


class COCOBCDataset(COCODataset):
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

        height, width = data_numpy.shape[:2]

        # each entry has multiple annotations. 
        # if self.is_train:
        #     # XXX Pick one randomly for augmentation center
        #     pos = random.randint(0,len(db_rec['annotations'])-1)
        #     sel_ann = db_rec['annotations'][pos]
        #     c = sel_ann['center']
        #     s = sel_ann['scale']
        #     # s[0] = max(1.0,s[0]) # dont allow too match zoom in suBjects
        #     # s[1] = max(1.0,s[1])

        # else:
        # # XXX Full image 
        c = np.array([width/2.,height/2.], dtype=np.float32)
        sf = max(width, height)/max(self.image_size)
        s = np.array([sf, sf], dtype=np.float32)

        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0


        
        trans = get_affine_transform(c, s, r, self.image_size, max(self.image_size))
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
                if joints_vis[i, 0] > 0.5:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            annotations_meta.append({
                'joints':joints,
                'joints_vis':joints_vis,
            })


        joints, fields, bc, joints_weight = self.generate_target(annotations_meta)
        
        joints = torch.from_numpy(joints)
        fields = torch.from_numpy(fields)
        bc     = torch.from_numpy(bc)
        joints_weight = torch.from_numpy(joints_weight)
        
        meta = {
            'image': image_file,
            'annotations': annotations_meta,
            'num_pred': len(annotations_meta),
            'center': c,
            'scale': s,
            'rotation': r,
            'flip': flip,
        }

        return input, joints, fields, bc, joints_weight, meta

    def generate_heatmaps_for_annotation(self,p):
        joints = p['joints']
        joints_vis = p['joints_vis']
        jw = joints_vis[:,0]
        feat_stride = self.image_size / self.heatmap_size
        tmp_size = self.sigma * 3
        
        bc_x = bc_y = 0 # barycenter coords
        vis_joint_count = 0 # number of visible joints (for computing barycenter)

        
        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

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

                # update barycenter coords with this joint 
                bc_x += mu_x
                bc_y += mu_y
                vis_joint_count += 1
                
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

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        if vis_joint_count>0:           
            barycenter = [bc_x/vis_joint_count, bc_y/vis_joint_count, vis_joint_count]
        else:
            barycenter = [0, 0, 0]

        return target, jw, barycenter


    def generate_target(self, annotations):
        '''
        :param annotations: list of annotations. Each annotation is a dict containing joints and joints_vs.
        joints:  [num_joints, 3]
        joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'


        joints_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)
        

        # prepare fields base
        # create indices
        indices = np.indices((self.heatmap_size[1],self.heatmap_size[0])).astype(np.float32)
        indices[0] /= self.heatmap_size[1] # normalize to [0,1)
        indices[1] /= self.heatmap_size[0]

        target_bc = np.zeros((1, self.heatmap_size[1],self.heatmap_size[0]), dtype=np.float32)
        target_fields = np.zeros((self.num_joints*2,self.heatmap_size[1],self.heatmap_size[0]), dtype=np.float32)

        bc_weight = 0
        for p in annotations:
            heatmaps, jw, barycenter = self.generate_heatmaps_for_annotation(p)
            p['barycenter'] = barycenter
            # print("Barycenter", barycenter)
            
            if barycenter[2]>0: # valid annotation
                bc_weight = 1 # at least one valid annotation in this image
                joints += heatmaps
                joints_weight[:] += jw[:, np.newaxis]

                # create and add barycenter gaussian
                bc = self.generate_barycenter_heatmap(barycenter)
                target_bc[0] += bc
                
                # create fields
                fields = self.generate_fields_for_barycenter(barycenter, indices)
                # mask fields using the heatmaps
                hm_mask = heatmaps>0.25
                for j in range(self.num_joints):
                    target_fields[2*j][hm_mask[j]] = fields[2*j][hm_mask[j]]
                    target_fields[2*j+1][hm_mask[j]] = fields[2*j+1][hm_mask[j]]

                # target_fields[hm_mask] = fields[hm_mask]


        joints_weight[joints_weight>0] = 1
        if bc_weight==0 and np.sum(joints_weight)>0:
            raise RuntimeError("BC Weights must be 1 if joints are visible")

        return joints, target_fields, target_bc, joints_weight


    def generate_barycenter_heatmap(self, barycenter):
        mu_x, mu_y, _ = barycenter

        tmp_size = self.sigma * 3

        target = np.zeros((self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # This should not happen since harycenter by definition is in the image
            raise RuntimeError("Barrycenter is outside the image %d,%d"%(mu_x,mu_y))


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

        target[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target




    def generate_fields_for_barycenter(self, barycenter, indices_base):

        fields = []
        bcx, bcy = barycenter[:2]
        # normalize barycenter coords
        bcx /= self.heatmap_size[0] 
        bcy /= self.heatmap_size[1]

        indices = np.copy(indices_base)
        indices[0] -= bcy
        indices[1] -= bcx

        # one pair (X,Y) copy for each joint
        fields = np.stack((indices,)*self.num_joints).reshape(-1,self.heatmap_size[1],self.heatmap_size[0])
        return fields




    # need double check this API and classes field
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.image_set)

        # person x (keypoints)
        _kpts = []
        for idx, sk in enumerate(preds):
            _kpts.append({
                'keypoints': sk.joints,
                # 'center': all_boxes[idx][0:2],
                # 'scale': all_boxes[idx][2:4],
                'area': sk.area,
                # 'score': all_boxes[idx][5],
                'image': int(sk.image_path[-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                # box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score #* box_score
            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0
