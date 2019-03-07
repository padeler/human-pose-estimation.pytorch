import numpy as np
import cv2
import sys

from utils.transforms import transform_preds

sys.path.append("../../HumanTracker/build/py_tracker_tools")
import PyTrackerTools as tt

import math


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [
              0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LIMBSEQ = [
    [0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], [6,8], [7,9], [8,10],
    [5,6], [5,11], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16],
]

class Skeleton(object):
    def __init__(self, num_joints= 17):
        self.joints = np.zeros((num_joints,3),dtype=np.float32) # x,y, score
        self.indices = np.array([-1,]*num_joints, dtype=np.int32) # -1 is invalid
        self.dist = np.zeros((num_joints,),dtype=np.float32) # joint distances from closest barycenter
        self.median = 0
        self.mean = 0
        self.area = 0
        self.image_path = None


def visualize_joints(canvas, joints):
    for i, group in enumerate(joints):
        if group is not None:
            for joint in group:
                x,y = joint[0:2]
                cv2.circle(canvas, (int(x*4.0),int(y*4.0)) , 4, COLORS[i], thickness=-1)

    return canvas

def visualize_skeletons(canvas, skeletons):
    stickwidth = 3
    for s_idx, skel in enumerate(skeletons): 
        # print(s_idx, "Sk joints",skel.joints)               
        for limp in LIMBSEQ:
            start, end = int(skel.indices[limp[0]]),int(skel.indices[limp[1]])
            if start>-1 and end>-1:
                p0 = skel.joints[limp[0]][:2]
                p1 = skel.joints[limp[1]][:2]
                cv2.line(canvas, (int(p0[0]),int(p0[1])),(int(p1[0]),int(p1[1])),COLORS[limp[0]], stickwidth, lineType=cv2.LINE_AA)

    return canvas


def findJointBC(joint, j_idx, fields):
    x, y, score, _ = joint
    h,w = fields.shape[:2]
    
    d = 1
    x0 = max(int(x-d),0)
    y0 = max(int(y-d),0)
    x1 = min(int(x+d),w-1)
    y1 = min(int(y+d),h-1)

    dx = np.median(fields[y0:y1,x0:x1,j_idx*2+1]) * w
    dy = np.median(fields[y0:y1,x0:x1,j_idx*2]) * h
    
    # dx = fields[int(y), int(x), j_idx*2+1] * fields.shape[1]
    # dy =  fields[int(y), int(x), j_idx*2]   * fields.shape[0]

    jbc_y = y - dy
    jbc_x = x - dx

    return [jbc_x, jbc_y]


def findClosestJoint(x,y, jg_bc):
    best_dist = 10000.0
    best_idx = -1
    for idx, bc in enumerate(jg_bc):
        cx,cy = bc
        dist = math.sqrt((x-cx)**2 + (y-cy)**2)
        if dist<best_dist:
            best_dist = dist
            best_idx = idx
    
    return best_idx, best_dist


def fields2skeletons(heatmaps, fields, bc):
    thre1 = 0.1
    thre2 = 0.1
    
    # get barycenters and joints
    joints = tt.FindPeaks(heatmaps, thre1, 1.0, 1.0)
    barycenters = tt.FindPeaks(bc, thre2, 1.0, 1.0)[0]
    
    joints_bc = []
    #Compute the BC Candidate position for each joint
    # from the fields
    for j_idx,joints_group in enumerate(joints):
        jg_bc = []
        if joints_group is not None:
            for i, cand in enumerate(joints_group):
                j_bc = findJointBC(cand, j_idx, fields)
                jg_bc.append(j_bc)
                # print(j_idx,i,"cand",cand,"bc",j_bc)
        
        joints_bc.append(jg_bc)

    
    #for each barycenter compute the minimum distance from each joint
    skeletons = []
    if barycenters is not None:
        
        for idx, bc in enumerate(barycenters):
            # print(idx, ",", bc)
            x, y, score, _ = bc
            sk = Skeleton()

            for j_idx, jg_bc in enumerate(joints_bc):
                best_idx, best_dist = findClosestJoint(x,y,jg_bc)
                # print(idx,j_idx,"===>",best_idx,best_dist)
                if best_idx>-1:
                    sk.indices[j_idx] = best_idx
                    sk.dist[j_idx] = best_dist
                    sk.joints[j_idx,:] = joints[j_idx][best_idx][:3]
                

                # skeletons[idx,j_idx,0] = best_idx
                # skeletons[idx,j_idx,1] = best_dist
            
            # compute meta (mean and median)
            sk_dist = sk.dist[sk.indices>-1]
            sk.mean = np.mean(sk_dist)
            sk.median = np.median(sk_dist)
            
            skeletons.append(sk)
            
    
    # TODO 
    # when a joint is used by multiple skeletons
    #  - keep the one that is closest to its barycenter
    #  - or keep the ones that are less than the median joint distance of the skeleton

    # for each skeleton check if a joint is shared
    if True:
        for idx, sk in enumerate(skeletons): # for all skeletons
            for j_idx in range(len(sk.indices)):  
                if sk.indices[j_idx]>-1: # for each assigned joint 
                    for t_idx, tsk in enumerate(skeletons[(idx+1):]): # for the rest of the skeletons
                        if tsk.indices[j_idx]==sk.indices[j_idx]: # if joint is shared
                            if sk.dist[j_idx]<tsk.dist[j_idx]:
                                tsk.indices[j_idx] = -1
                            else:
                                sk.indices[j_idx] = -1
                                break # move to next joint of sk

                            # if sk[j_idx,1]>skeletons_meta[idx][1]: # if dist is less than skeletons median dist
                            #     sk[j_idx,0] = -1 # remove it 
                            #     print("Removing joint ",idx, j_idx)
                            #     break
                            # if tsk[j_idx,1]>skeletons_meta[t_idx][1]: # same for test skeleton
                            #     tsk[j_idx,0] = -1 
                            #     print("Removing joint ",t_idx, j_idx)
    

    return joints, barycenters, joints_bc, skeletons

    

def _get_image_predictions(heatmaps, fields, bc):

    joints, sk_centers, joints_bc, skeletons = fields2skeletons(heatmaps, fields, bc)
    
    return skeletons

def compute_area(joints, indices):

    ul = [10000,10000]
    br = [-10000,-10000]
    count =0
    for idx,v in enumerate(indices):
        if v>-1:
            count +=1
            x,y = joints[idx][:2]
            ul[0] = min(ul[0],x)
            ul[1] = min(ul[1],y)
            
            br[0] = max(br[0],x)
            br[1] = max(br[1],y)
        

    if count>0:
        dx = br[0] - ul[0]
        dy = br[1] - ul[1]
        return dx*dy

    return 0
            

def get_batch_predictions(config, batch_joints, batch_fields, batch_bc, batch_centers, batch_scales, batch_images):
    
    heatmap_width, heatmap_height = config.MODEL.EXTRA.HEATMAP_SIZE
    num_joints = config.MODEL.NUM_JOINTS

    batch_predictions = []

    for im_hm, im_fields, im_bc, center, scale, image_path in zip(batch_joints, batch_fields, batch_bc, batch_centers, batch_scales, batch_images):
        
        hm = im_hm.transpose(1,2,0)
        fields = im_fields.transpose(1,2,0)
        bc = np.squeeze(im_bc)

        image_predictions = _get_image_predictions(hm, fields, bc)

        # Transform to full image size and add image path metadata
        for sk in image_predictions:
            sk.joints[:, :2] = transform_preds(sk.joints[:, :2], center, scale, [heatmap_width, heatmap_height], max(config.MODEL.IMAGE_SIZE))
            sk.area = compute_area(sk.joints,sk.indices)
            sk.image_path = image_path

        batch_predictions.extend(image_predictions)


    return batch_predictions

