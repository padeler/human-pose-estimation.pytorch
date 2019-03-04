import numpy as np
import cv2
import sys

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

def visualize_skeletons(canvas, joints, skeletons, show_peaks=True, show_skeletons=True):

    
    if show_peaks:
        for i, group in enumerate(joints):
            if group is not None:
                for joint in group:
                    x,y = joint[0:2]
                    cv2.circle(canvas, (int(x*4.0),int(y*4.0)) , 4, COLORS[i], thickness=-1)

    stickwidth = 3
    if show_skeletons:
        for s_idx, skel in enumerate(skeletons):                
            for limp in LIMBSEQ:
                start,end = int(skel[limp[0],0]),int(skel[limp[1],0])
                if start>-1 and end>-1:
                    p0 = joints[limp[0]][start][:2]
                    p1 = joints[limp[1]][end][:2]
                    cv2.line(canvas, (int(p0[0]*4.0),int(p0[1]*4.0)),(int(p1[0]*4.0),int(p1[1]*4.0)),COLORS[limp[0]], stickwidth, lineType=cv2.LINE_AA)



    return canvas


def findJointBC(joint, j_idx, fields):
    x, y, score, _ = joint

    dx = fields[int(y), int(x), j_idx*2+1] * fields.shape[1]
    dy =  fields[int(y), int(x), j_idx*2]   * fields.shape[0]

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
    thre1 = 0.2
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
                print(j_idx,i,"cand",cand,"bc",j_bc)
        
        joints_bc.append(jg_bc)

    
    #for each barycenter compute the minimum distance from each joint
    skeletons = []
    if barycenters is not None:
        skeletons = np.zeros((len(barycenters), len(joints_bc), 2),dtype=np.float32)
        skeletons[:,:,0] = -1 # invalid joint flag

        for idx, bc in enumerate(barycenters):
            print(idx, ",", bc)
            x, y, score, _ = bc

            for j_idx, jg_bc in enumerate(joints_bc):
                best_idx, best_dist = findClosestJoint(x,y,jg_bc)
                print(idx,j_idx,"===>",best_idx,best_dist)
                skeletons[idx,j_idx,0] = best_idx
                skeletons[idx,j_idx,1] = best_dist

        
    # TODO 
    # when a joint is used by multiple skeletons
    #  - keep the one that is closest to its barycenter
    #  - or keep the ones that are less than the median joint distance of the skeleton


    return joints,barycenters,joints_bc, skeletons

    
    