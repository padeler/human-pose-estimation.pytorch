'''
Evaluate a joint estimation network on cropped person images from the coco dataset.
'''
import os
import glob
import json
import pprint

import torch
import torchvision.transforms as transforms
import _init_paths

from core.config import config
from utils.utils import create_logger
import models

import torch.backends.cudnn as cudnn


import cv2
import numpy as np
from valid import parse_args,reset_config
from pycocotools.coco import COCO

import sys
sys.path.extend(["../../deepjoint_bodypose_models/build/py_joint_tools","../../deepjoint_bodypose_models"])
import PyMBVCore 
import PyJointTools as jt

from tools.util import visualize_points_jt
from tools.config import coco_part_str


def predict(cropped_frame, model, preproc):
    im = cropped_frame
    # compute output
    tensor = preproc(im)
    batch = tensor.reshape([1,]+list(tensor.shape))
    output = model(batch)
    result = output.cpu().numpy()

    hm = np.squeeze(result).transpose(1,2,0)
    oc = np.zeros_like(hm[...,0])
    nose = hm[...,0]
    print("NOSE:", nose.min(), nose.max())
    hm_sum = np.sum(hm,axis=-1)
    cv2.imshow("hm",hm_sum)
    
    scale = 4.0
    peaks = jt.FindPeaks(hm[...,:17], oc, 0.1, scale, scale)

    return peaks


def Load_COCO_GT(annType ='keypoints', dataType='val2017'):
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    #initialize COCO ground truth api
    dataDir='/media/storage/home/padeler/work/datasets/coco_2017'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    cocoGt=COCO(annFile)
    return cocoGt


def validate_bbox(config, model, preproc):
    dataset_path = "/media/storage/home/padeler/work/datasets/coco_2017/val2017"
    max_count = 100
        

    delay = {
        True: 0,
        False: 5,
    }
    paused = True
    
    out_file = "my_sbl_50epocs_coco_val2017.json"

    print("Loading COCO GT")
    coco_gt = Load_COCO_GT()

    val_set = glob.glob(dataset_path+os.sep+"*.jpg")
    val_set.sort()
    print("Validation set size ",len(val_set))

    k = 0
    count=0
    all_results = []

    # switch to evaluate mode
    model.eval()

    delay = {True:0,False:1}
    paused = True

    k = 0
    idx = 0
    print("Validation demo loop.")
    with torch.no_grad():

        for fname in val_set:
            image_id = int(os.path.os.path.basename(fname)[:-4])
            frame = cv2.imread(fname)
            
            results = []

            # 1. load coco annotations for image_id

            gt_anns_ids = coco_gt.getAnnIds(image_id, catIds=[1],iscrowd=False)
            gt_anns = coco_gt.loadAnns(gt_anns_ids)
            
            pred_size = (256, 192)

            for ann in gt_anns:
                if ann['num_keypoints']>0:
                    
                    bbox = ann['bbox']        
                    x,y,w,h = [int(v) for v in bbox]
                    crop = frame[y:y+h,x:x+w]

                    # scale and pad
                    sc = pred_size[1] / w
                    if sc*h>pred_size[0]:
                        sc = pred_size[0]/h
                    crop_res = cv2.resize(crop,(0,0),fx=sc,fy=sc)
                    
                    padded = np.zeros(pred_size+(3,),dtype=np.uint8)
                    print("PADDED ",padded.shape,"CROP",crop_res.shape)
                    padded[:crop_res.shape[0], :crop_res.shape[1]] = crop_res

                    cv2.imshow("Padded", padded)
                    peaks = predict(padded, model, preproc)
                    res = []
                    viz = np.copy(padded)
                    viz = visualize_points_jt(padded, peaks, coco_part_str[:-1])
                    cv2.imshow("Viz",viz)
                    print("annotation-----")
                    cv2.waitKey(0)


                    # res = util.FieldsToCOCOResultList(image_id, peaks, hyp_vec)
                    # # move results to the coordinate frame of the original image
                    # for d in res:
                    #     joints = d['keypoints']
                    #     for i in range(17):
                    #         joints[i*3] += -cx + x
                    #         joints[i*3+1] += -cy + y 

                    results += res
            
            all_results += results
            img = np.copy(frame)
            # img = util.visualize_coco_results_list(img, results, config.coco_part_str)
            cv2.imshow("Frame", img)

            count+=1

            k = cv2.waitKey(delay[paused])

            if k&0xFF==ord('q') or count==max_count:
                break
            if k&0xFF==ord('p'):
                paused = not paused


    cv2.destroyAllWindows()
    print("Saving keypoint results to ", out_file)
    with open(out_file,"w") as f:
        json.dump(all_results, f)


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
    model = models.pose_resnet.get_pose_net(config, is_train=False)
    # model = eval('models.'+config.MODEL.NAME+'.get_pose_net')( config, is_train=False )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))



    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    preproc = transforms.Compose([ transforms.ToTensor(), normalize, ])

    # evaluate on validation set
    validate_bbox(config, model, preproc)


if __name__ == '__main__':
    main()
