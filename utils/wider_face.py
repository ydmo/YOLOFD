# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pickle

gt_keys = ["x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"]

def parseSet(imgsfolder = "", filelist = ""):
    line = 0
    print(line)
    # 
    wider_dicts = []
    with open(filelist, 'r') as file:
        # 
        while(True):
            wider_dict = {}
            # 
            img_path = file.readline()[:-1]
            line += 1; print(line)
            if img_path is None or img_path is "":
                break
            # 
            img = cv2.imread(imgsfolder+"/"+img_path, 1)
            # if img is None:
            #     assert(0)
            # 
            num_obj_str = file.readline()[:-1]
            line += 1; print(line)
            if num_obj_str is None:
                break
            if num_obj_str is "":
                break
            num_obj = int(num_obj_str)
            if num_obj == 0:
                assert(0)
                break
            # 
            wider_dict["image_path"] = imgsfolder+"/"+img_path
            # 
            gt_dicts = []
            for n in range(num_obj):
                targets_str = file.readline()[:-1]
                line += 1; print(line)
                if targets_str is None:
                    assert(0)
                targets_str = targets_str.split(" ")
                gt_dict = { }
                for idx in range(len(targets_str)):
                    if len(targets_str[idx]) != 0:
                        gt_dict[gt_keys[idx]] = int(targets_str[idx])
                gt_dicts.append(gt_dict)
                img = cv2.rectangle(img, (gt_dict["x1"], gt_dict["y1"]), (gt_dict["x1"]+gt_dict["w"], gt_dict["y1"]+gt_dict["h"]), [0, 256, 0], 1)
            wider_dict["objects"] = gt_dicts
            # 
            cv2.imshow("img", img)
            cv2.waitKey(0)
            wider_dicts.append(wider_dict)
    # 
    cv2.destroyAllWindows()
    # 
    return wider_dicts

dicts_save = parseSet( \
    imgsfolder = "/home/gtx1060/Documents/DataSets/wider_face/WIDER_val/images", \
    filelist = "/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt")

with open('/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.pkl','wb') as fw:
    pickle.dump(dicts_save, fw, -1) 

dicts_save = parseSet( \
    imgsfolder = "/home/gtx1060/Documents/DataSets/wider_face/WIDER_train/images", \
    filelist = "/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt")

with open('/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl','wb') as fw:
    pickle.dump(dicts_save, fw, -1) 

# with open('/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl','rb') as fw:
#     dicts_load = pickle.load(fw)
#     for dict in dicts_load:
#         img = cv2.imread(dict["image_path"], 1)
#         for obj in dict["objects"]:
#             img = cv2.rectangle(img, (obj["x1"], obj["y1"]), (obj["x1"]+obj["w"], obj["y1"]+obj["h"]), [0, 256, 0], 1)
#         cv2.imshow("img", img)
#         cv2.waitKey(1)
# cv2.destroyAllWindows()