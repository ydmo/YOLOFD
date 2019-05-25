# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import collections
import numpy as np
import cv2
import pickle
import random
import math
import json
import scipy.io
import os


# <WiderFaceParser/>
class WiderFaceParser(object):
    """Some Information about WiderFaceParser"""
    
    # <__init__/>
    def __init__(
        self, 
        path_to_train_images = './WIDER_train/images', 
        wider_face_train_bbx_gt_mat = './wider_face_train.mat', 
        path_to_val_images = './WIDER_val/images',
        wider_face_val_bbx_gt_mat = './wider_face_val.mat',
        path_to_test_images = './WIDER_test/images',
        wider_face_test_filelist_mat = './wider_face_test.mat'
        ):
        super(WiderFaceParser, self).__init__()
        
        self.trainset = []
        self.valset = []
        self.testset = []
        
        # load trian dataset
        try:
            fmat = scipy.io.loadmat(wider_face_train_bbx_gt_mat)
            event_list = fmat.get('event_list')
            file_list = fmat.get('file_list')            
            face_bbx_list = fmat.get('face_bbx_list')
            blur_label_list = fmat.get('blur_label_list')
            expression_label_list = fmat.get('expression_label_list')
            illumination_label_list = fmat.get('illumination_label_list')
            invalid_label_list = fmat.get('invalid_label_list')
            occlusion_label_list = fmat.get('occlusion_label_list')
            pose_label_list = fmat.get('pose_label_list')
            for event_idx, event in enumerate(event_list):
                event_name = event[0][0]
                for im_idx, im in enumerate(file_list[event_idx][0]):
                    im_name = im[0][0]
                    face_bbx = face_bbx_list[event_idx][0][im_idx][0]
                    face_bbx_blur = blur_label_list[event_idx][0][im_idx][0]
                    face_bbx_expression = expression_label_list[event_idx][0][im_idx][0]
                    face_bbx_illumination = illumination_label_list[event_idx][0][im_idx][0]
                    face_bbx_invalid = invalid_label_list[event_idx][0][im_idx][0]
                    face_bbx_occlusion = occlusion_label_list[event_idx][0][im_idx][0]
                    face_bbx_pose = pose_label_list[event_idx][0][im_idx][0]
                    bboxes = []
                    image_info_dict = {}
                    image_info_dict['name'] = im_name
                    image_info_dict['path'] = os.path.join(path_to_train_images, event_name, im_name + '.jpg')
                    image_info_dict['event'] = event_name
                    image_info_dict['bboxes'] = []
                    for i in range(face_bbx.shape[0]):
                        w = float(face_bbx[i][2])
                        h = float(face_bbx[i][3])
                        cx = float(face_bbx[i][0]) + w / 2
                        cy = float(face_bbx[i][1]) + h / 2
                        image_info_dict['bboxes'].append( 
                            { 
                                "xywh": (cx, cy, w, h),
                                'blur': int(face_bbx_blur[i][0]),
                                'expression': int(face_bbx_expression[i][0]),
                                'illumination': int(face_bbx_illumination[i][0]),
                                'invalid': int(face_bbx_invalid[i][0]),
                                'occlusion': int(face_bbx_occlusion[i][0]),
                                'pose': int(face_bbx_pose[i][0]),
                            } 
                            )
                    self.trainset.append(image_info_dict)
        except:
            print('Error in parsing wider_face_train_bbx_gt mat')
        
        # load val dataset
        try:
            fmat = scipy.io.loadmat(wider_face_val_bbx_gt_mat)
            event_list = fmat.get('event_list')
            file_list = fmat.get('file_list')            
            face_bbx_list = fmat.get('face_bbx_list')
            blur_label_list = fmat.get('blur_label_list')
            expression_label_list = fmat.get('expression_label_list')
            illumination_label_list = fmat.get('illumination_label_list')
            invalid_label_list = fmat.get('invalid_label_list')
            occlusion_label_list = fmat.get('occlusion_label_list')
            pose_label_list = fmat.get('pose_label_list')
            for event_idx, event in enumerate(event_list):
                event_name = event[0][0]
                for im_idx, im in enumerate(file_list[event_idx][0]):
                    im_name = im[0][0]
                    face_bbx = face_bbx_list[event_idx][0][im_idx][0]
                    face_bbx_blur = blur_label_list[event_idx][0][im_idx][0]
                    face_bbx_expression = expression_label_list[event_idx][0][im_idx][0]
                    face_bbx_illumination = illumination_label_list[event_idx][0][im_idx][0]
                    face_bbx_invalid = invalid_label_list[event_idx][0][im_idx][0]
                    face_bbx_occlusion = occlusion_label_list[event_idx][0][im_idx][0]
                    face_bbx_pose = pose_label_list[event_idx][0][im_idx][0]
                    bboxes = []
                    image_info_dict = {}
                    image_info_dict['name'] = im_name
                    image_info_dict['path'] = os.path.join(path_to_val_images, event_name, im_name + '.jpg')
                    image_info_dict['event'] = event_name
                    image_info_dict['bboxes'] = []
                    for i in range(face_bbx.shape[0]):
                        w = float(face_bbx[i][2])
                        h = float(face_bbx[i][3])
                        cx = float(face_bbx[i][0]) + w / 2
                        cy = float(face_bbx[i][1]) + h / 2
                        image_info_dict['bboxes'].append( 
                            { 
                                "xywh": (cx, cy, w, h),
                                'blur': int(face_bbx_blur[i][0]),
                                'expression': int(face_bbx_expression[i][0]),
                                'illumination': int(face_bbx_illumination[i][0]),
                                'invalid': int(face_bbx_invalid[i][0]),
                                'occlusion': int(face_bbx_occlusion[i][0]),
                                'pose': int(face_bbx_pose[i][0]),
                            } 
                            )
                    self.trainset.append(image_info_dict)
        except:
            print('Error in parsing wider_face_val_bbx_gt mat')
        
        # load test list
        try:
            fmat = scipy.io.loadmat(wider_face_test_filelist_mat)
            event_list = fmat.get('event_list')
            file_list = fmat.get('file_list')
            for event_idx, event in enumerate(event_list):
                event_name = event[0][0]
                for im_idx, im in enumerate(file_list[event_idx][0]):            
                    im_name = im[0][0]
                    image_info_dict = {}
                    image_info_dict['name'] = im_name
                    image_info_dict['path'] = os.path.join(path_to_val_images, event_name, im_name + '.jpg')
                    image_info_dict['event'] = event_name
                    self.testset.append(image_info_dict)
        except:
            print('Error in parsing wider_face_test_filelist mat')
        
        pass
    # </__init__>

    # <save_to_trainset_json/>
    def save_to_trainset_json(self, json_file = './wider_face_trainset.json'):
        try:
            with open(json_file, 'w') as f:
                json.dump(self.trainset, f)
        except:
            print("Error in save_to_trainset_json")
            return False
        return True
    # </save_to_trainset_json>
    
    # <save_to_valset_json/>
    def save_to_valset_json(self, json_file = './wider_face_valset.json'):
        try:
            with open(json_file, 'w') as f:
                json.dump(self.valset, f)
        except:
            print("Error in save_to_trainset_json")
            return False
        return True
    # </save_to_valset_json>
    
    # <save_to_testset_json/>
    def save_to_testset_json(self, json_file = './wider_face_valset.json'):
        try:
            with open(json_file, 'w') as f:
                json.dump(self.testset, f)
        except:
            print("Error in save_to_testset_json")
            return False
        return True
    # </save_to_testset_json>

# </WiderFaceParser>

# <WiderFaceDataset/>
class WiderFaceDataset(Dataset):
    def __init__(self, pickle_path, img_size = 480, grid_size = 15, max_boxes = 128):
        pickle_load = None
        with open(pickle_path,'rb') as fw:
            pickle_load = pickle.load(fw)
        assert(pickle_load is not None)
        self.data = pickle_load
        self.img_size = img_size
        self.grid_size = grid_size
        self.max_boxes = max_boxes
        pass
    
    def __getitem__(self, index):
        # load data ...
        img_path, objects = self.data[index]["image_path"], self.data[index]["objects"]
        image_np = cv2.imread(img_path, 1)
        # random crop ...
        d = max(128, 0.1 + 0.9 * random.random() * math.sqrt(image_np.shape[0] * image_np.shape[0] + image_np.shape[1] * image_np.shape[1]))
        y1 = random.randint(0, int(image_np.shape[0] / 2))
        x1 = random.randint(0, int(image_np.shape[1] / 2))
        h = int(d * (0.9 + random.random() * 0.2))
        w = int(d * (0.9 + random.random() * 0.2))
        y2 = min(y1 + h, image_np.shape[0])
        x2 = min(x1 + w, image_np.shape[1])
        cropped_img_np = image_np[y1:y2, x1:x2, :]
        # 
        object_np_arr = []
        for idx in range(len(objects)):
            bx1 = max(objects[idx]["x1"] - x1, 0)
            by1 = max(objects[idx]["y1"] - y1, 0)
            bx2 = objects[idx]["x1"] + objects[idx]["w"] - 1 - x1
            by2 = objects[idx]["y1"] + objects[idx]["h"] - 1 - y1
            bcls = 1 # 1 is face 0 is bg
            if bx2 > 0 and by2 > 0 and bx1 < cropped_img_np.shape[1] and by1 < cropped_img_np.shape[0]:
                object_np = torch.zeros((5))
                object_np[1] = float(by1 + by2) / 2 / (cropped_img_np.shape[0]-1)
                object_np[2] = float(bx1 + bx2) / 2 / (cropped_img_np.shape[1]-1)
                object_np[3] = float(min(by2, cropped_img_np.shape[0]) - by1) / (cropped_img_np.shape[0]-1)
                object_np[4] = float(min(bx2, cropped_img_np.shape[1]) - bx1) / (cropped_img_np.shape[1]-1)
                object_np[0] = bcls
                ratio = object_np[3] * object_np[4] * (cropped_img_np.shape[0]-1) * (cropped_img_np.shape[1]-1) / (objects[idx]["w"] * objects[idx]["h"])
                if ratio > 0.5 and object_np[3] * object_np[4] * (cropped_img_np.shape[0]-1) * (cropped_img_np.shape[1]-1) > 256:
                    object_np_arr.append(object_np)
                    # cropped_img_np = cv2.rectangle(cropped_img_np, (bx1, by1), (bx2, by2), [0, 256, 0], 1)
        # resize to image_size and box normalize to 0.0 ~ 1.0
        image = cv2.resize(cropped_img_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        target = torch.zeros(self.max_boxes, 5)
        for idx in range(len(object_np_arr)):
            target[idx, :] = object_np_arr[idx]
        # 
        # cv2.imshow("cropped_img_np", cropped_img_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image, target

    def __len__(self):
        return len(self.data)
# </WiderFaceDataset>

if __name__ == '__main__':
    path_to_train_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_train/images'
    path_to_val_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_val/images'
    path_to_test_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_test/images'
    wider_face_train_bbx_gt_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_train.mat'
    wider_face_val_bbx_gt_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_val.mat'
    wider_face_test_filelist_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_test.mat'
    parser = WiderFaceParser(path_to_train_images, wider_face_train_bbx_gt_mat, path_to_val_images, wider_face_val_bbx_gt_mat, path_to_test_images, wider_face_test_filelist_mat)
    parser.save_to_trainset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_train/wider_face_train.json')
    parser.save_to_valset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_val/wider_face_val.json')
    parser.save_to_testset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_test/wider_face_test.json')
