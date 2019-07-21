# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from utils.box import bbox_iou

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
                    self.valset.append(image_info_dict)
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

    # <save_to_trainset_pickle/>
    def save_to_trainset_pickle(self, pickle_file = './wider_face_trainset.json'):
        try:
            with open(pickle_file, 'w') as f:
                pickle.dump(self.trainset, f)
        except:
            print("Error in save_to_trainset_json")
            return False
        return True        
    # </save_to_trainset_pickle>
    
    # <save_to_valset_json/>
    def save_to_valset_json(self, json_file = './wider_face_valset.json'):
        try:
            with open(json_file, 'w') as f:
                json.dump(self.valset, f)
        except:
            print("Error in save_to_valset_json")
            return False
        return True
    # </save_to_valset_json>

    # <save_to_valset_pickle/>
    def save_to_valset_pickle(self, pickle_file = './wider_face_valset.pickle'):
        try:
            with open(pickle_file, 'w') as f:
                pickle.dump(self.valset, f)
        except:
            print("Error in save_to_valset_pickle")
            return False
        return True
    # </save_to_valset_pickle>
    
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

    # <save_to_testset_pickle/>
    def save_to_testset_pickle(self, pickle_file = './wider_face_valset.pickle'):
        try:
            with open(pickle_file, 'w') as f:
                pickle.dump(self.testset, f)
        except:
            print("Error in save_to_testset_pickle")
            return False
        return True
    # </save_to_testset_pickle>
# </WiderFaceParser>

# <class WiderFaceDataRandomCrop(object)>
class WiderFaceDataRandomCrop(object):
    # <method __init__>
    def __init__(self, min_crop_size=(512, 512)):
        self._min_crop_size = min_crop_size    
    # <method __init__>

    # <method __call__>
    def __call__(self, *args, **kwargs):
        image_np = args[0]["image"]
        boxes_np = args[0]["target"]
        # 
        image_h_origin = image_np.shape[0]
        image_w_origin = image_np.shape[1]
        # random crop ...
        crop_w = random.randint(min(image_w_origin, self._min_crop_size[0]), image_w_origin)
        crop_h = random.randint(min(image_h_origin, self._min_crop_size[1]), image_h_origin)
        crop_x1 = random.randint(0, max(0, image_w_origin - crop_w - 1))
        crop_y1 = random.randint(0, max(0, image_h_origin - crop_h - 1))
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        cropped_img_np = image_np[crop_y1:crop_y2, crop_x1:crop_x2, :]
        # filting the boxes of or corp image ...
        cropped_boxes_np = boxes_np.copy()
        for box_idx in range(cropped_boxes_np.shape[0]):
            x = cropped_boxes_np[box_idx, 0]
            y = cropped_boxes_np[box_idx, 1]
            if x < crop_x1 or x >= crop_x2 or y < crop_y1 or y >= crop_y2:
                cropped_boxes_np[box_idx, 4] = 0
            cropped_boxes_np[box_idx, 0] = (x - crop_x1) / crop_w
            cropped_boxes_np[box_idx, 1] = (y - crop_y1) / crop_h
            cropped_boxes_np[box_idx, 2] = cropped_boxes_np[box_idx, 2] / crop_w
            cropped_boxes_np[box_idx, 3] = cropped_boxes_np[box_idx, 3] / crop_h            
        # 
        return {"image": cropped_img_np, "target": cropped_boxes_np}
    # <method __call__>
# <class WiderFaceDataRandomCrop(object)>

# <class WiderFaceDataImgResize(object)>
class WiderFaceDataImgResize(object):
    # <method __init__>
    def __init__(self, size):
        self._size = size
    # <method __init__>

    # <method __call__>
    def __call__(self, *args, **kwargs):
        image_np = args[0]["image"]
        boxes_np = args[0]["target"]
        image_np = cv2.resize(image_np, self._size, interpolation=cv2.INTER_CUBIC)
        return {"image": image_np.astype(np.float32), "target": boxes_np}
    # <method __call__>
# <class WiderFaceDataImgResize(object)>

# <class WiderFaceDataToTensor(object)>
class WiderFaceDataToTensor(object):
    # <method __init__>
    def __init__(self):
        pass
    # <method __init__>

    # <method __call__>
    def __call__(self, *args, **kwargs):
        image_np = np.transpose(args[0]["image"], (2, 0, 1))
        boxes_np = args[0]["target"]
        return {"image": torch.from_numpy(image_np), "target": torch.from_numpy(boxes_np)}
    # <method __call__>
# <class WiderFaceDataToTensor(object)>

# <class WiderFaceDataBoxEncode(object)>
class WiderFaceDataBoxEncode(object):
    """
    grids (tuple of tuple): output feature size. \n
    anchors (tuple of tuple of tuple): anchor boxes size of each grid size. \n
    num_classes (int): number of object classes. \n
    ignore_threshold (float): iou threshold to ignoring the anchor boxes. \n
    """
    # <method __init__>
    def __init__(
        self,
        grids = (
            (52, 52),
            (26, 26), 
            (13, 13),
            ), 
        anchors = (
            ((10, 13), (16,  30), (33, 23)), 
            ((30, 61), (62,  45), (59, 119)), 
            ((116,90), (156,198), (373,326))
            ),
        num_classes = 2,
        ignore_threshold = 0.5,
        ):
        assert(len(grids) == len(anchors))
        self._grids = grids
        self._anchors = np.array(anchors).astype(np.float32)
        self._num_classes = num_classes
        self._ignore_threshold = ignore_threshold
    # <method __init__>
    
    # <method __call__>
    def __call__(self, *args, **kwargs):
        image = args[0]["image"]
        boxes = args[0]["target"]
        # 
        nI_c, nI_h, nI_w = image.size()
        # 
        masks = []
        conf_masks = []
        fms = []
        for nL in range(len(self._grids)):            
            nT = boxes.size(0) # number of targets
            nA = len(self._anchors[nL])
            nC = self._num_classes
            nG_w = self._grids[nL][0]
            nG_h = self._grids[nL][1]
            # 
            tx =    torch.zeros(nG_h, nG_w, nA, 1)
            ty =    torch.zeros(nG_h, nG_w, nA, 1)
            tw =    torch.zeros(nG_h, nG_w, nA, 1)
            th =    torch.zeros(nG_h, nG_w, nA, 1)
            tconf = torch.zeros(nG_h, nG_w, nA, 1)
            tcls =  torch.zeros(nG_h, nG_w, nA, nC)
            # 
            mask =  torch.zeros(nG_h, nG_w, nA)
            conf_mask = torch.ones(nG_h, nG_w, nA)
            # 
            for t in range(nT): # "t" is target index
                if boxes[t, 4] == 0:
                    continue
                gx = boxes[t, 0] * nG_w # Center x of the ground turth box
                gy = boxes[t, 1] * nG_h # Center y of the ground turth box
                gw = boxes[t, 2] * nG_w # Width of the ground turth box
                gh = boxes[t, 3] * nG_h # Height of the ground turth box
                # get grid position(gj, gi) that the target box is belong to.
                gj = int(gy)
                gi = int(gx)
                # 
                gt_box = torch.FloatTensor([(0, 0, gw, gh) for _ in range(nA)]) # gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                anchor_shapes = torch.cat( ( torch.zeros(nA, 2), torch.from_numpy(self._anchors[0]) ) , 1)
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # try:
                conf_mask[gj, gi, anch_ious > self._ignore_threshold] = 0 # Where the overlap is larger than threshold set mask to zero (ignore)
                # except:
                #     import pdb; pdb.set_trace()
                # 
                best_n = torch.argmax(anch_ious)
                # 
                mask[gj, gi, best_n] = 1
                conf_mask[gj, gi, best_n] = 1
                # 
                tx[gj, gi, best_n] = gx - gi
                ty[gj, gi, best_n] = gy - gj
                tw[gj, gi, best_n] = torch.log(gw / self._anchors[0][best_n][0] + 1e-9)
                th[gj, gi, best_n] = torch.log(gh / self._anchors[0][best_n][1] + 1e-9)
                tconf[gj, gi, best_n] = 1
                tcls[gj, gi, best_n, int(boxes[t, 4])] = 1
            # endfor
            fm = torch.cat((tx, ty, tw, th, tconf, tcls), 3)
            # 
            fms.append(fm.view(-1, 4 + 1 + nC)) # [nGh * nGw * nA, 4+1+nC]
            masks.append(mask.view(-1, 1))
            conf_masks.append(conf_mask.view(-1, 1))
        # endfor
        ret = {"image": image, "target": (torch.cat(fms, 0), torch.cat(masks, 0), torch.cat(conf_masks, 0))}
        return ret
    # <method __call__>
# <class WiderFaceDataBoxEncode(object)>

# <class WiderFaceDataset>
class WiderFaceDataset(torch.utils.data.Dataset):
    """
    WiderFaceDataset is a Dataset
    """
    # <method __init__>
    def __init__(self, json_path = None, pickle_path = None, transform=None):
        if json_path:
            with open(json_path, 'r') as fw:
                self.data = json.load(fw)
        elif pickle_path:
            with open(pickle_path,'rb') as fw:
                self.data = pickle.load(fw)
        else:
            RuntimeError("Error in WiderFaceDataset.__init__, both json_path and pickle_path are None")
        # endif
        # 
        self._transform = transform
        pass
    # <method __init__>
    
    # <method __len__>
    def __len__(self):
        return len(self.data)
    # <method __len__>

    # <method __getitem__>
    def __getitem__(self, index):
        # load data ...
        img_path, bboxes = self.data[index]["path"], self.data[index]["bboxes"]
        # get numpy array of image
        image_np = cv2.imread(img_path, 1)
        # get numpy array of boxes
        remain_boxes = []
        for bbox in bboxes:
            x = bbox['xywh'][0] # center x of bounding box in origin image
            y = bbox['xywh'][1] # center y of bounding box in origin image
            w = bbox['xywh'][2] # width of bounding box in origin image
            h = bbox['xywh'][3] # height of bounding box in origin image
            blur = bbox['blur']
            expression = bbox['expression']
            illumination = bbox['illumination']
            occlusion = bbox['occlusion']
            pose = bbox['pose']
            # 
            bbox_np = np.zeros(10)
            bbox_np[0] = x
            bbox_np[1] = y
            bbox_np[2] = w
            bbox_np[3] = h
            bbox_np[4] = 1
            bbox_np[5] = blur
            bbox_np[6] = expression
            bbox_np[7] = illumination
            bbox_np[8] = occlusion
            bbox_np[9] = pose
            # 
            remain_boxes.append(np.expand_dims(bbox_np, axis=0))
        # endfor
        if len(remain_boxes):
            boxes_np = np.concatenate(remain_boxes, axis=0)
        else:
            boxes_np = np.zeros((0, 10))
        # endif
        data = {"image": image_np, "target": boxes_np}
        if self._transform:
            data = self._transform(data)
        # endif
        return data
    # <method __getitem__>
# <class WiderFaceDataset>



if __name__ == '__main__':
    # path_to_train_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_train/images'
    # path_to_val_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_val/images'
    # path_to_test_images = '/home/yuda/projects/YOLOFD/data/wider_face/WIDER_test/images'
    # wider_face_train_bbx_gt_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_train.mat'
    # wider_face_val_bbx_gt_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_val.mat'
    # wider_face_test_filelist_mat = '/home/yuda/projects/YOLOFD/data/wider_face/wider_face_split/wider_face_test.mat'
    # parser = WiderFaceParser(path_to_train_images, wider_face_train_bbx_gt_mat, path_to_val_images, wider_face_val_bbx_gt_mat, path_to_test_images, wider_face_test_filelist_mat)
    # parser.save_to_trainset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_train/wider_face_train.json')
    # parser.save_to_valset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_val/wider_face_val.json')
    # parser.save_to_testset_json('/home/yuda/projects/YOLOFD/data/wider_face/WIDER_test/wider_face_test.json')

    dataset = WiderFaceDataset(
        json_path='/home/yuda/projects/YOLOFD/data/wider_face/WIDER_val/wider_face_val.json', 
        transform=transforms.Compose([
            WiderFaceDataRandomCrop((512, 512)), 
            WiderFaceDataImgResize((416, 416)),
            WiderFaceDataToTensor(),
            WiderFaceDataBoxEncode()
            ])
        )
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        img_show = np.transpose(sample_batched["image"][0].numpy(), (1, 2, 0)) / 256.0
        # for idx in range(boxes_np.shape[0]):
        #     x = boxes_np[idx][0] * img_show.shape[1]
        #     y = boxes_np[idx][1] * img_show.shape[0]
        #     w = boxes_np[idx][2] * img_show.shape[1]
        #     h = boxes_np[idx][3] * img_show.shape[0]
        #     if boxes_np[idx][4]:
        #         img_show = cv2.rectangle(img_show, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h // 2)), [0, 256, 0], 1)
        cv2.imshow("img_show", img_show)
        cv2.waitKey(0)  
        break 