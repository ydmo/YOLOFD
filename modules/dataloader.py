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

class WiderFaceDataset(Dataset):
    def __init__(self, pickle_path, img_size = 480, grid_size = 15):
        pickle_load = None
        with open(pickle_path,'rb') as fw:
            pickle_load = pickle.load(fw)
        assert(pickle_load is not None)
        self.data = pickle_load
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
            if bx2 > 0 and by2 > 0 and bx1 < cropped_img_np.shape[1] and by1 < cropped_img_np.shape[0]:
                object_np = np.zeros((1, 4))
                object_np[0, 0] = (bx1 + bx2) / 2
                object_np[0, 1] = (by1 + by2) / 2
                object_np[0, 2] = (min(bx2, cropped_img_np.shape[1]) - bx1)
                object_np[0, 3] = (min(by2, cropped_img_np.shape[0]) - by1)
                ratio = object_np[0, 2] * object_np[0, 3] / float(objects[idx]["w"]*objects[idx]["h"])
                print("ratio: ", ratio)
                if ratio > 0.5 and object_np[0, 2] * object_np[0, 3] > 256:
                    object_np_arr.append(object_np)
                    cropped_img_np = cv2.rectangle(cropped_img_np, (bx1, by1), (bx2, by2), [0, 256, 0], 1)
        target = None
        if len(object_np_arr) == 1:
            target = torch.from_numpy(object_np_arr[0]).float()
        elif len(object_np_arr) > 1:
            target = torch.from_numpy(np.concatenate(object_np_arr, 0)).float()
        else:
            pass
        # 
        cv2.imshow("cropped_img_np", cropped_img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return torch.from_numpy(cropped_img_np.transpose(2, 0, 1)).float(), target

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    trainset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl")
    valset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.pkl")
    while True:
        # print(len(trainset))
        img, label = trainset[random.randint(0, len(trainset) - 1)]
        # 
        print(img)
        print(img.shape)
        if label is not None:
            print(label)
            print(label.shape)
        
        # 
        # print(len(valset))
        img, label = valset[random.randint(0, len(valset) - 1)]
        # 
        print(img)
        print(img.shape)
        if label is not None:
            print(label)
            print(label.shape)
    # 
    assert(0)