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

class WiderFaceDataset(Dataset):
    def __init__(self, pickle_path, img_size = 480, grid_size = 15):
        pickle_load = None
        with open(pickle_path,'rb') as fw:
            pickle_load = pickle.load(fw)
        assert(pickle_load is not None)
        self.data = pickle_load
        pass
    
    def __getitem__(self, index):
        img_path, objects = self.data[index]["image_path"], self.data[index]["objects"]
        image_np = cv2.imread(img_path, 1)
        objects_np = np.zeros((len(objects), 4))
        for idx in range(len(objects)):
            objects_np[idx, 0] = objects[idx]["x1"] + objects[idx]["w"] / 2
            objects_np[idx, 1] = objects[idx]["y1"] + objects[idx]["h"] / 2
            objects_np[idx, 2] = objects[idx]["w"]
            objects_np[idx, 3] = objects[idx]["h"]
        import pdb; pdb.set_trace()
        # random crop

        # rotate

        # to Tensor and return
        return torch.from_numpy(image_np.transpose(2, 0, 1)).float(), torch.from_numpy(objects_np).float()

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    trainset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl")
    print(len(trainset))
    img, label = trainset[len(trainset) - 2]
    # 
    print(img)
    print(img.shape)
    print(label)
    print(label.shape)
    
    # 
    valset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.pkl")
    print(len(valset))
    img, label = valset[len(valset) - 2]
    # 
    print(img)
    print(img.shape)
    print(label)
    print(label.shape)
    # 
    assert(0)