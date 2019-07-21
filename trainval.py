import os
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tensorboardX import SummaryWriter
from modules.yolo import FDNet
from utils.loss import YoloLoss
from utils.boxcoder import BoxCoder
from utils.anchor import genAnchor
from modules.initializer import module_weight_init
from modules.dataloader import WiderFaceDataset
from demo import test



if __name__ == "__main__":
    """
    ***********************************************************************************************************************************************
    Add load args parser for hyper-params
    ***********************************************************************************************************************************************
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="runx", help="name of this run")
    parser.add_argument("--batch_size", type=int, default=32, help="how many samples per batch to load")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="base learning rate")
    parser.add_argument("--momentum", type=float, default=9e-1, help="momentum factor in SGD")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 penalty) in SGD")
    parser.add_argument("--jobs", type=int, default=4, help="how many subprocesses to use for data loading")
    parser.add_argument("--gpu", type=int, default=4, help="if use nvidia gpu in training")
    parser.add_argument("--checkpoints_folder", type=str, default="./checkpoints/", help="checkpoints folder path")
    args = parser.parse_args()

    """
    ***********************************************************************************************************************************************
    Add DataLoaders for train and val
    ***********************************************************************************************************************************************
    """
    trainset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl", img_size = 416, grid_size = 13)
    traindataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.pkl")
    valdataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

































    


########################################################################
# load modules
num_classes = 2
grid_size = (13, 13)
grid_area = grid_size[0] * grid_size[1]
anchors = genAnchor(areas=(grid_area, grid_area / 4, grid_area / 16))
# 
net = FDNet(anchors=anchors, num_classes=num_classes)
boxcoder = BoxCoder(anchors=anchors, num_classes=num_classes, grid_size=grid_size)
loss_calculator = YoloLoss()
# 
module_weight_init(net)


########################################################################
# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrs_setpsize, gamma=args.lrs_gamma)


########################################################################
# new a tensorboardx writer
tbxsw = SummaryWriter(log_dir="./checkpoints/"+args.run_name)


########################################################################
# load transforms and dataset




