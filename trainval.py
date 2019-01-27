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


########################################################################
# load args parser for hyper-params
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="runx", help="name of this run")
parser.add_argument("--batch_size", type=int, default=32, help="how many samples per batch to load")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-1, help="base learning rate")
parser.add_argument("--momentum", type=float, default=9e-1, help="momentum factor in SGD")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 penalty) in SGD")
parser.add_argument("--lrs_setpsize", type=int, default=100, help="step size used in step-learning-rate scheduler")
parser.add_argument("--lrs_gamma", type=float, default=0.1, help="factor of learning rate decay used in step-learning-rate scheduler")
parser.add_argument("--num_workers", type=int, default=4, help="how many subprocesses to use for data loading")
parser.add_argument("--cuda", help="if use nvidia gpu in training", action="store_true")
parser.add_argument("--check_point", type=str, default="./checkpoints/runx/cp.pth", help="checkpoint path")
group = parser.add_mutually_exclusive_group()
parser.add_argument("--train", help="if train", action="store_true")
parser.add_argument("--test", help="if test", action="store_true")
args = parser.parse_args()


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
trainset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_train_bbx_gt.pkl", img_size = 416, grid_size = 13)
traindataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valset = WiderFaceDataset(pickle_path="/home/gtx1060/Documents/DataSets/wider_face/wider_face_split/wider_face_val_bbx_gt.pkl")
valdataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

def train(epoch = -1, summarywriter = None):    
    for i, data in enumerate(traindataloader, 0):
        images, labels = data
        # check data ...
        # for idx in range(images.size(0)):
        #     label = labels[idx]
        #     image_np = images[idx].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        #     for l in range(label.size(0)):
        #         if label[l][4] > 0:
        #             cy = label[l][0] * (image_np.shape[0] - 1)
        #             cx = label[l][1] * (image_np.shape[1] - 1)
        #             h = label[l][2] * (image_np.shape[0] - 1)
        #             w = label[l][3] * (image_np.shape[1] - 1)
        #             image_np = cv2.rectangle(image_np, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 1)
        #     cv2.imshow("image_np", image_np)
        #     cv2.waitKey(0)
        # 
        pred = net(images)
        import pdb; pdb.set_trace()
        # losses = loss_calculator()

            
    pass


def val(epoch = -1, summarywriter = None):
    for i, data in enumerate(valdataloader, 0):
        images, labels = data
    pass


if __name__ == "__main__":
    
    with torch.no_grad():
        tbxsw.add_graph(model=net, input_to_model=torch.rand((1, 3, 416, 416)))

    if args.cuda:
        net = net.cuda()

    if args.train:    
        print("Training ...")
        for epoch in range(args.epochs):
            scheduler.step(epoch)
            train(epoch, tbxsw)
            val(epoch, tbxsw)
        print("Train done.")

    if args.test:
        print("Testing ...")
        test(net)
        print("Test done.")
        pass

    tbxsw.close()

    print("Exit.")