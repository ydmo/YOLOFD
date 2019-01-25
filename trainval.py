import os
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from modules.yolo import TinyYolo416
from modules.initializer import module_weight_init


########################################################################
# load args parser
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="runx", help="how many samples per batch to load")
parser.add_argument("--batch_size", type=int, default=32, help="how many samples per batch to load")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-1, help="base learning rate")
parser.add_argument("--lrs_setpsize", type=int, default=100, help="step size used in step-learning-rate scheduler")
parser.add_argument("--lrs_gamma", type=float, default=0.1, help="factor of learning rate decay used in step-learning-rate scheduler")
parser.add_argument("--num_workes", type=int, default=4, help="how many subprocesses to use for data loading")
parser.add_argument("--cuda", help="if use nvidia gpu in training", action="store_true")
parser.add_argument("--check_point", type=str, default="./checkpoints/runx/cp.pth", help="checkpoint path")
group = parser.add_mutually_exclusive_group()
parser.add_argument("--train", help="if train", action="store_true")
parser.add_argument("--test", help="if test", action="store_true")
args = parser.parse_args()


########################################################################
# load module
net = TinyYolo416()
module_weight_init(net)


########################################################################
# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrs_setpsize, gamma=args.lrs_gamma)


########################################################################
# new a tensorboardx writer
tbxsw = SummaryWriter(log_dir="./checkpoints/"+args.run_name)


########################################################################
# load transforms and dataset
# ...


def train():
    pass


def test():
    pass


if __name__ == "__main__":
    
    with torch.no_grad():
        tbxsw.add_graph(model=net, input_to_model=torch.rand((1, 3, 416, 416)))

    if args.cuda:
        net = net.cuda()

    if args.train:    
        print("Training ...")
        train()

    if args.test:
        print("Testing ...")
        test()
        pass

    tbxsw.close()

    import pdb; pdb.set_trace()