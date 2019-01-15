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
from modules.initializer import weights_init_normal

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")

dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

model = TinyYolo416()
model.apply(weights_init_normal)
if cuda:
    model = model.cuda()
model.train()

writer = SummaryWriter()