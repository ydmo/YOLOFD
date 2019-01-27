import torch
import torch.nn as nn
import torch.nn.functional as F

gt_box = torch.FloatTensor([(0, 0, 3, 4) for _ in range(5)])
print(gt_box)

def test(net, video_path=None):
    pass