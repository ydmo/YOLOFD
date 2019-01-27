import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.region import Region
from utils.anchor import genAnchor
from modules.initializer import module_weight_init

# <Module FDNet/>
class FDNet(nn.Module):
    """Main CNN Network as a module that named Yolo"""
    def __init__(self, anchors, num_classes):
        super(FDNet, self).__init__()
        # 
        self.module_dict = nn.ModuleDict()
        # 
        self.module_dict['conv_0'] = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_1'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # 
        self.module_dict['conv_2'] = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_3'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)            
        # 
        self.module_dict['conv_4'] = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_5'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)            
        # 
        self.module_dict['conv_6'] = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_7'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)                    
        # 
        self.module_dict['conv_8'] = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_9'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)                    
        # 
        self.module_dict['conv_10'] = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['maxpool_11'] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)                    
        # 
        self.module_dict['conv_12'] = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['conv_13'] = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['conv_14'] = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        # 
        self.module_dict['conv_15'] = nn.Conv2d(in_channels=128, out_channels=7*9, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        # 
        self.module_dict['region_16'] = Region(anchors=anchors, num_classes=num_classes)
        # 
        pass

    def forward(self, x):
        for i in range(5):
            x = self.module_dict["""conv_{}""".format(i * 2)](x)
            x = self.module_dict["""maxpool_{}""".format(i * 2 + 1)](x)
        x = self.module_dict["conv_12"](x)
        x = self.module_dict["conv_13"](x)
        x = self.module_dict["conv_14"](x)
        x = self.module_dict["conv_15"](x)
        x = self.module_dict["region_16"](x)
        return x
# </Module FDNet>

if __name__ == '__main__':
    yolo = FDNet()
    module_weight_init(yolo)
    yolo.eval()
    inp = torch.rand(1, 3, 416, 416)
    print(inp.shape)
    out = yolo(inp)
    print(out)
    print(out.shape)