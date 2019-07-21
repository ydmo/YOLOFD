import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.anchor import genAnchor
from modules.initializer import module_weight_init

# <class YoloLayer(nn.Module)>
class YoloLayer(nn.Module):
    """Some Information about YoloLayer"""
    # <method __init__>
    def __init__(self):
        super(YoloLayer, self).__init__()
    # <method __init__>

    # <method forward>
    def forward(self, x):
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        io = p.clone()  # inference output
        io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
        # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
        io[..., :4] *= self.stride
        if self.nc == 1:
            io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return io.view(bs, -1, 5 + self.nc), p
    # <method forward>
# <class YoloLayer(nn.Module)>

# <Module FDNet/>
class FDNet(nn.Module):
    """Main CNN Network as a module that named Yolo"""
    def __init__(self):
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
        pass

    def forward(self, x):
        for i in range(5):
            x = self.module_dict["""conv_{}""".format(i * 2)](x)
            x = self.module_dict["""maxpool_{}""".format(i * 2 + 1)](x)
        x = self.module_dict["conv_12"](x)
        x = self.module_dict["conv_13"](x)
        x = self.module_dict["conv_14"](x)
        x = self.module_dict["conv_15"](x)
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