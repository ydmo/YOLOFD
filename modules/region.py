import torch
import torch.nn as nn
import torch.nn.functional as F

# <Class Region/>
class Region(torch.nn.Module):
    
    # <Region.__init__/>
    def __init__(self, anchors, num_classes):
        super(Region, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.anchors = torch.FloatTensor([(a_w, a_h) for a_w, a_h in anchors])
        self.anchor_w = self.anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
    # </Region.__init__>

    # <Region.forward/>
    def forward(self, x):
        # get feature size:
        nB   = x.size(0)                 # number of Batch
        nG_h = x.size(2)                 # number of Grid_H
        nG_w = x.size(3)                 # number of Grid_W
        # get pred:
        pred = x.view(nB, self.num_anchors, self.num_classes + 5, nG_h, nG_w).permute(0, 1, 3, 4, 2).contiguous() # [batch][anchor][grid_y][grid_x][box_attuributes]
        # 
        # sigmoid the x, y, conf and cls_pred
        pred[..., 0 ] = torch.sigmoid(pred[..., 0])                     # Box_Center_X - grid_X
        pred[..., 1 ] = torch.sigmoid(pred[..., 1])                     # Box_Center_Y - grid_Y
        # pred[..., 2 ] = pred[..., 2]                                  # log(e, Box_W / Anchor_W)
        # pred[..., 3 ] = pred[..., 3]                                  # log(e, Box_H / Anchor_H)
        pred[..., 4 ] = torch.sigmoid(pred[..., 4])                     # Box Conf
        # pred[..., 5:] = torch.sigmoid(pred[..., 5:])                    # Box Cls pred.
        #  
        return pred
    # </Region.forward>
    
# </Class Region>

if __name__ == '__main__':
    from utils.anchor import genAnchor
    anchors = genAnchor(areas=(16*9, 4*9, 1*9))
    num_anchors = len(anchors)
    num_classes = 2
    yolo = Region(anchors, num_classes)    
    inp = torch.zeros(1, num_anchors * (num_classes + 5), 13, 13)
    print(inp)
    print(inp.shape)
    outp = yolo(inp)
    print(outp)
    print(outp.shape)