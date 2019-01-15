import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, pred, target, object_mask, conf_mask):
        nB = pred.size(0)
        # 
        px = pred[..., 0 ]          # Box Center x
        py = pred[..., 1 ]          # Box Center y
        pw = pred[..., 2 ]          # Box Width
        ph = pred[..., 3 ]          # Box Height
        pconf = pred[..., 4 ]       # Box Conf
        pcls = pred[..., 5:]        # Box Cls pred.
        # 
        tx = target[..., 0 ]          # Box Center x
        ty = target[..., 1 ]          # Box Center y
        tw = target[..., 2 ]          # Box Width
        th = target[..., 3 ]          # Box Height
        tconf = target[..., 4 ]       # Box Conf
        tcls = target[..., 5:]        # Box Cls pred.
        # 
        object_mask = object_mask.byte()
        conf_mask = conf_mask.byte()
        # Get conf mask where gt and where there is no gt, ignore the area that iou > threshold but not best anchor.
        conf_mask_true = object_mask
        conf_mask_false = conf_mask - object_mask
        # 
        loss_x = self.mse_loss(px[object_mask], tx[object_mask])
        loss_y = self.mse_loss(py[object_mask], ty[object_mask])
        loss_w = self.mse_loss(pw[object_mask], tw[object_mask])
        loss_h = self.mse_loss(ph[object_mask], th[object_mask])
        loss_conf = self.bce_loss(pconf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pconf[conf_mask_true], tconf[conf_mask_true])
        loss_cls = self.ce_loss(pcls[object_mask], torch.argmax(tcls[object_mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls 
        # 
        return loss, {"loss":loss, "loss_x":loss_x, "loss_y":loss_y, "loss_w":loss_w, "loss_h":loss_h, "loss_conf":loss_conf, "loss_cls":loss_cls}

if __name__ == '__main__':

    from utils.boxcoder import BoxCoder
    from utils.anchor import genAnchor
    anchors = genAnchor(areas=(16*9, 4*9, 1*9))
    num_classes = 2
    grid_size = (13, 13)
    bcoder = BoxCoder(anchors = anchors, num_classes = num_classes, grid_size = grid_size)
    boxes = torch.rand((1, 3, 5))
    boxes[:, :, 0] = 1
    print(boxes)
    tx, ty, tw, th, tconf, tcls, mask, conf_mask, fm = bcoder.encode(boxes)
    batchboxes = bcoder.decode(fm)
    print(batchboxes.shape)
    print(batchboxes)

    yololoss = YoloLoss()
    loss, loss_dict = yololoss(fm, fm, mask, conf_mask)
    print(loss)
    print(loss_dict)