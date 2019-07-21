import torch
import torch.nn as nn
import torch.nn.functional as F

def widerface_detection_loss(self, pred, target, object_mask, conf_mask, num_anchors, num_classes):
    """
    pred: predict of network, shape is [Batch, AncherBoxNum, Grid_x * Grid_y, 5 + num_class]
    target: predict of network, shape is [Batch, AncherBoxNum, Grid_x * Grid_y, 5 + num_class]
    conf_mask: [Batch, AncherBoxNum * Grid_x * Grid_y, 1]
    """
    nB   = pred.size(0)                 # number of Batch
    nC   = pred.size(1)                 # number of num_anchors * box_attuributes
    nG_h = pred.size(2)                 # number of Grid_H
    nG_w = pred.size(3)                 # number of Grid_W
    assert(nC == num_anchors * (5 + num_classes))
    # pred is shape of [batch][anchor][grid_y][grid_x][box_attuributes]
    pred = pred.contiguous().view(nB, num_anchors, num_classes + 5, nG_h, nG_w).permute(0, 1, 3, 4, 2) 
    pred = pred.view(-1, np)
    target = target.view(-1, np)
    # 
    px =    torch.sigmoid(pred[:, 0 ])          # Box Center x
    py =    torch.sigmoid(pred[:, 1 ])          # Box Center y
    pw =    pred[:, 2 ]                         # Box Width
    ph =    pred[:, 3 ]                         # Box Height
    pconf = torch.sigmoid(pred[:, 4 ])          # Box Conf
    pcls =  pred[:, 5:]                         # Box Cls pred.
    # 
    tx =    target[:, 0 ]       # Box Center x
    ty =    target[:, 1 ]       # Box Center y
    tw =    target[:, 2 ]       # Box Width
    th =    target[:, 3 ]       # Box Height
    tconf = target[:, 4 ]       # Box Conf
    tcls =  target[:, 5:]       # Box Cls pred.
    # 
    object_mask = object_mask.byte()
    conf_mask = conf_mask.byte()
    conf_mask_true = object_mask
    conf_mask_false = conf_mask - object_mask
    # get xywh loss :
    loss_x = F.mse_loss(px[object_mask], tx[object_mask])
    loss_y = F.mse_loss(py[object_mask], ty[object_mask])
    loss_w = F.mse_loss(pw[object_mask], tw[object_mask])
    loss_h = F.mse_loss(ph[object_mask], th[object_mask])
    # get classification loss :
    loss_cls = F.cross_entropy(pcls[object_mask], torch.argmax(tcls[object_mask], 1))
    # get confidence loss :
    loss_conf = F.binary_cross_entropy(pconf[conf_mask_false], tconf[conf_mask_false]) + F.binary_cross_entropy(pconf[conf_mask_true], tconf[conf_mask_true])
    # 
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls 
    # 
    losses_dict = {
        "total_loss":   loss, 
        "loss_x":       loss_x, 
        "loss_y":       loss_y, 
        "loss_w":       loss_w, 
        "loss_h":       loss_h, 
        "loss_conf":    loss_conf, 
        "loss_cls":     loss_cls
        }
    return losses_dict