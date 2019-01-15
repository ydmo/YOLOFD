import torch
import torch.nn as nn
import torch.nn.functional as F

# <bbox_iou/>
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # 
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # 
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    # 
    return iou
# </bbox_iou>

# <nms/>
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output
# </nms>

# <Class BoxCoder>
class BoxCoder:
    """
    anchors: boxe sizes for regression
    num_classes: number of classes
    grid_size: (grid_h, grid_w)
    """
    def __init__(self, anchors, num_classes, grid_size, ignore_thres = 0.5):
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors = torch.FloatTensor([(a_w, a_h) for a_w, a_h in anchors])
        self.anchor_w = self.anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        self.ignore_thres = ignore_thres

    """
    boxes is a tensor that storge batches of targets
    """
    def encode(self, boxes):
        nB = boxes.size(0)
        nT = boxes.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        nG_h = self.grid_size[0]
        nG_w = self.grid_size[1]
        # 
        tx =    torch.zeros(nB, nA, nG_h, nG_w, 1)
        ty =    torch.zeros(nB, nA, nG_h, nG_w, 1)
        tw =    torch.zeros(nB, nA, nG_h, nG_w, 1)
        th =    torch.zeros(nB, nA, nG_h, nG_w, 1)
        tconf = torch.zeros(nB, nA, nG_h, nG_w, 1)
        tcls =  torch.zeros(nB, nA, nG_h, nG_w, nC)
        # 
        mask =  torch.zeros(nB, nA, nG_h, nG_w)
        conf_mask = torch.ones(nB, nA, nG_h, nG_w)
        # 
        for b in range(nB):
            for t in range(boxes.shape[1]): # "t" is target index
                gx = boxes[b, t, 1] * nG_w # Center x of the ground turth box
                gy = boxes[b, t, 2] * nG_h # Center y of the ground turth box
                gw = boxes[b, t, 3] * nG_w # Width of the ground turth box
                gh = boxes[b, t, 4] * nG_h # Height of the ground turth box
                # get grid position(gj, gi) that the target box is belong to.
                gj = int(gy)
                gi = int(gx)
                # 
                gt_box = torch.FloatTensor([(0, 0, gw, gh) for _ in range(self.num_anchors)]) # gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                anchor_shapes = torch.cat((torch.zeros(self.num_anchors, 2), self.anchors), 1)
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                conf_mask[b, anch_ious > self.ignore_thres, gj, gi] = 0 # Where the overlap is larger than threshold set mask to zero (ignore)
                # 
                best_n = torch.argmax(anch_ious)
                # 
                mask[b, best_n, gj, gi] = 1
                conf_mask[b, best_n, gj, gi] = 1
                # 
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = torch.log(gw / self.anchors[best_n][0] + 1e-9)
                th[b, best_n, gj, gi] = torch.log(gh / self.anchors[best_n][1] + 1e-9)
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(boxes[b, t, 0])] = 1
                # 
                # gt_box = torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0)

        fm = torch.cat((tx, ty, tw, th, tconf, tcls), 4)
        
        return tx, ty, tw, th, tconf, tcls, mask, conf_mask, fm
    
    def decode(self, fm):
        nB   = fm.size(0)                 # number of Batch
        nG_h = fm.size(2)                 # number of Grid_H
        nG_w = fm.size(3)                 # number of Grid_W
        grid_x = torch.arange(nG_w).float().repeat(nG_h, 1).view([1, 1, nG_h, nG_w])
        grid_y = torch.arange(nG_h).float().repeat(nG_w, 1).t().view([1, 1, nG_h, nG_w])
        # feature map decode to boxes.
        fm0 = torch.zeros((fm.size(0), fm.size(1), fm.size(2), fm.size(3), 7))
        fm0[..., 0 ] = (fm[..., 0 ] + grid_x) / nG_w                                # Object Box_Center_X
        fm0[..., 1 ] = (fm[..., 1 ] + grid_y) / nG_h                                # Object Box_Center_Y
        fm0[..., 2 ] = torch.exp(fm[..., 2]) * self.anchor_w / nG_w                 # Object Box_W
        fm0[..., 3 ] = torch.exp(fm[..., 3]) * self.anchor_h / nG_h                 # Object Box_H
        fm0[..., 4 ] = fm[..., 4 ]                                                  # Object Conf
        fm0[..., 5 ] = torch.argmax(fm[..., 5:], dim=4)                             # Object Type
        fm0[..., 6 ] = F.softmax(fm[..., 5:], 4).max()                              # Object Type Conf
        #
        fm0 = fm0.view([nB, -1, fm0.size(4)])
        batchboxes = []
        for batch in range(nB):
            boxes = []
            for idx in range(fm0.size(1)):
                if fm0[batch, idx, 4] > self.ignore_thres:
                    boxes.append( fm0[batch, idx, :].unsqueeze(0).unsqueeze(0) )
            boxes = torch.cat(boxes, 1)
            batchboxes.append(boxes)
        batchboxes = torch.cat(batchboxes, 0)
        # 
        return batchboxes
# <Class BoxCoder>

if __name__ == '__main__':
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
