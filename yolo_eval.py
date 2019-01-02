# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from config import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv
from util.bbox import box_ious


def yolo_filter_boxes(boxes_pred, conf_pred, classes_pred, confidence_threshold=0.6):
    """
    Filter boxes whose confidence is lower than a given threshold

    Arguments:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4) (x1, y1, x2, y2) predicted boxes
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)
    threshold -- float, threshold used to filter boxes

    Returns:
    filtered_boxes -- tensor of shape (num_positive, 4)
    filtered_conf -- tensor of shape (num_positive, 1)
    filtered_classes -- tensor of shape (num_positive, num_classes)
    """

    if cfg.debug:
        H = 13
        W = 13
        cell_index = 58
        conf_pred = conf_pred.view(H*W, 5, 1)
        class_pred = classes_pred.view(H*W, 5, 20)
        conf_in_cell = conf_pred[cell_index, :, :]
        class_in_cell = class_pred[cell_index, :, :]
        print(conf_in_cell)
        print(class_in_cell)

        conf_pred = conf_pred.view(-1, 1)

    # multiply class scores and objectiveness score
    # use class confidence score
    # TODO: use objectiveness (IOU) score or class confidence score
    max_cls_score, _ = torch.max(classes_pred, dim=-1, keepdim=True)
    cls_conf = conf_pred * max_cls_score

    pos_inds = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = boxes_pred[pos_inds, :]

    filtered_conf = cls_conf[pos_inds, :]

    filtered_classes = classes_pred[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_classes


def yolo_nms(boxes, scores, threshold):
    """
    Apply Non-Maximum-Suppression on boxes according to their scores

    Arguments:
    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    scores -- tensor of shape (N) confidence
    threshold -- float. NMS threshold

    Returns:
    keep -- tensor of shape (None), index of boxes which should be retain.
    """

    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:

        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)

        ious = box_ious(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def generate_prediction_boxes(deltas_pred):
    """
    Apply deltas prediction to pre-defined anchors

    Arguments:
    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)

    Returns:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4)  (x1, y1, x2, y2)
    """

    H = 13 # hard code for now
    W = 13 # hard code for now

    anchors = torch.FloatTensor(cfg.anchors)
    all_anchors_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)

    all_anchors_xywh = deltas_pred.new(*all_anchors_xywh.size()).copy_(all_anchors_xywh)

    # simply use anchors(xxyy) instead of predicted boxes
    boxes_pred = box_transform_inv(all_anchors_xywh, deltas_pred)
    boxes_pred = xywh2xxyy(boxes_pred)

    # boxes_pred = xywh2xxyy(all_anchors_xywh)

    return boxes_pred


def scale_boxes(boxes, im_info):
    """
    scale predicted boxes

    Arguments:
    boxes -- tensor of shape (N, 4) xxyy format
    im_info -- dictionary {width:, height:}

    Returns:
    scaled_boxes -- tensor of shape (N, 4) xxyy format

    """

    h = im_info['height']
    w = im_info['width']

    input_h, input_w = cfg.input_size
    scale_h, scale_w = input_h / h, input_w / w

    # scale the boxes
    boxes *= cfg.strides

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h

    # clamp boxes
    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes


def yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4,):
    """
    Evaluate the yolo output, generate the final predicted boxes

    Arguments:
    yolo_output -- list of tensors (deltas_pred, conf_pred, classes_pred)

    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)

    im_info -- dictionary {w:, h:}

    threshold -- float, threshold used to filter boxes


    Returns:
    detections -- tensor of shape (None, 7) (x1, y1, x2, y2, cls_conf, cls)
    """

    deltas_pred = yolo_output[0]
    conf_pred = yolo_output[1]
    classes_pred = yolo_output[2]

    num_classes = classes_pred.size(1)
    # apply deltas to anchors
    boxes_pred = generate_prediction_boxes(deltas_pred)

    # filter boxes on confidence score
    boxes_pred, conf_pred, classes_pred = yolo_filter_boxes(boxes_pred, conf_pred, classes_pred, conf_threshold)

    # no detection !
    if boxes_pred.size(0) == 0:
        return []

    # scale boxes
    boxes_pred = scale_boxes(boxes_pred, im_info)

    # calculate the predicted class
    class_score, classes = torch.max(classes_pred, dim=-1)

    detections = []

    # apply NMS classwise

    for cls in range(num_classes):
        cls_mask = classes == cls
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_pred_class = boxes_pred[inds, :].view(-1, 4)
        conf_pred_class = conf_pred[inds, :].view(-1, 1)
        classes_class = classes[inds].view(-1, 1)

        nms_keep = yolo_nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep, classes_class_keep.float()]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)










