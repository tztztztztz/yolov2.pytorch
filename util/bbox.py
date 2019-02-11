# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box


def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors








