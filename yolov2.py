# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import config as cfg
from darknet import Darknet19
from darknet import conv_bn_leaky
from loss import build_target, yolo_loss


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class Yolov2(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        darknet19 = Darknet19()

        if weights_file:
            print('load pretrained weight from {}'.format(weights_file))
            darknet19.load_weights(weights_file)
            print('pretrained weight loaded!')

        # darknet backbone
        self.conv1 = nn.Sequential(darknet19.layer0, darknet19.layer1,
                                   darknet19.layer2, darknet19.layer3, darknet19.layer4)

        self.conv2 = darknet19.layer5

        # detection layers
        self.conv3 = nn.Sequential(conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
                                   conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.conv4 = nn.Sequential(conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

        self.reorg = ReorgLayer()

    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        x = self.conv1(x)
        shortcut = self.reorg(self.downsampler(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([shortcut, x], dim=1)
        out = self.conv4(x)

        if cfg.debug:
            print('check output', out.view(-1)[0:10])

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

if __name__ == '__main__':
    model = Yolov2()
    im = np.random.randn(1, 3, 416, 416)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)
    delta_pred, conf_pred, class_pred = out
    print('delta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())



