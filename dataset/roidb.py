"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from config import config as cfg
from util.augmentation import augment_img


class RoiDataset(Dataset):
    def __init__(self, imdb, train=True):
        super(RoiDataset, self).__init__()
        self._imdb = imdb
        self._roidb = imdb.roidb
        self.train = train
        self._image_paths = [self._imdb.image_path_at(i) for i in range(len(self._roidb))]

    def roi_at(self, i):
        image_path = self._image_paths[i]
        im_data = Image.open(image_path)
        boxes = self._roidb[i]['boxes']
        gt_classes = self._roidb[i]['gt_classes']

        return im_data, boxes, gt_classes

    def __getitem__(self, i):
        im_data, boxes, gt_classes = self.roi_at(i)
        # w, h
        im_info = torch.FloatTensor([im_data.size[0], im_data.size[1]])

        if self.train:
            im_data, boxes, gt_classes = augment_img(im_data, boxes, gt_classes)

            w, h = im_data.size[0], im_data.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize image
            input_h, input_w = cfg.input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            boxes = torch.from_numpy(boxes)
            gt_classes = torch.from_numpy(gt_classes)
            num_obj = torch.Tensor([boxes.size(0)]).long()
            return im_data_resize, boxes, gt_classes, num_obj

        else:
            input_h, input_w = cfg.test_input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            return im_data_resize, im_info

    def __len__(self):
        return len(self._roidb)

    def __add__(self, other):
        self._roidb = self._roidb + other._roidb
        self._image_paths = self._image_paths + other._image_paths
        return self


def detection_collate(batch):
    """
    Collate data of different batch, it is because the boxes and gt_classes have changeable length.
    This function will pad the boxes and gt_classes with zero.

    Arguments:
    batch -- list of tuple (im, boxes, gt_classes)

    im_data -- tensor of shape (3, H, W)
    boxes -- tensor of shape (N, 4)
    gt_classes -- tensor of shape (N)
    num_obj -- tensor of shape (1)

    Returns:

    tuple
    1) tensor of shape (batch_size, 3, H, W)
    2) tensor of shape (batch_size, N, 4)
    3) tensor of shape (batch_size, N)
    4) tensor of shape (batch_size, 1)

    """

    # kind of hack, this will break down a list of tuple into
    # individual list
    bsize = len(batch)
    im_data, boxes, gt_classes, num_obj = zip(*batch)
    max_num_obj = max([x.item() for x in num_obj])
    padded_boxes = torch.zeros((bsize, max_num_obj, 4))
    padded_classes = torch.zeros((bsize, max_num_obj,))

    for i in range(bsize):
        padded_boxes[i, :num_obj[i], :] = boxes[i]
        padded_classes[i, :num_obj[i]] = gt_classes[i]

    return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)


class TinyRoiDataset(RoiDataset):
    def __init__(self, imdb, num_roi):
        super(TinyRoiDataset, self).__init__(imdb)
        self._roidb = self._roidb[:num_roi]







