# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import os
import os.path as osp

root_dir = osp.join(osp.dirname(__file__), '..')
data_dir = osp.join(root_dir, 'data')


class imdb(object):
    def __init__(self, name):
        self._name = name
        self._classes = []
        self._image_index = []
        self._roidb = None
        self._roidb_handler = self.default_roidb

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_classes(self):
        return len(self._classes)


    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val


    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    def default_roidb(self):
        raise NotImplementedError


    @property
    def num_images(self):
        return len(self._image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):

        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(data_dir, 'cache'))
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path












