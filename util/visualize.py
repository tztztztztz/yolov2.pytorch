# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
# --------------------------------------------------------
# Modified by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

font_path = os.path.join(os.path.dirname(__file__), 'arial.ttf')

FONT = ImageFont.truetype(font_path, 20)


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font=FONT, color='black', thickness=2):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)

    # draw.rectangle(
    #     [(left, text_bottom - text_height - 2 * margin), (left + text_width,
    #                                                     text_bottom)],fill=color)

    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill=color,
        font=font)

    return image


def draw_detection_boxes(image, boxes, gt_classes=None, class_names=None):
    """
    Draw bounding boxes via PIL.Image library

    Arguments:
    im_data -- PIL.Image object
    boxes -- numpy array of shape (N, 5) N is number of boxes, (x1, y1, x2, y2, cls_score)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    class_names -- list of string. class names

    Return:
    im_data -- image data with boxes
    """

    num_boxes = boxes.shape[0]
    disp_image = image
    for i in range(num_boxes):
        bbox = tuple(np.round(boxes[i, :4]).astype(np.int64))
        score = boxes[i, 4]
        gt_class_ind = gt_classes[i]
        class_name = class_names[gt_class_ind]
        disp_str = '{}: {:.2f}'.format(class_name, score)
        disp_image = _draw_single_box(disp_image,
                                      bbox[0],
                                      bbox[1],
                                      bbox[2],
                                      bbox[3],
                                      disp_str,
                                      FONT,
                                      color=STANDARD_COLORS[gt_class_ind % NUM_COLORS])
    return disp_image


def plot_boxes(im_data, boxes, gt_classes=None, class_names=None):
    """
    Visualize the bounding boxes of objects in a image

    Arguments:
    im_data -- PIL.Image object or np.ndarray (read from cv2)
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    class_names -- list of string. class names

    Or:
    im_data -- tensor of shape (3, H, W)
    boxes -- tensor
    gt_classes -- tensor

    Return:

    im_data -- image data with boxes
    """
    if isinstance(im_data, torch.Tensor):
        im_data = im_data.permute(1, 2, 0).numpy() * 255
        im_data = im_data.astype(np.uint8)
        boxes = boxes.numpy()
        gt_classes = gt_classes.numpy() if gt_classes is not None else None
    elif isinstance(im_data, PIL.JpegImagePlugin.JpegImageFile):
        im_data = np.copy(np.array(im_data))
    elif isinstance(im_data, np.ndarray):
        im_data = np.copy(np.array(im_data))
    else:
        raise NotImplementedError
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        bbox = tuple(np.round(boxes[i, :]).astype(np.int64))
        cv2.rectangle(im_data, bbox[0:2], bbox[2:4], (0, 205, 0), 2)
        if gt_classes is not None:
            class_name = class_names[gt_classes[i]]
            cv2.putText(im_data, '%s' % class_name, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        2.0, (0, 0, 255), thickness=1)
    return im_data
