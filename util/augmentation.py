import cv2
import numpy as np


def random_scale_translation(img, boxes, im_info, factor=0.2):
    """

    Arguments:
    img -- numpy.ndarray
    boxes -- numpy array of shape (N, 4) N is number of boxes
    factor -- max scale size
    im_info -- dictionary {width:, height:}

    Returns:
    im_data -- numpy.ndarray
    boxes -- numpy array of shape (N, 4)
    """

    w = im_info['width']
    h = im_info['height']

    scale = np.random.uniform() * factor + 1

    # scale img
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # scale boxes
    boxes *= scale

    max_off_x = (scale - 1) * w
    max_off_y = (scale - 1) * h

    offx = int(np.random.uniform() * max_off_x)
    offy = int(np.random.uniform() * max_off_y)

    im_data = scaled_img[offy:(offy+h), offx:(offx+w)]

    # update boxes accordingly
    boxes[:, 0::2] -= offx
    boxes[:, 1::2] -= offy

    # clamp boxes
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, im_data.shape[1] - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, im_data.shape[0] - 1)

    # if flip
    if np.random.randint(2):
        im_data = im_data[:, ::-1]
        boxes[:, 0::2] = (w-1) - boxes[:, 2::-2]

    return im_data, boxes


def convert_color(img, source, dest):
    """
    Convert color

    Arguments:
    img -- numpy.ndarray
    source -- str, original color space
    dest -- str, target color space.

    Returns:
    img -- numpy.ndarray
    """

    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def random_hue(img, rate=.1):
    """
    adjust hue

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    """
    adjust saturation

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    """
    adjust exposure (In fact, this function change V (HSV))

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img


def augment_img(img, boxes, gt_classes, im_info):
    """
    Apply data augmentation.
    1. convert color to HSV
    2. adjust hue(.1), saturation(1.5), exposure(1.5)
    3. convert color to RGB
    4. random scale (up to 20%)
    5. translation (up to 20%)
    6. resize to given input size.

    Arguments:
    img -- PIL.Image object
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    im_info -- dictionary {width:, height:}

    Returns:
    au_img -- numpy array of shape (H, W, 3)
    au_boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    au_gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    """

    img = np.array(img).astype(np.float32)
    boxes = np.copy(boxes).astype(np.float32)
    img = convert_color(img, source='RGB', dest='HSV')

    # adjust color
    img = random_hue(img)
    img = random_saturation(img)
    img = random_exposure(img)

    img = convert_color(img, source='HSV', dest='RGB')

    img, boxes = random_scale_translation(img, boxes, im_info)

    return img, boxes, gt_classes








