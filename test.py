import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
from yolov2 import Yolov2
from dataset.factory import get_imdb
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07trainval', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=1, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = 416, 416
    im_data = img.resize((H, W), Image.BILINEAR)

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float()

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def test():
    args = parse_args()
    print('Called with args:')
    print(args)

    # prepare dataset

    if args.dataset == 'voc07train':
        args.imdb_name = 'voc_2007_train'
        args.imdbval_name = 'voc_2007_train'

    elif args.dataset == 'voc07trainval':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'

    else:
        raise NotImplementedError

    val_imdb = get_imdb(args.imdbval_name)

    # load model
    model = Yolov2()
    model_path = os.path.join(args.output_dir, args.model_name+'.pth')
    print('loading model from {}'.format(model_path))
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')

    dataset_size = len(val_imdb.image_index)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

    det_file = os.path.join(args.output_dir, 'detections.pkl')

    for i in range(dataset_size):
        image_path = val_imdb.image_path_at(i)
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)

        if len(detections) > 0:
            for cls in range(val_imdb.num_classes):
                inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                if inds.numel() > 0:
                    cls_det = detections[inds, :5]
                    all_boxes[cls][i] = cls_det.cpu().numpy()

        toc = time.time()
        cost_time = toc - tic
        print('im detect [{}/{}], cost time {:4f}, FPS: {}'.format(
            i+1, dataset_size, toc-tic, int(1 / cost_time)))

        if args.vis:
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            # im2show = plot_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
            im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
            plt.figure()
            plt.imshow(im2show)
            plt.show()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)

if __name__ == '__main__':
    test()















