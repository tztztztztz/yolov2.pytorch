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
from dataset.roidb import RoiDataset
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
from config import config as cfg


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=2, type=int)
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
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def test():
    args = parse_args()
    args.conf_thresh = 0.005
    args.nms_thresh = 0.45
    if args.vis:
        args.conf_thresh = 0.5
    print('Called with args:')
    print(args)

    # prepare dataset

    if args.dataset == 'voc07trainval':
        args.imdbval_name = 'voc_2007_trainval'

    elif args.dataset == 'voc07test':
        args.imdbval_name = 'voc_2007_test'

    else:
        raise NotImplementedError

    val_imdb = get_imdb(args.imdbval_name)

    val_dataset = RoiDataset(val_imdb, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    model = Yolov2()
    # weight_loader = WeightLoader()
    # weight_loader.load(model, 'yolo-voc.weights')
    # print('loaded')

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

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(val_dataloader):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh)
                print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                if args.vis:
                    img = Image.open(val_imdb.image_path_at(img_id))
                    if len(detections) == 0:
                        continue
                    det_boxes = detections[:, :5].cpu().numpy()
                    det_classes = detections[:, -1].long().cpu().numpy()
                    im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
                    plt.figure()
                    plt.imshow(im2show)
                    plt.show()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)

if __name__ == '__main__':
    test()















