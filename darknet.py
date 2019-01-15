# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from util.network import WeightLoader


def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class Darknet19(nn.Module):

    cfg = {
        'layer0': [32],
        'layer1': ['M', 64],
        'layer2': ['M', 128, 64, 128],
        'layer3': ['M', 256, 128, 256],
        'layer4': ['M', 512, 256, 512, 256, 512],
        'layer5': ['M', 1024, 512, 1024, 512, 1024]
    }

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.in_channels = 3

        self.layer0 = self._make_layers(self.cfg['layer0'])
        self.layer1 = self._make_layers(self.cfg['layer1'])
        self.layer2 = self._make_layers(self.cfg['layer2'])
        self.layer3 = self._make_layers(self.cfg['layer3'])
        self.layer4 = self._make_layers(self.cfg['layer4'])
        self.layer5 = self._make_layers(self.cfg['layer5'])

        self.conv = nn.Conv2d(self.in_channels, num_classes, kernel_size=1, stride=1)
        self.avgpool = GlobalAvgPool2d()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv(x)
        x = self.avgpool(x)
        x = self.softmax(x)

        return x

    def _make_layers(self, layer_cfg):
        layers = []

        # set the kernel size of the first conv block = 3
        kernel_size = 3
        for v in layer_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += conv_bn_leaky(self.in_channels, v, kernel_size)
                kernel_size = 1 if kernel_size == 3 else 3
                self.in_channels = v
        return nn.Sequential(*layers)

    # very ugly code !! need to reconstruct
    def load_weights(self, weights_file):
        weights_loader = WeightLoader()
        weights_loader.load(self, weights_file)
        # def load_layer_weights(module, buf, start):
        #     children = list(module.named_children())
        #     for i in range(len(children)):
        #         name, m = children[i]
        #         if isinstance(m, torch.nn.Conv2d):
        #             conv = m
        #             bn = children[i + 1][1]
        #             num_w = conv.weight.data.numel()
        #             num_b = bn.weight.data.numel()
        #             bn.bias.data.copy_(torch.reshape(torch.from_numpy(buf[start:start + num_b]), bn.bias.data.size()))
        #             start += num_b
        #             bn.weight.data.copy_(
        #                 torch.reshape(torch.from_numpy(buf[start:start + num_b]), bn.weight.data.size()))
        #             start += num_b
        #             bn.running_mean.data.copy_(
        #                 torch.reshape(torch.from_numpy(buf[start:start + num_b]), bn.running_mean.data.size()))
        #             start += num_b
        #             bn.running_var.data.copy_(
        #                 torch.reshape(torch.from_numpy(buf[start:start + num_b]), bn.running_var.data.size()))
        #             start += num_b
        #             conv.weight.data.copy_(
        #                 torch.reshape(torch.from_numpy(buf[start:start + num_w]), conv.weight.data.size()))
        #             start += num_w
        #     return start
        #
        # fp = open(weights_file, 'rb')
        # header = np.fromfile(fp, count=4, dtype=np.int32)
        # buf = np.fromfile(fp, dtype=np.float32)
        # fp.close()
        # size = buf.size
        # start = 0
        #
        # for name, m in self.named_children():
        #     if 'layer' in name:
        #         start = load_layer_weights(m, buf, start)
        #     elif name == 'conv':
        #         conv = m
        #         num_w = conv.weight.data.numel()
        #         num_b = conv.bias.data.numel()
        #         conv.bias.data.copy_(torch.reshape(torch.from_numpy(buf[start:start + num_b]), conv.bias.data.size()))
        #         start += num_b
        #         conv.weight.data.copy_(
        #             torch.reshape(torch.from_numpy(buf[start:start + num_w]), conv.weight.data.size()))
        #         start += num_w
        #     elif name == 'avgpool':
        #         pass
        #     elif name == 'softmax':
        #         pass
        #     else:
        #         raise NotImplementedError
        #
        # assert start == size

if __name__ == '__main__':
    im = np.random.randn(1, 3, 224, 224)
    im_variable = Variable(torch.from_numpy(im)).float()
    model = Darknet19()
    out = model(im_variable)
    print(out.size())
    print(model)
