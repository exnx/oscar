# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
from utils import HParams, load_config, print_config, ProgressBar
from utils import hu_moments, resize_maxdim, read_image_url, downsize_shortest_edge
from utils import make_new_dir, Locker, Timer, get_gpu_free_mem
from . import cnn_resnet


class CNNHash(object):
    def __init__(self, cnn_weight, params='', device=None):
        # setup model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Model CNN hashing on %s' % self.device)
        model_dir = os.path.dirname(cnn_weight)
        config = os.path.join(model_dir, 'config.json')
        self.vhp = cnn_resnet.default_hparams()
        load_config(self.vhp, config, False)
        if params:
            self.vhp.parse(params)
        print_config(self.vhp)
        if self.vhp.do_greedy or self.vhp.do_quantise or self.vhp.do_balance:
            print('CNNHash - this model produces binary hash code')
            self.binary = True
        else:
            self.binary = False
        vmodel = cnn_resnet.ResnetModel(self.vhp)
        vmodel.load_pretrained_weight(cnn_weight)
        self.vmodel = vmodel.to(self.device)
        self.vmodel.eval()

    def hash(self, cv2_imgs, return_prehash=False):
        out = self.vmodel.predict_from_cv2_images(cv2_imgs)
        out_bin = out > 0 if self.binary else out 
        if return_prehash:
            return out_bin, out 
        else:
            return out_bin


class ImageNetHash(object):
    def __init__(self, cnn_weight=None, params='', device=None):
        # setup model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Model ImageNet hashing on %s' % self.device)
        model = torchvision.models.resnet50(pretrained=True, progress=False)
        self.vmodel = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.vmodel.eval()
        self.vhp = cnn_resnet.default_hparams()  # for slack notification only
        if params:
            self.vhp.parse(params)
        self.batch_size = self.vhp.batch_size if 'batch_size' in params else 20  # enough for 8GB GPU
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    def preprocess(self, cv2_im):
        im = resize_maxdim(cv2_im, 224).astype(np.float32).transpose(2, 0, 1)[::-1]/255
        im = self.normalizer(torch.tensor(im, dtype=torch.float))
        return im

    def hash(self, cv2_imgs):
        n = len(cv2_imgs)
        out = []
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                start_id, end_id = i, min(n, i+self.batch_size)
                imgs = cv2_imgs[start_id:end_id]
                imgs = [self.preprocess(im) for im in imgs]
                imgs = torch.stack(imgs).to(self.device)
                res = self.vmodel(imgs)
                out.append(res.cpu().numpy())
        out = np.concatenate(out).squeeze()
        return out


class MobileNetHash(object):
    def __init__(self, cnn_weight=None, params='', device=None):
        # setup model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Model MobileNet hashing on %s' % self.device)
        model = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
        self.vmodel = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)  # this output 1280x7x7
        self.vmodel.eval()
        self.vhp = cnn_resnet.default_hparams()  # for slack notification only
        if params:
            self.vhp.parse(params)
        self.batch_size = self.vhp.batch_size if 'batch_size' in params else 20  # enough for 8GB GPU
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    def preprocess(self, cv2_im):
        im = cv2.resize(cv2_im, (224, 224), cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1)[::-1]/255
        im = self.normalizer(torch.tensor(im, dtype=torch.float))
        return im

    def hash(self, cv2_imgs):
        n = len(cv2_imgs)
        out = []
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                start_id, end_id = i, min(n, i+self.batch_size)
                imgs = cv2_imgs[start_id:end_id]
                imgs = [self.preprocess(im) for im in imgs]
                imgs = torch.stack(imgs).to(self.device)
                res = self.vmodel(imgs)
                res = F.adaptive_avg_pool2d(res, 1).reshape(res.shape[0], -1)
                out.append(res.cpu().numpy())
        out = np.concatenate(out).squeeze()
        return out
