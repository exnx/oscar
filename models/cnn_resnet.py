# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from .torch_layers import FCSeries, GreedyHashLoss
from utils import HParams, Timer, resize_maxdim
from copy import deepcopy


def default_hparams():
    hparams = HParams(
        name='ResnetModel',  # this should match the class name below
        # model params
        d_model=128,  # embedding layer
        dropout=0.1,  # dropout rate
        freeze_bn=False,  # freeze batch norm

        do_triplet=True,
        triplet_margin=1.0,
        triplet_metric='l2',  # ['l2', 'cosine']
        do_buffer=False,  # whether there is buffer layers between bottleneck & triplet/simclr loss
        buffer_dim=[128],  # dimension of buffer layers
        buffer_relu_last=False,

        do_simclr=False,  # use NTXent in SimCLR mode
        simclr_version=1,  # version of simclr [1,2]
        simclr_temperature=0.8,

        do_greedy=False,  # do greedy binarisation hashing with intergrated sign() fn
        greedy_weight=0.01,

        do_quantise=False,  # binary quantization loss
        quantise_weight=1.0,

        do_balance=False,  # balance loss bits 
        balance_weight=0.01,

        # training params
        dataset='MSCOCO',  # MSCOCO, PSBattles
        train_all_layers=True,  # if False train the last layer only
        nepochs=10,
        batch_size=16,
        npos=2,  # number of positives in a batch
        optimizer='Adam',
        lr=0.0001,
        lr_steps=[1.1],  # step decay for lr; value > 1 means lr is fixed
        lr_sf=0.1,  # scale factor for learning rate when condition is met
        neg_random_rate=0.,  # used in dataloader for psbattles, random sample neg from org
        resume=True,
        save_every=5,  # save model every x epoch
        report_every=100,  # tensorboard report every x iterations
        val_every=3,  # validate every x epoch
        checkpoint_path='./',  # path to save/restore checkpoint
        slack_token='slack_token.txt'  # token for slack messenger
    )
    return hparams


class ResnetModel(nn.Module):

    def __init__(self, hps):
        self.hps = hps
        super(ResnetModel, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True, progress=False)
        if not hps.train_all_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        # change last layer
        self.model.fc = nn.Linear(self.model.fc.in_features, hps.d_model)

        # train attributes
        self.device = None
        self.optimizer = None
        self.writer = None
        # loss
        if hps.do_greedy:
            self.binariser = GreedyHashLoss()
        buffer_dim = hps.buffer_dim if hps.do_buffer else []
        self.buffer_layer = FCSeries(hps.d_model, buffer_dim, relu_last=hps.buffer_relu_last)

        # eval attributes
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        # param initialization
        self.init_weights()

    def init_weights(self):
        # initialize weights for final layer only
        initrange = 0.1
        self.model.fc.bias.data.zero_()
        self.model.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        output = {}
        embed_float = self.model(x)  # fingerprint
        if self.hps.do_greedy:
            output['embedding'], output['greedy_loss'] = self.binariser(embed_float)
        elif self.hps.do_quantise or self.hps.do_balance:
            output['embedding'] = embed_float.tanh()
        else:
            output['embedding'] = embed_float
        output['regress'] = self.buffer_layer(output['embedding'])  # for loss
        return output

    def predict_from_cv2_images(self, img_lst):
        device = next(self.parameters()).device
        # preprocess
        num_images = len(img_lst)
        pre_x = [resize_maxdim(im, 224).astype(np.float32).transpose(2, 0, 1)[::-1]/255 for im in img_lst]
        pre_x = [self.normalizer(torch.tensor(x_, dtype=torch.float)) for x_ in pre_x]
        out = []
        with torch.no_grad():
            for id_ in range(0, num_images, self.hps.batch_size):
                start_, end_ = id_, min(id_ + self.hps.batch_size, num_images)
                batch = pre_x[start_:end_]
                batch = torch.stack(batch).to(device)
                pred = self.__call__(batch)['embedding'].cpu().numpy()
                out.append(pred)
        out = np.concatenate(out)
        return out

    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # not updating running mean/var
            module.weight.requires_grad = False  # not updating weight/bis, or alpha/beta in the paper
            module.bias.requires_grad = False

    def train(self, mode=True):
        """
        override train fn with freezing batchnorm
        """
        super().train(mode)
        if self.hps.freeze_bn:
            self.model.apply(self.freeze_bn)  # freeze running mean/var in bn layers in cnn_model only

    def preprocess(self, x, y):
        """
        preprocess data, model dependent
        """
        if self.device is None:  # check if device is set
            self.device = next(self.parameters()).device  # current device
        x = [x_.astype(np.float32).transpose(2, 0, 1)[::-1]/255. for x_ in x]
        x = [self.normalizer(torch.tensor(x_, dtype=torch.float32)) for x_ in x]
        x = torch.stack(x).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        return x, y

    def get_optimizer(self):
        if self.hps.optimizer == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hps.lr,
                                     betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-4)
        elif self.hps.optimizer == 'SGD':
            optim = torch.optim.SGD(self.parameters(), lr=self.hps.lr, momentum=0.9,
                                    weight_decay=5e-4)
        return optim

    def load_pretrained_weight(self, pretrain_path):
        device = next(self.parameters()).device  # load to current device
        print('Loading pretrained model %s.' % pretrain_path)
        pretrained_state = torch.load(pretrain_path, map_location=device)
        if 'model_state_dict' in pretrained_state:
            print('This pretrained model is a checkpoint, loading model_state_dict only.')
            pretrained_state = pretrained_state['model_state_dict']
        model_state = self.state_dict()
        matched_keys, not_matched_keys = [], []
        for k,v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                matched_keys.append(k)
            else:
                not_matched_keys.append(k)
        if len(not_matched_keys):
            print('[%s] The following keys are not loaded: %s' % (self.hps.name, not_matched_keys))
            pretrained_state = {k: pretrained_state[k] for k in matched_keys}
        # pretrained_state = { k:v for k,v in pretrained_state.items() if k in \
        #                     model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        self.load_state_dict(model_state)

    def load_checkpoint(self, checkpoint_path):
        device = next(self.parameters()).device  # load to current device
        print('Resuming from %s.' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # return the rest
        excl_keys = ['model_state_dict', 'optimizer_state_dict']
        out = {key: checkpoint[key] for key in checkpoint if key not in excl_keys}
        return out

    def save_checkpoint(self, checkpoint_path, save_optimizer=True, **kwargs):
        print('Saving checkpoint at %s' % checkpoint_path)
        checkpoint = {'model_state_dict': self.state_dict()}
        if save_optimizer:
            checkpoint.update(optimizer_state_dict=self.optimizer.state_dict())
            checkpoint.update(lr_scheduler_state_dict=self.lr_scheduler.state_dict())
        checkpoint.update(**kwargs)
        torch.save(checkpoint, checkpoint_path)