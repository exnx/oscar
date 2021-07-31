# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
This demo contains inference code only, the train part has been stripped off.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import numpy as np
from copy import deepcopy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .torch_layers import SelfAttn, FCSeries
from .torch_layers import ObjectClassifier, GreedyHashLoss
from utils import HParams, load_config, print_config, Timer, ExThread, hu_moments
from . import object_detection as objdet
from . import hash_end2end
from . import cnn_resnet



def default_hparams(model='TransformerModel'):
    """
    default base hyper-params for generic models, used in both transformer and GCN
    """
    hps_base = HParams(
        name='GenericModel',  # this should match the class name below
        # model base params
        d_embed=256,  # embedding dimension
        embed_activation='none',  # activation after embed layer
        d_vis=256,  # projected dim of visual feature, always enabled
        inc_geo=True,  # include geometry feature
        d_geo=64,  # projected dim of geo feature
        inc_rela=True,  # include rela feature
        d_rela=256,  # projected dim of rela feature
        inc_shape=False,  # include moment feature of object masks
        d_shape=64,
        inc_sem=False,
        do_full_proj=True,
        proj_dropout=0.4,
        proj_leaky_relu=False,
        inc_cnn=False,  # include cnn model on whole image
        cnn_name='CNNHash',  # name of cnn model
        cnn_weight='/src/weight/object/best.pt',
        cnn_size=224,  # image size to cnn model
        cnn_train_all=True,
        cnn_double_fc=False,
        cnn_freeze_bn=False,

        # loss settings
        tendrill_loss=False,  # also apply regress loss to all branches prior concatenation
        tendrill_weight=0.3,

        do_classify=False,  # enable classification layer?, need number of classes
        class_num=10,  # need update per dataset
        class_weight=1.0,  # weight of classification loss

        do_triplet=True,  # enable triplet
        triplet_margin=1.0,
        triplet_metric='l2',  # ['l2', 'cosine']
        neg_push=0.0,  # custom triplet loss, explicitly push neg further
        regress_weight=1.0,
        do_buffer=True,  # whether there is buffer layers between bottleneck & triplet/simclr loss
        buffer_dim=[128],  # dimension of buffer layers
        buffer_relu_last=True,  # when buffer is True, whether relu is appended to the last FC layer

        do_contrast=False,  # contrastive loss
        contrast_margin=1.0,
        contrast_metric='l2',

        do_simclr=False,  # use NTXent in SimCLR mode
        simclr_temperature=0.8,

        do_logistic=False,  # enable logistic classification [real/photoshop]
        logistic_weight=1.0,

        do_recon=False,
        do_attn_balance=False,  # balacing attention weight
        attn_balance_weight=1.0,

        do_greedy=False,  # do greedy binarisation hashing with intergrated sign() fn
        greedy_weight=0.01,

        do_quantise=False,  # binary quantization loss
        quantise_weight=0.0001,

        do_balance=False,  # balance loss bits 
        balance_weight=0.1,

        # training params
        dataset='PSBattles',  # MSCOCO, PSBattles
        data_invert=True,  # if True, data is transposed to (Seqlen, Bsz, dim)
        seq_len=16,  # sequence length
        obj_detection_bsz=4,  # batch size of maskrcnn
        obj_extraction_bsz=8,  # batchsize of object feature extractor
        obj_big_box=1,  # 1 if inc whole image, 2 if background, 0 if not including any extra object in detection
        obj_feat_level=0,  # maskrcnn features level: 0-external cnn; 1-middle layer, 2-last layer
        obj_shuffle=False,  # whether to shuffle order of objects
        obj_feat_extractor='CNNHash',  # model to extract object features, apply only when obj_feat_level=0
        # model weight to extract object features, some models can directly download from model zoo:
        obj_feat_weight='/src/weight/object/best.pt',
        weight_init='gaussian',  # weight initialization
        nepochs=10,
        batch_size=10,
        npos=2,  # number of positives in a batch
        optimizer='Adam',
        grad_clip=0.,  # gradient clipping, 0 if no clip
        grad_iters=1,  # number of iterations before loss and gradient are computed and network params are updated
        lr_scheduler='multistep',  # step, multistep
        lr=0.0001,  # initial lr
        lr_step_size=1,  # for step lr, measured in epochs
        lr_steps=[0.6],  # step decay for multistep lr; value > 1 means lr is fixed
        lr_sf=0.1,  # scale factor for learning rate when condition is met
        resume=True,
        save_every=3,  # save model every x epoch
        report_every=100,  # tensorboard report every x iterations
        val_every=3,  # validate every x epoch
        checkpoint_path='./',  # path to save/restore checkpoint
        slack_token='slack_token.txt',  # token for slack messenger
        cuda_all=False,
    )
    if model == 'TransformerModel':
        hps_model = HParams(
            name=model,
            d_model_vis=512,  # model internal dim for visual
            d_model_rela=512,  # model internal dim for rela
            nhead=8,  # attention heads
            nhid=1024,  # fastforward dimension
            nlayers=6,  # num layers
            model_dropout=0.1,
            attn_mask=True,
            use_tform=True,
            tform_branch='both',  # ['vis', 'rela', 'both'] tform models created for visual, rela or both branches
            )

    return hps_base + hps_model


def convert_torch_dict(dictionary):
    """
    convert dictionary of torch tensors to normal dict
    NOTE: only work on scalar values, to convert array use .cpu().numpy() instead
    """
    out = {}
    for key, val in dictionary.items():
        out[key] = 0 if val is None else val.item()
    return out


def build_projection_layer(d_input, d_output, leaky_relu=False, dropout=0.4, do_full_proj=True):
    if do_full_proj:
        return nn.Sequential(nn.Linear(d_input, d_output),
            nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True),
            nn.Dropout(dropout))
    else:
        return nn.Linear(d_input, d_output)


def get_cnn_model(name, weight, train_all=True, double_fc=False):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    if name == 'CNNHash':
        pretrained_state = torch.load(weight)
        out_dim = pretrained_state['model.fc.bias'].size(0)
        model = torchvision.models.resnet50(pretrained=True, progress=False)
        if not train_all:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, out_dim)
        state_dict = model.state_dict()
        matched_keys = []
        for k, v in state_dict.items():
            matched_key = 'model.'+k
            if matched_key in pretrained_state and v.size() == pretrained_state[matched_key].size():
                matched_keys.append(k)
        print('Full Image branch using CNN model: loading %d/%d matched keys.' % \
            (len(matched_keys), len(pretrained_state.keys())))
        pretrained_state = {k: pretrained_state['model.' + k] for k in matched_keys}
        state_dict.update(pretrained_state)
        model.load_state_dict(state_dict)
        
    elif name == 'ImageNet':
        out_dim = 128
        model = torchvision.models.resnet50(pretrained=True, progress=False)
        if not train_all:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, out_dim)
        print('Full Image branch using ImageNet Resnet50 pretrain model.')
    elif name == 'MobileNet':
        out_dim = 128
        model = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
        model = torch.nn.Sequential(*list(model.children())[:-1],
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(model.classifier[1].in_features, out_dim))  # output (B, out_dim)
    else:
        raise ValueError(f'Full image cnn branch {name} not supported.')
    if double_fc:
        fc2 = nn.Linear(out_dim, out_dim)
        model = nn.Sequential(model, fc2)

    return {'model': model, 'out_dim': out_dim, 'normalizer': normalizer}


class GenericModel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # check GPU number
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        else:
            raise EnvironmentError('This model needs GPU but cuda is not detected.')

        # setup object detection
        det_cfg = objdet.default_cfg('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.obj_detector = objdet.BatchPredictor(det_cfg, hps.obj_detection_bsz, hps.obj_feat_level, hps.obj_big_box)
        self.ncats = self.obj_detector.ncats
        
        # setup object feat extractor
        if hps.obj_feat_level == 0:
            obj_feat_class = getattr(hash_end2end, hps.obj_feat_extractor)
            cnn_bsz = hps.obj_extraction_bsz if torch.cuda.device_count() == 1 else 2*hps.obj_extraction_bsz
            self.obj_feat_extractor = obj_feat_class(hps.obj_feat_weight, f'batch_size={cnn_bsz}', self.device)
        
        self.cnn_normalizer = lambda x: x  # dummy normalizer, will be set later

        # get dimension of dummy input data
        # dummy_x is a dict of input data, notable keys are x_vis, x_geo, x_rela, y_img, y_obj, mask_vis and mask_rela
        dummy_x = self.preprocess([np.zeros((224, 224, 3), dtype=np.uint8)], None)

        self.build_model(hps, dummy_x)  # build model based on hyper-param hps and input data dimension
        
        # param initialization
        self.apply(self.init_weights(hps.weight_init))
        self.to(self.device)  # transfer model param to device

        # train attributes
        self.optimizer = None
        self.writer = None

    def init_weights(self, init_type='gaussian'):
        initrange = 0.1
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        return init_fun

    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # not updating running mean/var
            module.weight.requires_grad = False  # not updating weight/bis, or alpha/beta in the paper
            module.bias.requires_grad = False

    @staticmethod
    def create_new_rela(rela, obj, edge):
        """
        concatenate obj to 2 sides of rela to make a triplet (obj_i, rela_ij, obj_j)
        """
        # convert 3D to 2D, note the order of dimension here is S, B, D
        obj_seq_len, bsz = obj.shape[:2]
        rela_seq_len = rela.shape[0]
        obj = obj.view(obj_seq_len*bsz, -1)
        rela = rela.view(rela_seq_len*bsz, -1)
        edge = edge * bsz + np.arange(bsz)[None,:,None]  # offset indices for correct 2D indexing
        edge = edge.reshape(rela_seq_len*bsz, 2)

        id_i = edge[..., 0]
        id_j = edge[..., 1]
        obj_i = obj[id_i]
        obj_j = obj[id_j]

        new_rela = torch.cat((obj_i, rela, obj_j), dim=1).view(rela_seq_len, bsz, -1)
        return new_rela

    def extract_object_features(self, x, proposals):
        """
        extract object features
        :param proposals [list] each element is the proposals for an image
        return: proposals with key box_features updated.
        """
        if self.hps.obj_feat_level:  # features are extract inside object detector
            return proposals
        # construct a list of cropped objects
        cursor = np.cumsum([proposal['count'] for proposal in proposals])[:-1]

        crops = []
        feats = []
        for i, proposal in enumerate(proposals):
            boxes_int = np.int64(proposal['box'])  # Nx4
            img_crops = [x[i][box[1]:box[3], box[0]:box[2]] for box in boxes_int]
            crops.extend(img_crops)
            if len(crops) > 64:  # enough buffer, extract feat
                feats.append(self.obj_feat_extractor.hash(crops))
                crops = []
        if len(crops) > 0:  # whatever left
            feats.append(self.obj_feat_extractor.hash(crops) + np.zeros((1,1), dtype=np.float32))
        feats = np.split(np.concatenate(feats), cursor)
        for i in range(len(proposals)):
            proposals[i].update(box_features=feats[i])
        return proposals

    def preprocess(self, x, y):
        """
        preprocess data, model dependent
        """
        # object detection
        proposals = self.preprocess_object_detection(x)
        return self.preprocess_object_feat_extraction(x, y, proposals)

    def preprocess_object_detection(self, x):
        proposals = objdet.get_maskrcnn_annotations(self.obj_detector(x), self.hps.seq_len, self.hps.obj_shuffle)
        if self.hps.inc_rela or self.hps.inc_geo:
            rela = objdet.get_geo_relation(proposals, [im.shape[:2] for im in x])
            for i in range(len(x)):
                proposals[i]['geo'], proposals[i]['rela'], proposals[i]['edge'] = rela[i]
        return proposals

    def preprocess_object_detection_next_chunk_train(self):
        x, y = next(self.train_loader)
        proposals = self.preprocess_object_detection(x)
        self.next_chunk_train = x, y, proposals

    def preprocess_object_detection_next_chunk_val(self):
        x, y = next(self.val_loader)
        proposals = self.preprocess_object_detection(x)
        self.next_chunk_val = x, y, proposals

    def train_update_current_chunk(self):
        self.train_thread.join()
        self.current_chunk_train = self.next_chunk_train
        self.train_thread = ExThread(target=self.preprocess_object_detection_next_chunk_train)
        self.train_thread.start()

    def val_update_current_chunk(self):
        self.val_thread.join()
        self.current_chunk_val = self.next_chunk_val
        self.val_thread = ExThread(target=self.preprocess_object_detection_next_chunk_val)
        self.val_thread.start()

    def preprocess_object_shape_extraction(self, pred_masks):
        out = [np.array([hu_moments(mask).astype(np.float32) for mask in masks]) for masks in pred_masks]
        return out

    def preprocess_object_feat_extraction(self, x, y, obj_proposals):
        proposals = self.extract_object_features(x, obj_proposals)  # extract features
        detect_res = objdet.join_dict(proposals)

        x_vis = detect_res['box_features']
        y_obj = detect_res['class']
        
        x_vis, y_obj, mask_vis = self.do_padding(x_vis, y_obj)
        out_y_img = None if y is None else torch.tensor(y, dtype=torch.float32).to(self.device)  # label at img level
        out = { 'x_vis': torch.tensor(x_vis).to(self.device),  # visual feat, (seqlen, B, D) or (B, seqlen, D)
                'y_obj': torch.tensor(y_obj, dtype=torch.long).to(self.device),  # object labels, (B, seqlen)
                'y_img': out_y_img,  # image label, (B,)
                'mask_vis': torch.tensor(mask_vis).to(self.device),  # visual mask, (B, seqlen)
                }
        out['x_sem'] = out['y_obj']  # object class label could also be training data
        if self.hps.inc_shape:
            shape_feats = self.preprocess_object_shape_extraction(detect_res['mask'])
            x_shape, _, _ = self.do_padding(shape_feats, detect_res['class'])  # pass but not using y_obj
            out['x_shape'] = torch.tensor(x_shape).to(self.device)

        if self.hps.inc_rela or self.hps.inc_geo:
            x_geo, x_rela, x_edge = detect_res['geo'], detect_res['rela'], detect_res['edge']
            x_geo, x_rela, x_edge, mask_geo, mask_rela = self.do_padding_rela(x_geo, x_rela, x_edge)
            out['x_geo'] = torch.tensor(x_geo).to(self.device)  # (seqlen, B, D) or (B, seqlen, D)
            out['x_rela'] = torch.tensor(x_rela).to(self.device)  # (seqlen^2, B, D) or (B, seqlen^2, D)
            if self.hps.cuda_all:
                out['x_edge'] = torch.tensor(x_edge).to(self.device)
            else:
                out['x_edge'] = x_edge  # (seqlen^2, B, 2) or (B, seqlen^2, 2)
            # out['mask_geo'] = mask_geo  # (B, seqlen), not being used as it is same as mask_vis
            out['mask_rela'] = torch.tensor(mask_rela).to(self.device)  # (B, seqlen^2)

        if self.hps.inc_cnn:
            x_img = [cv2.resize(im, (self.hps.cnn_size, self.hps.cnn_size), interpolation=cv2.INTER_LINEAR).astype(
                np.float32).transpose(2, 0, 1)[::-1]/255 for im in x]
            x_img = [self.cnn_normalizer(torch.tensor(im, dtype=torch.float)) for im in x_img]
            out['x_img'] = torch.stack(x_img).to(self.device)

        return out

    def do_padding(self, x, y):
        """
        pad data x with zeros and label y with -1 to seq_len, also return padding mask
        :param x: (B,SixD)  list of arrays of features
        :param y: (B,Si)
        :return: ((BxSxD), (BxS), (BxS))
        """
        bsz = len(y)
        max_seq_len = max([x[i].shape[0] for i in range(bsz)])
        x_out = np.zeros((bsz, max_seq_len, x[0].shape[-1]), dtype=np.float32)
        y_out = np.zeros((bsz, max_seq_len), dtype=int)
        padding_out = np.zeros_like(y_out, dtype=np.bool)
        for i in range(bsz):
            pad_len = max_seq_len - len(y[i])
            x_out[i, ...] = np.pad(x[i], ((0, pad_len), (0, 0)), mode='constant')
            y_out[i, ...] = np.pad(y[i], (0, pad_len), mode='constant', constant_values=self.ncats-1)
            if pad_len > 0:
                padding_out[i, -pad_len:] = True
        if self.hps.data_invert:
            x_out = x_out.transpose(1,0,2)
        return x_out, y_out, padding_out

    def do_padding_rela(self, x_geo, x_rela, x_edge):
        bsz = len(x_geo)
        max_geo_len = max([x_geo[i].shape[0] for i in range(bsz)])
        max_rela_len = max([x_rela[i].shape[0] for i in range(bsz)])
        assert max_rela_len == max_geo_len**2, "Error! nrela %d != nobj %d ^2" % (max_rela_len, max_geo_len)
        geo_out = np.zeros((bsz, max_geo_len, x_geo[0].shape[-1]), dtype=np.float32)
        rela_out = np.zeros((bsz, max_rela_len, x_rela[0].shape[-1]), dtype=np.float32)
        edge_out = np.zeros((bsz, max_rela_len, 2), dtype=int)
        mask_geo = np.zeros((bsz, max_geo_len), dtype=np.bool)
        mask_rela = np.zeros((bsz, max_rela_len), dtype=np.bool)
        for i in range(bsz):
            pad_geo = max_geo_len - len(x_geo[i])
            pad_rela = max_rela_len - len(x_rela[i])
            geo_out[i, ...] = np.pad(x_geo[i], ((0, pad_geo), (0, 0)))
            rela_out[i, ...] = np.pad(x_rela[i], ((0, pad_rela), (0, 0)))
            if pad_rela:
                edge_out[i, ...] = np.r_[x_edge[i], np.ones((pad_rela, 2), dtype=int)*(max_geo_len-1)]
                mask_rela[i, -pad_rela:] = True 
            else:
                edge_out[i, ...] = x_edge[i]
            if pad_geo:
                mask_geo[i, -pad_geo:] = True 
        if self.hps.data_invert:
            geo_out = geo_out.transpose(1,0,2)
            rela_out = rela_out.transpose(1,0,2)
            edge_out = edge_out.transpose(1,0,2)
        return geo_out, rela_out, edge_out, mask_geo, mask_rela

    def get_optimizer(self, hps):
        if hps.optimizer == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=hps.lr,
                                     betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-4)
        elif hps.optimizer == 'SGD':
            optim = torch.optim.SGD(self.parameters(), lr=hps.lr, momentum=0.9,
                                    weight_decay=5e-4)
        return optim

    def get_lr_scheduler(self, optimizer, hps):
        if hps.lr_scheduler == 'multistep':
            milestones = [int(hps.nepochs * i) for i in hps.lr_steps]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, hps.lr_sf)
        elif hps.lr_scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hps.lr_step_size, hps.lr_sf)
        return lr_scheduler
    
    @staticmethod
    def clip_gradient(optimizer, grad_clip):
        if grad_clip:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)   

    def load_pretrained_weight(self, pretrain_path):
        device = next(self.parameters()).device  # load to current device
        print('Loading pretrained model %s' % pretrain_path)
        pretrained_state = torch.load(pretrain_path, map_location=device)
        if 'model_state_dict' in pretrained_state:
            print('This pretrained model is a checkpoint, loading model_state_dict only.')
            pretrained_state = pretrained_state['model_state_dict']
        if 'state_dict' in pretrained_state:
            print('This is a bit odd! This pretrained model is an ALIENT checkpoint, try loading model_state_dict only.')
            pretrained_state = pretrained_state['state_dict']

        model_state = self.state_dict()
        matched_keys, not_matched_keys = [], []
        # import pdb; pdb.set_trace()
        for k,v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                matched_keys.append(k)
            else:
                not_matched_keys.append(k)
        if len(not_matched_keys):
            print('[%s] The following keys are not loaded: %s' % (self.hps.name, not_matched_keys))
            pretrained_state = {k: pretrained_state[k] for k in matched_keys}

        model_state.update(pretrained_state)
        self.load_state_dict(model_state)

    def load_checkpoint(self, checkpoint_path):
        print('Resuming from %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
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

    def train(self, mode=True):
        """
        override train fn with freezing batchnorm
        """
        super().train(mode)
        if self.hps.inc_cnn and self.hps.cnn_freeze_bn:
            self.cnn_model.apply(self.freeze_bn)  # freeze running mean/var in bn layers in cnn_model only

    ############################ to be overloaded #############################
    def build_model(self, hps, dummy_x):
        # build model and loss layers
        # pass, tobe overloaded
        pass
        
    def encode(self, x):
        # compute image representation
        # pass, to be overloaded
        pass

    def forward(self, src):
        """
        by default calling self.encode() but could be overloaded
        :param src: dict of input data
        :return: dict of output data, containing all losses
        """
        return self.encode(src)


class TransformerModel(GenericModel):
    def __init__(self, hps):
        assert hps.data_invert, 'Error! data_invert must set to True for transformer.'
        assert hps.tform_branch in ['vis', 'rela', 'both'], f'Error! Tform param {hps.tform_branch} not recognized.'
        super().__init__(hps)

    def build_model(self, hps, dummy_x):
        d_input = dummy_x['x_vis'].shape[-1]

        self.project_vis = build_projection_layer(d_input, hps.d_vis, hps.proj_leaky_relu, hps.proj_dropout, hps.do_full_proj)
        dim_x = hps.d_vis
        if hps.inc_geo:
            d_x_geo = dummy_x['x_geo'].shape[-1]
            self.project_geo = build_projection_layer(d_x_geo, hps.d_geo, hps.proj_leaky_relu, hps.proj_dropout, hps.do_full_proj)
            dim_x += hps.d_geo
        if hps.inc_shape:
            d_x_shape = dummy_x['x_shape'].shape[-1]
            self.project_shape = build_projection_layer(d_x_shape, hps.d_shape, hps.proj_leaky_relu, hps.proj_dropout, hps.do_full_proj)
            dim_x += hps.d_shape
        # vis projection prior transformer
        self.project_vis_tform = build_projection_layer(dim_x, hps.d_model_vis, hps.proj_leaky_relu, 0.1, hps.do_full_proj)
        full_attn = True if hps.tform_branch in ['both', 'vis'] else False
        if hps.use_tform:
            self.encoder = TransformerBaseEncoder(hps.d_model_vis,
                hps.nlayers, hps.nhead, hps.nhid, hps.model_dropout, hps.attn_mask, full_attn, name='tform_vis')
        else:
            self.encoder = GCNBaseEncoder(hps.d_model_vis, hps.nlayers, hps.model_dropout, hps.attn_mask, full_attn, name='gcn_vis')
        
        buffer_dim = hps.buffer_dim if hps.do_buffer else []
        if hps.tendrill_loss:
            self.buffer_vis = FCSeries(hps.d_model_vis, buffer_dim, relu_last=hps.buffer_relu_last)
        
        d_preemb = hps.d_model_vis  # input of embedding layer
        
        if hps.inc_rela:
            d_x_rela = dummy_x['x_rela'].shape[-1]
            self.project_rela = build_projection_layer(d_x_rela, hps.d_rela, hps.proj_leaky_relu, hps.proj_dropout, hps.do_full_proj)
            new_d_rela = hps.d_rela + 2*dim_x
            self.project_rela_tform = build_projection_layer(new_d_rela, hps.d_model_rela, hps.proj_leaky_relu, 0.1, hps.do_full_proj)
            full_attn = True if hps.tform_branch in ['both', 'rela'] else False
            if hps.use_tform:
                self.encoder_rela = TransformerBaseEncoder(hps.d_model_rela,
                hps.nlayers, hps.nhead, hps.nhid, hps.model_dropout, hps.attn_mask, full_attn, name='tform_rela')
            else:
                self.encoder_rela = GCNBaseEncoder(hps.d_model_rela,
                hps.nlayers, hps.model_dropout, hps.attn_mask, full_attn, name='gcn_rela')
            if hps.tendrill_loss:
                self.buffer_rela = FCSeries(hps.d_model_rela, buffer_dim, relu_last=hps.buffer_relu_last)
            d_preemb += hps.d_model_rela
        
        if hps.inc_cnn:
            cnn_model=get_cnn_model(hps.cnn_name, hps.cnn_weight, hps.cnn_train_all, hps.cnn_double_fc)
            self.cnn_model, self.cnn_normalizer, cnn_dim = cnn_model['model'], cnn_model['normalizer'], cnn_model['out_dim']
            if hps.tendrill_loss:
                self.buffer_cnn = FCSeries(cnn_dim, buffer_dim, relu_last=hps.buffer_relu_last)
            d_preemb += cnn_dim

        if hps.embed_activation == 'tanh':
            self.embed_layer = nn.Sequential(nn.Linear(d_preemb, hps.d_embed), nn.Tanh())
        else:
            self.embed_layer = nn.Linear(d_preemb, hps.d_embed)

        # classification (object level)
        if hps.do_classify:
            self.classifier = ObjectClassifier(hps.d_embed, hps.class_num)
        if hps.do_logistic:
            self.logistic_classifier = nn.Linear(hps.d_embed, 1)
        if hps.do_greedy:
            self.binariser = GreedyHashLoss()

        if hps.do_triplet or hps.do_simclr or hps.do_contrast:
            self.buffer_layer = FCSeries(hps.d_embed, buffer_dim, relu_last=hps.buffer_relu_last)

    def encode(self, x):
        # encoder
        output = dict(embedding=None, attn_weight=None, classify=None, regress=None, logistic=None)
        x_vis = self.project_vis(x['x_vis'])  # (S_v,N_v,D_model)
        if self.hps.inc_geo:
            x_geo = self.project_geo(x['x_geo'])  # (S_g,N_g, D_model), N_g = N_v, S_g=S_v
            assert x_geo.shape[0] == x_vis.shape[0] and x_geo.shape[1] == x_vis.shape[1]  # check seq_len and bsz
            x_vis = torch.cat((x_vis, x_geo), dim=2)  # (S_v, N_v, 2*D_model)
        if self.hps.inc_shape:
            x_shape = self.project_shape(x['x_shape'])
            x_vis = torch.cat((x_vis, x_shape), dim=2)
        vis_output = self.project_vis_tform(x_vis)
        vis_output = self.encoder(vis_output, x['mask_vis'])  # return {'embedding':, 'attn_weight'}
        # output.update(**vis_output)
        output['vis_attn'] = vis_output['attn_weight']
        bottleneck = vis_output['embedding']
        
        if self.hps.inc_rela:
            x_rela = self.project_rela(x['x_rela'])  # (S_r, N_r, D_model)
            x_rela2 = self.create_new_rela(x_rela, x_vis, x['x_edge'])
            rela_output = self.project_rela_tform(x_rela2)
            rela_output = self.encoder_rela(rela_output, x['mask_rela'])
            output['rela_attn'] = rela_output['attn_weight']
            bottleneck = torch.cat((vis_output['embedding'], rela_output['embedding']), dim=-1)
        
        if self.hps.inc_cnn:
            x_cnn = self.cnn_model(x['x_img'])
            bottleneck = torch.cat((bottleneck, x_cnn), dim=-1)
            
        embed_float = self.embed_layer(bottleneck)
        if self.hps.do_greedy:
            output['embedding'], output['greedy_loss'] = self.binariser(embed_float)
        elif self.hps.do_quantise or self.hps.do_balance:
            output['embedding'] = embed_float.tanh()
        else:
            output['embedding'] = embed_float

        if self.hps.do_classify:
            output['classify'] = self.classifier(output['embedding'])

        if self.hps.do_logistic:
            output['logistic'] = self.logistic_classifier(output['embedding'])

        if self.hps.do_triplet or self.hps.do_simclr or self.hps.do_contrast:
            output['regress'] = self.buffer_layer(output['embedding'])
            if self.hps.tendrill_loss:
                output['regress_vis'] = self.buffer_vis(vis_output['embedding'])
                output['regress_rela'] = self.buffer_rela(rela_output['embedding']) if self.hps.inc_rela else 0
                output['regress_cnn'] = self.buffer_cnn(x_cnn) if self.hps.inc_cnn else 0

        return output


class TransformerBaseEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead, nhid, dropout_rate, 
            attn_mask=False, full_attn=True, name='tform_vis'):
        """
        Input
        :param d_model  input feature dimension
        :param n_layers number of tform layers
        :param nhead    number of attention head
        :param nhid     fastforward conv dimension
        :param dropout_rate     dropout in tform layer
        :param attn_mask        whether attn mask is also passed to self attention layer
        :param full_attn        if True both transformer and self attn is enabled, otherwise only self attn     
        """
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.full_attn = full_attn
        if full_attn:  # do transformer + self attn
            encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout_rate)
            self.encoder = TransformerEncoder(encoder_layers, n_layers)
        self.attn = SelfAttn(d_model)
        self.pass_attn_mask = attn_mask

    def forward(self, x, mask):
        output = dict(embedding=None, attn_weight=None)
        if self.full_attn:
            x = self.encoder(x, src_key_padding_mask=mask)  # (S,N,D_model)
        attn_mask = ~mask.T if self.pass_attn_mask else None  # (S, B)
        output['embedding'], output['attn_weight'] = self.attn(x, attn_mask)  # (N,D_model)
        return output

    def extra_repr(self):
        return '%s ??? -> ?x%d' % (self.name, self.d_model)


class TransformerHash(object):
    def __init__(self, transformer_weight, params=''):
        """
        hasher for transformer, following the format of hash_end2end
        Note: we define the hasher here instead to avoid circular import at hash_end2end
        """
        model_dir = os.path.dirname(transformer_weight)
        config = os.path.join(model_dir, 'config.json')
        self.vhp = default_hparams()
        load_config(self.vhp, config, False)
        self.vhp.obj_detection_bsz = 4  # to fit small GPU
        if 'obj_shuffle' in self.vhp._hparam_types:
            self.vhp.obj_shuffle = False  # remove randomness in inference, unless override below
        if params:
            self.vhp.parse(params)
        print_config(self.vhp)
        if self.vhp.do_greedy or self.vhp.do_quantise or self.vhp.do_balance:
            print('TransformerHash - this model produces binary hash code')
            self.binary = True
        else:
            self.binary = False
        self.vmodel = TransformerModel(self.vhp)
        self.vmodel.load_pretrained_weight(transformer_weight)
        self.vmodel.eval()
        print(self.vmodel)

    def hash(self, cv2_imgs, return_prehash=False):
        n = len(cv2_imgs)
        out = []
        with torch.no_grad():
            for i in range(0, n, self.vhp.batch_size):
                start_id, end_id = i, min(n, i+self.vhp.batch_size)
                imgs = cv2_imgs[start_id:end_id]
                data = self.vmodel.preprocess(imgs, None)
                res = self.vmodel.encode(data)
                res = res['embedding'].cpu().numpy()
                out.append(res)
        out = np.concatenate(out).squeeze()
        out_bin = out > 0 if self.binary else out 
        if return_prehash:
            return out_bin, out 
        else:
            return out_bin
