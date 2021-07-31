# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from PIL import Image
import requests
import cv2
from concurrent import futures



def hu_moments(binary_image, log=False):
    """compute 7 hu moments (log scale) of a binary image"""
    moments = cv2.moments(np.uint8(binary_image)*255)
    humoments = cv2.HuMoments(moments).squeeze()
    if log:
        humoments = - np.log10(np.abs(humoments)) * np.sign(humoments)
    return humoments


def read_image_url(url, mode=None):
    """read image from URL
    mode: if None, read as it is; else convert to the specified mode.
    return numpy array of RGB image"""
    assert mode in [None, 'RGB', 'L'], "Error! Only [None, RGB, L] is supported."
    response = requests.get(url)
    img = BytesIO(response.content)
    out = Image.open(img)
    if mode:
        out = out.convert(mode)
    return np.array(out)


def read_image_path_url(paths, urls, verbose=True):
    """
    read images from paths, if not avai from urls
    :param paths: list of full paths
    :param urls: list of urls
    :return: BGR numpy array [0, 255]
    """
    out_ = []
    for id_ in range(len(paths)):
        path_ = paths[id_]
        url_ = urls[id_]
        if os.path.exists(path_):
            out_.append(cv2.imread(path_, cv2.IMREAD_COLOR))
        else:
            if verbose:
                print('Local path %s not exist. Download online image.' % path_)
            out_.append(read_image_url(url_, 'RGB')[:, :, ::-1])
    return out_


def resize_maxdim(im_array, max_size=224, pad_mode='constant', **kwargs):
    """ resize image to have fixed max dimension keep aspect ratio, then pad to have square size
    pad_mode follow np.pad settings: {'constant', 'edge', 'maximum', 'mean', 'reflect', 'symmetric', 'wrap', etc.}
    **kwargs follow np.pad settings wrt. pad_mode
    e.g.
    x = np.ones(200, 100)
    y = resize_maxdim(x, 224, 'constant', constant_values=0)
    z = resize_maxdim(x, 224, 'edge')
    """
    h, w = im_array.shape[:2]
    scale = float(max_size) / max(h, w)
    if h > w:
        newh, neww = max_size, int(scale * w)
        padx = int((max_size - neww)/2)
        pad_width = [(0, 0), (padx, max_size - padx - neww)]
    else:
        newh, neww = int(scale*h), max_size
        pady = int((max_size - newh)/2)
        pad_width = [(pady, max_size - pady - newh), (0, 0)]
    if len(im_array.shape) > 2:  # color image
        pad_width.append((0, 0))
    pil_img = Image.fromarray(im_array).resize((neww, newh), Image.LINEAR)  # channel order doesn't matter
    im_out = np.asarray(pil_img)
    im_out = np.pad(im_out, pad_width, pad_mode, **kwargs)
    return im_out


def downsize_shortest_edge(im_array, shortest_edge=800):
    """
    resize image if shortest edge is above a given value, keep aspect ratio
    """
    h, w = im_array.shape[:2]
    if min(h, w) > shortest_edge:  # resize
        ratio = shortest_edge / min(h, w)
        newh, neww = ratio * h, ratio * w
        newh, neww = int(newh+0.5), int(neww+0.5)
        out = cv2.resize(im_array, (neww, newh), interpolation=cv2.INTER_LINEAR)
    else:
        out = im_array
    return out