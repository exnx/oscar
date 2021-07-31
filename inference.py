# -*- coding: utf-8 -*-
#!/usr/bin/env python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import argparse
from models.generic_models import TransformerHash


IN = 'examples/1a7p8x.jpg'
WEIGHT = 'weight/best.pt'


def extract(model, cv2_im):
    out = model.hash([cv2_im])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image hash with Graph Transformer Network')
    parser.add_argument('-i', '--input', default=IN, help='input image')
    parser.add_argument('-w', '--weight', default=WEIGHT, help='model weight')
    args = parser.parse_args()
    
    gtn = TransformerHash(args.weight)
    im = cv2.imread(args.input, cv2.IMREAD_COLOR)
    feat = extract(gtn, im)
    print(f'Input image: {args.input}\nOutput: {feat}')
