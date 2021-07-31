# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import numpy as np
from models.generic_models import TransformerHash


org_path = 'examples/original.jpg'
aug_path = 'examples/benign.jpg'
pho_path = 'examples/manipulated.jpg'
WEIGHT = 'weight/best.pt'


if __name__ == '__main__':
    ims = [cv2.imread(path, cv2.IMREAD_COLOR) for path in (org_path, aug_path, pho_path)]
    gtn = TransformerHash(WEIGHT)
    feats = gtn.hash(ims)

    aug_dist = np.bitwise_xor(feats[0], feats[1]).sum()
    pho_dist = np.bitwise_xor(feats[0], feats[2]).sum()

    print(f'Hamming ({os.path.basename(org_path)}, {os.path.basename(aug_path)}): {aug_dist}')
    print(f'Hamming ({os.path.basename(org_path)}, {os.path.basename(pho_path)}): {pho_dist}')
    