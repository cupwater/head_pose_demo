'''
Author: Peng Bo
Date: 2022-11-21 09:50:38
LastEditTime: 2022-11-21 10:30:35
Description: 

'''
import numpy as np
from math import ceil
from itertools import product

import pdb

def gen_bbox(img_size=(320, 180)):
    min_sizes_list = [[128, 256], [256, 512]]
    variance  = [0.1, 0.2]
    steps = [16, 32]
    feature_maps = [[ceil(img_size[0]/step), ceil(img_size[1]/step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_list[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx, s_ky = min_size/img_size[1], min_size/img_size[0]
                dense_cx = [x*steps[k]/img_size[1] for x in [j+0.5]]
                dense_cy = [y*steps[k]/img_size[0] for y in [i+0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    anchors = np.array(anchors).reshape(-1, 4)
    return np.clip(anchors, 0, 1), variance

def decode(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors,4]
        anchors: Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate([
                anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
                anchors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
            ], axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, anchors, variances):
    """Decode landm from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre: landm predictions for loc layers,
            Shape: [num_priors,10]
        anchors: Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate([
                    anchors[:, :2] + pre[:, :2] * variances[0] * anchors[:, 2:],
                    anchors[:, :2] + pre[:, 2:4] * variances[0] * anchors[:, 2:],
                    anchors[:, :2] + pre[:, 4:6] * variances[0] * anchors[:, 2:],
                    anchors[:, :2] + pre[:, 6:8] * variances[0] * anchors[:, 2:],
                    anchors[:, :2] + pre[:, 8:10] * variances[0] * anchors[:, 2:],
             ], axis=1)
    return landms