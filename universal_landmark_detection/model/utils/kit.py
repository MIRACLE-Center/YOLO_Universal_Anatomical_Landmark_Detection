import os
from itertools import product
from functools import partial
from collections.abc import Iterable

import torch
import numpy as np
import SimpleITK as sitk

from .graphics import *


def mkdir(path):
    def _mkdir(p):
        if p and not os.path.exists(p):
            _mkdir(p[:p.rfind(os.path.sep)])
            os.mkdir(p)
    _mkdir(os.path.abspath(path))


def getName(path):
    path = os.path.basename(path)
    n = path.rfind('.')
    if n != -1:
        path = path[:n]
    return path


def readDir(path, recursive=True, isSorted=True, filter_func=None):
    def _readDir(path, recursive=recursive):
        li = os.listdir(path)
        fs = []
        ds = []
        for i in li:
            f = os.path.join(path, i)
            if os.path.isfile(f):
                if filter_func is None or filter_func(f):
                    fs.append(f)
            else:
                ds.append(f)
        if recursive:
            for d in ds:
                fs += _readDir(d, recursive)
        return fs
    fs = _readDir(path, recursive)
    if isSorted:
        fs.sort()
    return fs


def norm(x, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = x.min(), x.max()
    else:
        x[x < vmin] = vmin
        x[x > vmax] = vmax
    if vmin == vmax:
        return x
    else:
        return (x-vmin)/(vmax-vmin)


def gaussianHeatmap(sigma, dim: int = 2, nsigma: int = 3):
    if nsigma <= 2:
        print('[Warning]: nsigma={} is recommended to be greater than 2'.format(nsigma))
    radius = round(nsigma*sigma)
    center = tuple([radius for i in range(dim)])
    mask_shape = tuple([2*radius for i in range(dim)])
    mask = np.zeros(mask_shape, dtype=np.float)
    sig2 = sigma**2
    coef = sigma*np.sqrt(2*np.pi)
    for p in product(*[range(i) for i in mask_shape]):
        d2 = sum((i-j)**2 for i, j in zip(center, p))
        mask[p] = np.exp(-d2/sig2/2)/coef
    mask = (mask-mask.min())/(mask.max()-mask.min()) # necessary?, yes, the output heatmap is processed with sigmoid

    def genHeatmap(point, shape):
        ret = np.zeros(shape, dtype=np.float)
        bboxs = [(max(0, point[ax]-radius), min(shape[ax], point[ax]+radius))
                 for ax in range(dim)]
        img_sls = tuple([slice(i, j) for i, j in bboxs])
        mask_begins = [max(0, radius-point[ax]) for ax in range(dim)]
        mask_sls = tuple([slice(beg, beg+j-i)
                          for beg, (i, j) in zip(mask_begins, bboxs)])
        ret[img_sls] = mask[mask_sls]
        return ret
    return genHeatmap


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def getPointsFromHeatmap(arr):
    ''' 
        arr: numpy.ndarray, channel x imageshape
        ret: [(x,y..)]* channel
    '''
    points = []
    for img in arr:
        index = img.argmax()
        points.append(unravel_index(index, img.shape))
    return points


def genPoints(start, end, n=6, min_ratio=0, max_ratio=1):
    '''
        start,end are n-dim points of a line
    '''
    start, end = np.array(start), np.array(end)
    diff = end-start
    for i in np.linspace(min_ratio, max_ratio, n):
        yield tuple((start+diff*i+0.5).astype(np.int16).tolist())


def color(arr, seeds, distance, max_distance, marker):
    for pt in globe(arr.shape, seeds, distance, max_distance):
        arr[pt] = marker


def colorRGB(arr, seeds, distance, max_distance, marker):
    '''arr is in shape of wxhx3, marker is 3-item list-like'''
    for i in range(3):
        color(arr[:, :, i], seeds, distance, max_distance, marker[i])


def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj
