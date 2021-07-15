import numpy as np
from skimage import transform as sktrans


def rotate(angle):
    '''
        angle: °
    '''
    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        for i in range(img.shape[0]):
            ret.append(sktrans.rotate(img[i], angle))
        return np.array(ret)
    return func


def translate(offsets):
    ''' translation
        offsets: n-item list-like, for each dim
    '''
    offsets = tuple(offsets)
    new_sls = tuple(slice(i, None) for i in offsets)

    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        size = img.shape[1:]
        old_sls = tuple(slice(0, j-i) for i, j in zip(offsets, size))

        for old in img:
            new = np.zeros(size)
            new[new_sls] = old[old_sls]
            ret.append(new)
        return np.array(ret)
    return func


def flip(axis=1):
    '''
    axis=0: flip all
       else flip axis
    '''
    f_sls = slice(None, None, -1)
    sls = slice(None, None)

    def func(img):
        dim = img.ndim
        cur_axis = axis % dim
        if cur_axis == 0:
            all_sls = tuple([f_sls])*dim
        else:
            all_sls = tuple(
                            f_sls if i == cur_axis else sls for i in range(dim))
            return img[all_sls]
    return func


def transformer(param_dic):
    fs = []

    if 'flip_rate' in param_dic and np.random.rand() < param_dic['flip_rate']:
        fs.append(flip(param_dic['axis']))
    if 'rotate_rate' in param_dic and np.random.rand() < param_dic['rotate_rate']:
        fs.append(rotate(param_dic['angle']))
    if 'translate_rate' in param_dic and np.random.rand() < param_dic['translate_rate']:
        fs.append(translate(param_dic['offsets']))

    def trans(*imgs):
        ''' img: chanel x imageshape
        '''
        ret = []
        for img in imgs:
            # copy is necessary, to avoid modifying origin data
            cur_img = img.copy()
            for f in fs:
                cur_img = f(cur_img)
            # copy is necessary, torch needs ascontiguousarray
            ret.append(cur_img.copy())
        return tuple(ret)
    return trans
