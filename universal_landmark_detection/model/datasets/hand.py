import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from ..utils import gaussianHeatmap, transformer


class Hand(data.Dataset):

    def __init__(self, prefix, phase, transform_params=dict(), sigma=10, num_landmark=19, size=[1000, 1400],use_background_channel=False):

        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark

        self.pth_Image = os.path.join(prefix, 'jpg')
        self.use_background_channel = use_background_channel
        self.labels = pd.read_csv(os.path.join(
            prefix, 'all.csv'), header=None, index_col=0)

        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        n = len(files)
        train_num = 550  # round(n*0.7)
        val_num = 59  # round(n*0.1)
        test_num = n - train_num - val_num
        if phase == 'train':
            self.indexes = files[:train_num]
        elif phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif phase == 'test':
            self.indexes = files[-test_num:]
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=phase))
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, origin_size = self.readImage(
            os.path.join(self.pth_Image, name+'.jpg'))

        points = self.readLandmark(name, origin_size)
        li = [self.genHeatmap(point, self.size) for point in points]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm>1]=1
            li.append(1-sm)
        gt = np.array(li)
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        li = list(self.labels.loc[int(name), :])
        r1, r2 = [i/j for i, j in zip(self.size, origin_size)]
        points = [tuple([round(li[i]*r1), round(li[i+1]*r2)])
                  for i in range(0, len(li), 2)]
        return points

    def readImage(self, path):
        '''Read image from path and return a numpy.ndarray in shape of cxwxh
        '''
        img = Image.open(path)
        origin_size = img.size

        # resize, width x height,  channel=1
        img = img.resize(self.size)
        arr = np.array(img)
        # channel x width x height: 1 x width x height
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float)
        # conveting to float is important, otherwise big bug occurs
        for i in range(arr.shape[0]):
            arr[i] = (arr[i]-arr[i].mean())/(arr[i].std()+1e-20)
        return arr, origin_size
