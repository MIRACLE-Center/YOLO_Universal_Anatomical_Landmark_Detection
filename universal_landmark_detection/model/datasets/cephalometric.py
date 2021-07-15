import os
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data

from ..utils import gaussianHeatmap, transformer


class Cephalometric(data.Dataset):

    def __init__(self, prefix, phase, transform_params=dict(), sigma=10, num_landmark=19, size=[640, 800], use_background_channel=False):

        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel

        self.pth_Image = os.path.join(prefix, 'raw')
        self.pth_label_junior = os.path.join(prefix, '400_junior')
        self.pth_label_senior = os.path.join(prefix, '400_senior')

        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        n = len(files)
        if phase == 'train':
            self.indexes = files[:130]
        elif phase == 'validate':
            self.indexes = files[130:150]
        elif phase == 'test':
            self.indexes = files[150:400]
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=phase))
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))

        # # todo
        # self.pth_Image='./missing_landmarks/images'
        # self.pth_label_junior = './missing_landmarks'
        # self.pth_label_senior = './missing_landmarks'
        # self.indexes = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, origin_size = self.readImage(
            os.path.join(self.pth_Image, name+'.bmp'))

        # # todo
        # name='001'

        points = self.readLandmark(name, origin_size)
        li = [self.genHeatmap(point, self.size) for point in points]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm > 1] = 1
            li.append(1-sm)
        gt = np.array(li)
        # gt = np.array([self.genHeatmap(point, self.size) for point in points])
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        points = []
        with open(os.path.join(self.pth_label_junior, name+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, name+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().rstrip('\n').split(',')
                    landmark2 = f2.readline().rstrip('\n').split(',')
                    landmark = [(float(i)+float(j))/2 for i,
                                j in zip(landmark1, landmark2)]
                    # todo
                    # if landmark[0]>origin_size[0] or landmark[1]>origin_size[1]:
                    #    landmark=[0,0]

                    points.append(tuple(round(p*new/old) for p, new,
                                        old in zip(landmark, self.size, origin_size)))
        return points

    def readImage(self, path):
        '''Read image from path and return a numpy.ndarray in shape of cxwxh
        '''
        # width x height:1935 x 2400, channel:3
        img = Image.open(path)
        origin_size = img.size

        # see also, cv2.resize
        # resize, width x height,  channel=3
        img = img.resize(self.size)
        # height x width, because all channels are the same
        arr = np.array(img)[:, :, 0]
        # channel x width x height: 1 x width x height
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float)
        # conveting to float is important, otherwise big bug occurs
        for i in range(arr.shape[0]):
            arr[i] = (arr[i]-arr[i].mean())/(arr[i].std()+1e-20)
        return arr, origin_size
