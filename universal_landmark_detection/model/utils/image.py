import os
import SimpleITK as sitk
import numpy as np


def coord2index(coord, origin, direction, spacing):
    '''
        direction: 9 item list-like
        spacing: 3 item list-like
        rtype: tuple
    '''
    D = np.matrix(direction).reshape(3, 3)
    S = np.diag(spacing)
    m = (D*S).I
    index = ((np.array(coord)-origin)*m).getA().flatten().tolist()
    # ::-1  convert numpy coord system to simpleitk
    return tuple(round(i) for i in index)[::-1]


def index2coord(index, origin, direction, spacing):
    '''
        direction: 9 item list-like
        spacing: 3 item list-like
        rtype: tuple
    '''
    # ::-1 convert numpy coord system to simpleitk
    index = np.array(index[::-1])
    D = np.matrix(direction).reshape(3, 3)
    S = np.diag(spacing)
    m = (D*S)
    coord = (index*m).getA().flatten()+origin
    return tuple(coord.tolist())


def getInfo(itk):
    '''
        itk image
    '''
    info = {}
    info['direction'] = itk.GetDirection()
    info['origin'] = itk.GetOrigin()
    info['spacing'] = itk.GetSpacing()
    return info


def setInfo(itk, info):
    ''' 
        itk img
        info: dic
    '''
    itk.SetDirection(info['direction'])
    itk.SetOrigin(info['origin'])
    itk.SetSpacing(info['spacing'])
    return itk


def getFlip(dires, target=(1, 0, 0, 0, -1, 0, 0, 0, -1)):
    ''' 
        flip arr according to itk directions,
        note that itk is zyx, and numpy array is xyz

        dires: itk direction, 9 items tuple
        rtype: channnelx3d = 4d  slices
    '''
    # channelx3d
    isRight = [True, dires[8]*target[8] > 0, dires[4]
               * target[4] > 0, dires[0]*target[0] > 0]
    return tuple(slice(None, None, 2*i-1) for i in isRight)


def readITK(path):
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        itk = reader.Execute()
    else:
        itk = sitk.ReadImage(path)
    return itk
