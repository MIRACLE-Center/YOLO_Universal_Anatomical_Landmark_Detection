import os
from collections.abc import Iterable
import argparse
from functools import partial
from PIL import Image
from PIL import ImageDraw, ImageFont

from tqdm import tqdm
import scipy.io as sio
from scipy.optimize import linear_sum_assignment as assign
import numpy as np

from model.utils import mkdir, toYaml, dis2, colorRGB, getPointsFromHeatmap, get_config

PATH_DIC = {
    'cephalometric': '../data/ISBI2015_ceph/raw',
    'hand': '../data/hand/jpg',
    'chest': '../data/chest/pngs',
}

FONT_PATH = './times.ttf'
THRESHOLD = [2, 2.5, 3, 4, 6, 9, 10]
CEPH_PHYSICAL_FACTOR = 0.46875
WRIST_WIDTH = 50  # mm
DRAW_TEXT_SIZE_FACTOR = { 'cephalometric': 1.13, 'hand': 1, 'chest': 1.39}



def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


def radial(pt1, pt2, factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5


def draw_text(image, text, factor=1):
    width = round(40*factor)
    padding = round(10*factor)
    margin = round(5*factor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, width)
    text_size = draw.textsize(text, font)
    text_w = padding
    text_h = image.height-width-padding
    text_w = text_h = padding
    pos = [text_w, text_h, text_w + text_size[0], text_h+text_size[1]]
    draw.rectangle(pos, fill='#000000')  # 用于填充
    draw.text((text_w, text_h), text, fill='#00ffff', font=font)  # blue
    return image


def cal_all_distance(points, gt_points, factor=1):
    '''
    points: [(x,y,z...)]
    gt_points: [(x,y,z...)]
    return : [d1,d2, ...]
    '''
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q in zip(points, gt_points)]


def assigned_distance(points, gt_points, factor=1):
    n = len(points)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = radial(points[i], gt_points[j])*factor
    return [mat[i, j] for i, j in zip(*assign(mat))]


def get_sdr(distance_list, threshold=THRESHOLD):
    ''' successfully detection rate (pixel)
    '''
    ret = {}
    n = len(distance_list)
    for th in threshold:
        ret[th] = sum(d <= th for d in distance_list)/n
    return ret


def saveLabels(path, points, size):
    with open(path, 'w') as f:
        f.write('{}\n'.format(len(points)))
        for pt in points:
            ratios = ['{:.4f}'.format(x/X) for x,X in zip(pt,size)]
            f.write(' '.join(ratios)+'\n')

def evaluate(input_path, output_path, save_img=False, assigned=False, IS_DRAW_TEXT=False):
    mkdir(output_path)
    dataset = os.path.basename(input_path).lower()
    image_path_pre = PATH_DIC[dataset]
    print('\n'+'-'*20+dataset+'-'*20)
    print('input : ', input_path)
    print('output: ', output_path)
    print('image : ', image_path_pre)
    gen = [gt_p for gt_p in os.listdir(input_path) if gt_p.endswith('_gt.npy')]
    pbar = tqdm(gen, ncols=80)
    data_num = len(gen)
    out_label_path = os.path.join(output_path, 'labels')
    mkdir(out_label_path)
    out_gt_path = os.path.join(output_path, 'gt_laels')
    mkdir(out_gt_path)
    if save_img:
        out_img_path = os.path.join(output_path, 'images')
        mkdir(out_img_path)
    physical_factor = 1
    if dataset == 'cephalometric':
        physical_factor = CEPH_PHYSICAL_FACTOR
    distance_list = []
    pixel_dis_list = []
    assigned_list = []
    for i, gt_p in enumerate(pbar):
        pbar.set_description('{:03d}/{:03d}: {}'.format(i+1, data_num, gt_p))
        name = gt_p[:-7]
        heatmaps = np.load(os.path.join(input_path, name+'_output.npy'))
        img_size = heatmaps.shape[1:]
        cur_points = getPointsFromHeatmap(heatmaps)
        gt_map = np.load(os.path.join(input_path, gt_p))
        cur_gt = getPointsFromHeatmap(gt_map)


        if dataset == 'hand':
            physical_factor = WRIST_WIDTH/radial(cur_gt[0], cur_gt[4])
        cur_distance_list = cal_all_distance(cur_points, cur_gt, physical_factor)
        cur_pixel_dis = cal_all_distance(cur_points, cur_gt, 1)
        distance_list += cur_distance_list
        pixel_dis_list += cur_pixel_dis
        if assigned:
            assigned_list += assigned_distance(cur_points,
                                               cur_gt, physical_factor)
        saveLabels(out_label_path+'/'+name+'.txt', cur_points, img_size)
        saveLabels(out_gt_path+'/'+name+'.txt', cur_gt, img_size)
        if save_img:
            if dataset == 'cephalometric':
                img_path = image_path_pre+'/'+name+'.bmp'
            elif dataset == 'hand':
                img_path = image_path_pre+'/'+name+'.jpg'
            else:
                img_path = image_path_pre+'/'+name+'.png'
            img = Image.open(img_path)
            img = img.resize(img_size)
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img = np.transpose(img, (1, 0, 2))
            for p, q in zip(cur_points, cur_gt):
                colorRGB(img, [q], partial(dis2, q), 20, [0, 255, 0])
                colorRGB(img, [p], partial(dis2, p), 20, [255, 0, 0])
            img = np.transpose(img, (1, 0, 2))
            img = Image.fromarray(img)
            mre = np.mean(cur_distance_list)
            mre_str = '{:.3f}'.format(mre)
            if IS_DRAW_TEXT:
                img = draw_text(img, mre_str, DRAW_TEXT_SIZE_FACTOR[dataset])
            img.save(out_img_path+'/'+name + '_' + mre_str+'.png')
    if assigned:
        print('assigned...')
    return assigned_list if assigned else distance_list, pixel_dis_list


def analysis(li1, dataset):
    print('\n'+'-'*20+dataset+'-'*20)
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    print('MRE:', mean1)
    print('STD:', std1)
    print('SDR:')
    for k in sorted(sdr1.keys()):
        print('     {}: {}'.format(k, sdr1[k]))
    return summary


def get_args():
    parser = argparse.ArgumentParser()
    # optinal
    parser.add_argument("-s", "--save_img", action='store_true')
    parser.add_argument("-d", "--draw_text", action='store_true')
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-a", "--assigned", action='store_true')
    # required
    parser.add_argument("-i", "--input", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dic = {}
    pixel_dic = {}
    if not args.output:
        output = os.path.join('.eval', args.input.replace('/', '_'))
    for d in os.listdir(args.input):
        inp = os.path.join(args.input, d)
        if os.path.isdir(inp):
            phy_dis, pixel_dis = evaluate(inp, os.path.join(output, d), args.save_img,
                                          args.assigned, args.draw_text)
            phy_dis = np2py(phy_dis)
            pixel_dis = np2py(pixel_dis)
            dic[d] = phy_dis
            pixel_dic[d+'_pixel'] = pixel_dis
    toYaml(output+'/distance.yaml', dic)
    summary = {}
    li_total = []
    for d, phy_dis in dic.items():
        pixel_dis = pixel_dic[d+'_pixel']
        summary[d] = analysis(phy_dis, d)
        li_total += pixel_dis
    summary['total'] = analysis(li_total, 'total')
    toYaml(output+'/summary.yaml', summary)
