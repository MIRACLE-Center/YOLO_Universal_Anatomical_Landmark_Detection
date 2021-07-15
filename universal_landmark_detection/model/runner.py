import os
from tqdm import tqdm
from PIL import Image
import random
import shutil

import SimpleITK as sitk
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from .utils import *
from .datasets import get_dataset
from .networks import get_loss, get_optim, get_net, get_scheduler


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.phase = self.args.phase

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_opts(self):
        self.opts = get_config(self.args.config)
        update_config(self.opts, self.args)
        self.run_name = self.opts.run_name if self.opts.run_name else self.opts.model
        self.run_dir = os.path.join(self.opts.run_dir, self.run_name)
        mkdir(self.run_dir)
        toYaml("{rd}/config_{ph}.yaml".format(rd=self.run_dir, ph=self.phase), self.opts)
        shutil.copy(self.args.config, '{run_dir}/config_origin.yaml'.format(run_dir=self.run_dir))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts.cuda_devices
        self.setup_seed(self.opts.seed)

    def get_loader(self):
        def _get(s='train'):
            dataset_list = []
            loader_list = []
            trans_dic = self.opts['transform_params'] if s == 'train' else {}
            for name in self.name_list:
                d = get_dataset(name)(phase=s, transform_params=trans_dic,
                                      use_background_channel=use_background_channel, **d_opts[name])
                dataset_list.append(d)
                loader_opts = self.opts.dataloader[s]
                if s == 'train' and 'batch_size_dic' in d_opts:
                    loader_opts['batch_size'] = d_opts['batch_size_dic'][name]

                l = DataLoader(d, **loader_opts)
                loader_list.append(l)
            setattr(self, s + '_dataset_list', dataset_list)
            setattr(self, s + '_loader_list', loader_list)

        d_opts = self.opts.dataset
        use_background_channel = self.opts.use_background_channel
        self.loss_weights = d_opts['loss_weights']
        self.name_list = d_opts['name_list']
        self.train_name_list = d_opts['name_list']
        if self.phase == 'train':
            _get('train')
            _get('validate')
            step = int(self.opts.mix_step)
            if step > 0:
                self.train_name_list = ['mix']
                self.train_loader_list = [MixIter(self.train_loader_list, step)]
        elif self.phase == 'test':
            _get('test')
        else:
            _get('validate')

    def get_model(self):
        def get_learner():
            learn = self.opts.learning
            self.loss = get_loss(learn['loss'])(**learn[learn['loss']])
            self.val_loss = get_loss(learn['loss'])(**learn[learn['loss']])
            self.optim = get_optim(learn['optim'])(
                self.model.parameters(), **learn[learn['optim']])
            if learn['use_scheduler']:
                self.scheduler = get_scheduler(learn['scheduler'])(
                    self.optim, **learn[learn['scheduler']])
            else:
                self.scheduler = None
        modelname = self.opts.model
        model_opts = self.opts[modelname] if modelname in self.opts else {}
        localNet = model_opts['localNet'] if 'localNet' in model_opts else None
        dataset = self.opts['dataset']
        channel_params = {'in_channels': [], 'out_channels': []}
        is_2d_flags = []
        img_size_list = []
        for name in dataset['name_list']:
            size = dataset[name]['size']
            is_2d_flags.append(len(size) == 2)
            img_size_list.append(size)
            channel_params['in_channels'].append(1)
            channel_params['out_channels'].append(dataset[name]['num_landmark'])
        if self.opts.use_background_channel:
            li = channel_params['out_channels']
            for i in range(len(li)):
                li[i] += 1

        globalNet_params = channel_params.copy()
        localNet_params = channel_params.copy()

        if modelname.startswith('gln'):
            globalNet_params_final = model_opts['globalNet_params']
            for k, v in globalNet_params.items():
                globalNet_params_final[k] = v
            self.model = get_net(modelname)(get_net(localNet), localNet_params, globalNet_params_final)
        else:
            net_params = net_params = self.opts[modelname] if modelname in self.opts else dict()
            net_params.update(channel_params)
            self.model = get_net(modelname)(** net_params)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # self.model = torch.nn.DataParallel(self.model)  # TODO
        # get_checkpoint
        if os.path.isfile(self.opts.checkpoint):
            print('loading checkpoint:', self.opts.checkpoint)
            checkpoint = torch.load(self.opts.checkpoint)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            get_learner()
        else:
            self.start_epoch = 0
            get_learner()
        with open(os.path.join(self.run_dir, 'network_graph.txt'), 'w') as f:
            f.write(str(self.model))
        if torch.cuda.is_available() and self.phase == 'train':
            self.loss = self.loss.cuda()
            self.val_loss = self.val_loss.cuda()
        self.device = next(self.model.parameters()).device

    def get_logger(self):
        pass

    def config(self):
        self.get_opts()
        self.get_loader()
        self.get_model()
        self.get_logger()

    def run(self):
        self.config()
        self.best_loss = self.train_loss = float('inf')
        if self.phase == 'train':
            self.train()
            self.phase = 'test'
            self.get_loader()
            self.get_model()
            self.validateTest(self.start_epoch)
        else:
            self.validateTest(self.start_epoch)

    def validateTest(self, epoch=None):
        def saveImage(path, arr):
            if len(arr.shape) == 3:
                path = path + '.nii'
                img = sitk.GetImageFromArray(arr)
                sitk.WriteImage(img, path)
            else:
                path = path + '.png'
                arr = norm(arr) * 255
                img = Image.fromarray(arr).convert('P')
                img.save(path)

        def save_data(data_dic):  # TODO
            '''
                outer vars:
                    read only: epoch, name, dest, 
            '''
            output_batch = data_dic['output'].detach().cpu().numpy()
            input_batch = data_dic['input'].detach().cpu().numpy()
            if 'gt' in data_dic:
                gt_batch = data_dic['gt'].detach().cpu().numpy()
            else:
                gt_batch = output_batch
            for n, (input_, gt, output) in enumerate(zip(input_batch, gt_batch, output_batch)):
                pre = dest + '/' + data_dic["name"][n]

                saveImage(pre + '_input', input_[0])
                saveImage(pre + '_output', output[0])
                np.save(pre + '_output.npy', output)
                np.save(pre + '_input.npy', input_)
                if 'gt' in data_dic:
                    np.save(pre + '_gt.npy', gt)
                    saveImage(pre + '_gt', gt[0])
                    if len(output.shape) == 3:  # 2d images, channel x width x height
                        out = visualMultiChannel(output)
                        saveImage(pre + '_output', out)
                        comp = np.concatenate((visualMultiChannel(gt), out), axis=-1)
                        saveImage(pre + '_gt-pred', comp)
        self.model.eval()  # important
        if epoch is None:
            epoch = self.start_epoch
        prefix = self.run_dir + '/results/' + self.phase + \
            '_epoch{epoch:03d}'.format(epoch=epoch)
        loss_dir = self.run_dir + '/results/loss'
        mkdir(loss_dir)
        mkdir(prefix)
        s = 'validate' if self.phase == 'train' else self.phase
        loader_list = getattr(self, '{}_loader_list'.format(s))
        val_loss = 0
        allep = self.opts.epochs
        for task_idx, (name, cur_loader) in enumerate(zip(self.name_list, loader_list)):
            dest = os.path.join(prefix, name)  # is read in func save_data
            mkdir(dest)
            batch_num = len(cur_loader)
            pbar = tqdm(enumerate(cur_loader))  # is read in func save_data
            name_loss_dic = {}
            for i, data_dic in pbar:
                for k in {'input', 'gt'}:
                    if k in data_dic:
                        data_dic[k] = torch.autograd.Variable(
                            data_dic[k]).to(self.device)
                with torch.no_grad():
                    data_dic.update(self.model(data_dic['input'], task_idx))
                if epoch + 1 >= allep or self.phase != 'train':
                    save_data(data_dic)
                if 'gt' in data_dic:
                    if data_dic['output'].shape != data_dic['gt'].shape:
                        print(data_dic['path'])
                        exit()
                    loss = self.val_loss(data_dic['output'], data_dic['gt'])  # TODO
                    if 'rec_image' in data_dic:
                        loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'], data_dic['rec_image'])
                    pbar.set_description("[{curPhase} {dataname}] epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, vali:{val_loss:.6f}".format(dataname=name.rjust(13),
                                                                                                                                                                           ep=epoch, allep=allep, num=(i + 1), batch_num=batch_num, val_loss=loss.item(), ls=self.train_loss, curPhase=s.ljust(8)))
                    name_loss_dic['_'.join(data_dic['name'])] = loss.item()
            if 'gt' in data_dic:
                mean = np.mean(list(name_loss_dic.values()))
                val_loss += mean
                path = os.path.join(
                    loss_dir, 'epoch_{:03d}_loss_{:.6f}.txt'.format(epoch, mean))
                with open(path, 'w') as f:
                    for k, v in name_loss_dic.items():
                        f.write('{:.6f} {}\n'.format(v, k))
        return val_loss

    def train(self):
        self.model.train()
        checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        mkdir(checkpoint_dir)
        pbar = tqdm(range(self.start_epoch, self.opts.epochs))
        xs, ys = [], []
        self.lr_list = []
        loss_file = os.path.join(checkpoint_dir, 'train_val_loss.txt')
        endEpoch = self.opts.epochs - 1
        save_freq = self.opts.save_freq
        eval_freq = self.opts.eval_freq
        for epoch in pbar:
            self.update_params(epoch, pbar)
            plot_2d(self.run_dir + '/learning_rate.png', list(range(len(self.lr_list))), self.lr_list, 'step', 'lr', 'lr-step')
            if epoch % eval_freq == 0 or epoch == endEpoch:
                val_loss = self.validateTest(epoch)
                xs.append(epoch)
                ys.append(val_loss)
                plot_2d(self.run_dir + '/loss.png', xs, ys,
                        'epoch', 'loss', 'epoch-loss')
                data = {
                    'epoch': epoch,
                    'model_name': self.run_name,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer': self.optim,
                    'scheduler': self.scheduler,
                }
                save_name = "{rn}_epoch{epoch:03d}_train{trainloss:.6f}_val{valloss:.6f}.pt".format(
                    rn=self.run_name, epoch=epoch, valloss=val_loss, trainloss=self.train_loss)
                with open(loss_file, 'a') as f:
                    f.write('{:03d},{:.6f},{:.6f}\n'.format(
                        epoch, self.train_loss, val_loss))
                if (save_freq != 0 and epoch % save_freq == 0) or epoch == endEpoch:
                    torch.save(data, os.path.join(checkpoint_dir, save_name))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    dest = os.path.join(checkpoint_dir, 'best_' + save_name)
                    self.opts.checkpoint = dest
                    torch.save(data, dest)

    def update_params(self, epoch, pbar):
        # to try: harmonic mean
        self.model.train()  # important
        self.train_loss = 0  # sum of different datasets' arithmetic mean
        allep = self.opts.epochs
        use_scheduler = self.opts.learning.use_scheduler
        for task_idx, (name, loader) in enumerate(zip(self.train_name_list, self.train_loader_list)):
            batch_num = len(loader)
            cur_loss = 0
            for i, data_dic in enumerate(loader):
                if isinstance(data_dic, tuple):
                    task_idx = data_dic[1]
                    data_dic = data_dic[0]
                for k in {'input', 'gt'}:
                    data_dic[k] = torch.autograd.Variable(data_dic[k]).to(self.device)
                data_dic.update(self.model(data_dic['input'], task_idx))
                self.optim.zero_grad()
                loss = self.loss(data_dic['output'], data_dic['gt'])  # TODO
                if 'rec_image' in data_dic:
                    loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'], data_dic['rec_image'])
                if hasattr(self, 'loss_weights'):
                    loss *= self.loss_weights[task_idx]
                cur_loss += loss.item()
                loss.backward()
                self.lr_list.append(self.optim.param_groups[0]['lr'])
                self.optim.step()
                if use_scheduler:
                    self.scheduler.step()  # behind optim.step()
                pbar.set_description("[train    {dataname}] epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, best:{val_loss:.6f}".format(
                    dataname=name.rjust(13), ep=epoch, allep=allep, num=i + 1, batch_num=batch_num, ls=loss.item(), val_loss=self.best_loss))
            self.train_loss += cur_loss / len(loader)
