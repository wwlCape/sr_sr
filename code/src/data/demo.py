import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data
import pdb

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        self.gtfilelist = []
        # for f in os.listdir(args.dir_demo):
        #     if f.find('.png') >= 0 or f.find('.bmp') >= 0:
        for root, dirs, files in os.walk(args.dir_demo):
            for file in files:
                file_index = file[:13]
                self.filelist.append(os.path.join(args.dir_demo,file_index, file))
                # pdb.set_trace()  
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

