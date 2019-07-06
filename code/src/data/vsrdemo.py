import sys
import os
import glob
import time
from skimage import transform, img_as_ubyte
from data import common
import pickle
import numpy as np
import imageio
import random
import torch
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt

class VSRDemo(data.Dataset):
    def __init__(self, args, name='VSRDemo', train=False, benchmark=False):
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
        print("filelists length: ", len(self.filelist))

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        if idx == 0 or idx == 1:
            lrs = [imageio.imread(self.filelist[i]) for i in [idx, idx + 1, idx, idx + 1, idx]]
        elif idx == 4999 or idx == 4998:
            lrs = [imageio.imread(self.filelist[i]) for i in [idx, idx - 1, idx, idx - 1, idx]]
        else:
            lrs = [imageio.imread(self.filelist[i]) for i in [idx - 2, idx - 1, idx, idx + 1, idx + 2]]
        #lr = imageio.imread(self.filelist[idx])
        lrs = common.set_channel(*lrs, n_channels=self.args.n_colors)
        lrs_t = common.np2Tensor(*lrs, rgb_range=self.args.rgb_range)

        return torch.stack(lrs_t), -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
