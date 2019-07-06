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

class VSRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale[0]
        self.idx_scale = 0
        self.n_seq = args.n_sequence
        self.args.task = " "
        print("n_seq:", args.n_sequence)
        # self.image_range : need to make it flexible in the test area
        self.img_range = 200
        self.n_frames_video = []
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        if train:
            self._set_filesystem_train(args.dir_data)
        else:
            self._set_filesystem_val(args.dir_data)
        self.images_hr, self.images_lr = self._scan()
        self.num_video = len(self.images_hr)
        print("Number of videos to load:", self.num_video, len(self.images_hr[0]))
        if train:
            print(args.test_every, self.num_video, self.args.batch_size)
            self.repeat = args.test_every // max((self.num_video // self.args.batch_size), 1)
        if args.process:
            self.data_hr, self.data_lr = self._load(self.num_video)

    # Below functions as used to prepare images
    def _scan(self):
        """
        Returns a list of image directories
        """
        if self.train:
            # training datasets are labeled as .../Video*/HR/*.png
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, 'Youku*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, 'Youku*')))
        else:
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, 'Youku*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, 'Youku*')))
        assert len(vid_hr_names) == len(vid_lr_names)

        names_hr = []
        names_lr = []

        if self.train:
            for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
                start = random.randint(0, self.img_range - self.args.n_frames_per_video)
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.bmp')))[start: start+self.args.n_frames_per_video]
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.bmp')))[start: start+self.args.n_frames_per_video]
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
                self.n_frames_video.append(len(hr_dir_names))
        else:
            for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.bmp')))
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.bmp')))
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
                self.n_frames_video.append(len(hr_dir_names))
        return names_hr, names_lr

    def _load(self, n_videos):
        data_lr = []
        data_hr = []
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" %idx)
            lrs, hrs, _ = self._load_file(idx)
            hrs = np.array([imageio.imread(hr_name) for hr_name in self.images_hr[idx]])
            lrs = np.array([imageio.imread(lr_name) for lr_name in self.images_lr[idx]])
            data_lr.append(lrs)
            data_hr.append(hrs)
        #data_lr = common.set_channel(*data_lr, n_channels=self.args.n_colors)
        #data_hr = common.set_channel(*data_hr, n_channels=self.args.n_colors)
        return data_hr, data_lr

    def _set_filesystem_train(self, dir_data):

        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'round1_train_label')
        self.dir_lr = os.path.join(self.apath, 'round1_train_input')

    def _set_filesystem_val(self, dir_data):

        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'round1_val_label')
        self.dir_lr = os.path.join(self.apath, 'round1_val_input')

    def __getitem__(self, idx):
        #start = time.time()
        if self.args.process:
            if self.args.task == "MC":
                lrs, filenames = self._load_file_from_loaded_data(idx)
            else:
                lrs, hrs, filenames = self._load_file_from_loaded_data(idx)
        else:
            if self.args.task == "MC":
                lrs, filenames = self._load_file(idx)
            else:
                lrs, hrs, filenames = self._load_file(idx)
        if self.args.task == "MC":
            _, ih, iw, _ = lrs.shape
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            ix = random.randrange(0, iw - self.args.patch_size_w + 1)
            iy = random.randrange(0, ih - self.args.patch_size_h + 1)
            patches = [self.get_patch(lr, ix, iy, hflip, vflip, rot90, 1) for lr in lrs]
            lrs = np.array([patch for patch in patches])
            lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
            lr_tensors = common.np2Tensor(*lrs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
            #end=time.time()
            #print(end-start)
            return torch.stack(lr_tensors), filenames
        else:
            _, ih, iw, _ = lrs.shape
            ix = random.randrange(0, iw - self.args.patch_size_w + 1)
            iy = random.randrange(0, ih - self.args.patch_size_h + 1)
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            patches = [self.get_patch(lr, ix, iy, hflip, vflip, rot90, 1) for lr in lrs]
            lrs = np.array([patch for patch in patches])
            hrs = self.get_patch(hrs, ix, iy, hflip, vflip, rot90, self.args.scale[0])
            lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
            hrs = np.array(common.set_channel(*[hrs], n_channels=self.args.n_colors))
            lr_tensors = common.np2Tensor(*lrs,  rgb_range=self.args.rgb_range)
            hr_tensors = common.np2Tensor(*hrs,  rgb_range=self.args.rgb_range)
            #end = time.time()
            #print(end - start)
            return torch.stack(lr_tensors), torch.stack(hr_tensors), filenames


    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            # if test, call all possible video sequence fragments
            return sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_video
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """
        idx = self._get_index(idx)
        if self.train:
            if self.args.task == "MC":
                f_lrs = self.images_lr[idx]
                start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.n_seq))
                filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_lrs[start:start + self.n_seq]]
                lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs[start:start + self.n_seq]])
            else:
                f_hrs = self.images_hr[idx]
                f_lrs = self.images_lr[idx]
                start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.n_seq))
                filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_hrs[start:start+self.n_seq]]
                hrs = np.array(imageio.imread(f_hrs[start+self.n_seq//2]))
                #hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs[start:start+self.n_seq]])
                lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs[start:start+self.n_seq]])

        else:
            if self.args.task == "MC":
                n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
                video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
                f_lrs = self.images_lr[video_idx][frame_idx:frame_idx + self.n_seq]
                filenames = [
                    os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for
                    file in f_lrs]
                lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
            else:
                n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
                video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
                f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
                f_lrs = self.images_lr[video_idx][frame_idx:frame_idx+self.n_seq]
                filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]
                hrs = np.array(imageio.imread(f_hrs[self.n_seq//2]))
                #hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs])
                lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
        if self.args.task == "MC":
            return lrs, filenames
        else:
            return lrs, hrs, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        if self.train:
            start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.n_seq))
            hrs = self.data_hr[idx][start:start+self.n_seq]
            lrs = self.data_lr[idx][start:start+self.n_seq]
            filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.images_hr[idx]]

        else:
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            hrs = self.data_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            lrs = self.data_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]

        return lrs, hrs, filenames

    def get_patch(self, lhr, ix, iy, hflip, vflip, rot90, scale):
        """
        Returns patches for multiple scales
        """
        '''if self.train:
            ih, iw = lr.shape[:2]
            ih -= ih % 4
            iw -= iw % 4
            lr = lr[:ih, :iw]
            hr = hr[:ih * scale, :iw * scale]'''
        if self.train:
            #patch_size = self.args.patch_size - (self.args.patch_size % 4)
            lhr = common.get_patch_onlyLR(
                    lhr,
                    patch_size_w=self.args.patch_size_w,
                    patch_size_h=self.args.patch_size_h,
                    ix=ix,
                    iy=iy,
                    scale=scale
                )
            if not self.args.no_augment:
                #lhr = img_as_ubyte(transform.rotate(lhr, jiao))
                lhr = common.augment(lhr, hflip, vflip, rot90)

        '''else:
            ih, iw, _ = lhr.shape
            ih = int(ih/scale)
            iw = int(iw/scale)
            ih -= ih % 4
            iw += iw % 4
            lhr = lhr[:ih*scale, :iw*scale]'''

        return lhr
