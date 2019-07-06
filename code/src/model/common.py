import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MotionCompensator(nn.Module):
    def __init__(self):
        self.device = 'cuda'
        '''if args.cpu:
            self.device = 'cpu' '''
        super(MotionCompensator, self).__init__()
        print("Creating Motion compensator")

        def _gconv(in_channels, out_channels, kernel_size=3, groups=1, stride=1, bias=True):
            return nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size, groups=groups, stride=stride,
                             padding=(kernel_size // 2), bias=bias)

        # Coarse flow
        coarse_flow = [_gconv(6, 32, kernel_size=5, stride=2), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        # coarse_flow.extend([_gconv(32, 32, kernel_size=3), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        # coarse_flow.extend([_gconv(24, 32, kernel_size=5, stride=2), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        # coarse_flow.extend([_gconv(32, 16, kernel_size=3), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(32, 16, kernel_size=3, dilation=2, padding=2), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        coarse_flow.extend([_gconv(16, 8, kernel_size=3), nn.Tanh()])
        # coarse_flow.extend([nn.PixelShuffle(4)])
        coarse_flow.extend([nn.PixelShuffle(2)])

        self.C_flow = nn.Sequential(*coarse_flow)

        # Fine flow
        fine_flow = [_gconv(11, 48, kernel_size=5, stride=2), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        for _ in range(3):
            fine_flow.extend([_gconv(48, 48, kernel_size=3), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        fine_flow.extend([_gconv(48, 8, kernel_size=3), nn.Tanh()])
        fine_flow.extend([nn.PixelShuffle(2)])

        self.F_flow = nn.Sequential(*fine_flow)

    def forward(self, frame_1, frame_2):
        # Create identity flow
        x = np.linspace(-1, 1, frame_1.shape[3])
        y = np.linspace(-1, 1, frame_1.shape[2])
        xv, yv = np.meshgrid(x, y)
        id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
        self.identity_flow = torch.from_numpy(id_flow).float().to(self.device)

        # Coarse flow
        coarse_in = torch.cat((frame_1, frame_2), dim=1)
        coarse_out = self.C_flow(coarse_in)
        coarse_out[:, 0] /= frame_1.shape[3]
        coarse_out[:, 1] /= frame_2.shape[2]
        frame_2_compensated_coarse = self.warp(frame_2, coarse_out)

        # Fine flow
        fine_in = torch.cat((frame_1, frame_2, frame_2_compensated_coarse, coarse_out), dim=1)
        fine_out = self.F_flow(fine_in)
        fine_out[:, 0] /= frame_1.shape[3]
        fine_out[:, 1] /= frame_2.shape[2]
        flow = (coarse_out + fine_out)

        frame_2_compensated = self.warp(frame_2, flow)

        return frame_2_compensated, flow

    def warp(self, img, flow):
        # https://discuss.pytorch.org/t/solved-how-to-do-the-interpolating-of-optical-flow/5019
        # permute flow N C H W -> N H W C
        img_compensated = F.grid_sample(img, (-flow.permute(0, 2, 3, 1) + self.identity_flow).clamp(-1, 1),
                                        padding_mode='border')
        return img_compensated
