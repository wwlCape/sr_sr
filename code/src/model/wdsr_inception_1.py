import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def make_model(args, parent=False):
    return MODEL(args)


class InceptionBlock(nn.Module):

    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(InceptionBlock, self).__init__()
        self.res_scale = res_scale
        body0 = []
        body1 = []
        expand = 6
        linear = 0.8
        body0.append(
            wn(nn.Conv2d(2 * n_feats, n_feats*expand, 1, padding=1//2)))
        body0.append(act)
        body0.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body0.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body0 = nn.Sequential(*body0)

        body1.append(
            wn(nn.Conv2d(2*n_feats, n_feats * expand, 1, padding=1 // 2)))
        body1.append(act)
        body1.append(
            wn(nn.Conv2d(n_feats * expand, int(n_feats * linear), 1, padding=1 // 2)))
        body1.append(
            wn(nn.Conv2d(int(n_feats * linear), n_feats, 1, padding=1 // 2)))

        self.body1 = nn.Sequential(*body1)

    def forward(self, x):
        body0 = self.body0(x)
        body1 = self.body1(x)
        body = torch.cat((body0, body1), dim=1)
        res = body * self.res_scale
        res += x
        return res

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class ChannelPool(nn.Module):
    def __init__(self, n_feats, n_split):
        super(ChannelPool, self).__init__()
        self.conv = nn.Conv2d(n_feats, 1, 1, padding=0)
        self.split = n_split

    def forward(self, x):
        size = [s for s in x.size()]
        size[1] = 0
        tmp = torch.empty(size).cuda()
        tt = x.split(self.split, dim=1)
        for _, i in enumerate(x.split(self.split, dim=1)):
            i = self.conv(i)
            tmp = torch.cat([tmp, i], dim=1).cuda()

        return tmp

class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        # scale = args.scale[0]
        scale = 2
        # n_resblocks = 16
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, 2 * n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                InceptionBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn))

        # define tail module
        tail = []
        n_scale = 4
        out_feats1 = n_scale*scale*scale*args.n_colors
        out_feats = scale*scale*args.n_colors
        tail.append(
            wn(nn.Conv2d(2*n_feats, out_feats1, 3, padding=3//2)))
        tail.append(ChannelPool(n_scale, n_scale))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = [torch.squeeze(frame, dim=1) for frame in x]
        x = x[0]
        x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x*127.5 + self.rgb_mean.cuda()*255
        return x
