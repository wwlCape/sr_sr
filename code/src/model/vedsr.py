from model import common
from approx_huber_loss import Approx_Huber_Loss

import torch
import torch.nn as nn


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return VEDSR(args)

class VEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VEDSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        #act = nn.ReLU(True)
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.mseloss = nn.MSELoss()
        self.huberloss = Approx_Huber_Loss()

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_fusion = [
            conv(3 * n_feats, n_feats, kernel_size), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale),
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale),
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale),
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale),
            conv(n_feats, n_feats, kernel_size)
        ]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        resb = [
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale),
            conv(n_feats, n_feats, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)


        self.mc = common.MotionCompensator()
        self.mc.load_state_dict(torch.load('../ModelMc.pt'), strict=False)

        self.tAtt_1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)

        self.fus = nn.Sequential(*m_fusion)
        self.resb = nn.Sequential(*resb)
        self.tail = nn.Sequential(*m_tail)
        '''for p in self.parameters():
             p.requires_grad = False'''
        self.fusion = nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1, bias=True)
    def forward(self, frame_list):
        # squeeze frames n_sequence * [N, 1, n_colors, H, W] -> n_sequence * [N, n_colors, H, W]
        frame_list = [torch.squeeze(frame, dim=1) for frame in frame_list]

        frame0 = frame_list[0]
        frame1 = frame_list[1]
        frame2 = frame_list[2]
        frame3 = frame_list[3]
        frame4 = frame_list[4]

        frame0 = self.sub_mean(frame0)
        frame1 = self.sub_mean(frame1)
        frame2 = self.sub_mean(frame2)
        frame3 = self.sub_mean(frame3)
        frame4 = self.sub_mean(frame4)

        frame0_c, flow0 = self.mc(frame2, frame0)
        head0 = self.head(frame0_c)

        frame1_c, flow1 = self.mc(frame2, frame1)
        head1 = self.head(frame1_c)

        head2 = self.head(frame2)

        frame3_c, flow3 = self.mc(frame2, frame3)
        head3 = self.head(frame3_c)

        frame4_c, flow4 = self.mc(frame4, frame4)
        head4 = self.head(frame4_c)

        res0 = self.body(head0)
        res0 = res0 + head0
        res1 = self.body(head1)
        res1 = res1 + head1
        res = self.body(head2)
        res3 = self.body(head3)
        res3 = res3 + head3
        res4 = self.body(head4)
        res4 = res4 + head4

        res1 = torch.cat((res0, res1), dim=1)
        res1 = self.fusion(res1)

        res3 = torch.cat((res4, res3), dim=1)
        res3 = self.fusion(res3)

        head = torch.stack((res1, res, res3), dim=1)
        B, N, C, H, W = head.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(head[:, 1, :, :, :].clone())
        emb = self.tAtt_1(head.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        head = head.view(B, -1, H, W) * cor_prob


        head = self.fus(head)
        res = self.resb(head)

        res = res + head2
        x = self.tail(res)

        x = self.add_mean(x)

        loss_soft_mc_mse = self.mseloss(frame1_c, frame3_c)
        loss_mc_mse = self.mseloss(frame1_c, frame2) + self.mseloss(frame3_c, frame2)
        loss_mc_huber = self.huberloss(flow1) + self.huberloss(flow3)

        return x, loss_mc_huber, loss_mc_mse, loss_soft_mc_mse

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


