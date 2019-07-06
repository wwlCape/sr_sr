import os
import math
from decimal import Decimal
import pdb
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer_VSR():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            #lr = lr[:, , :, :, :] #####################
            lr = list(torch.split(lr, 1, dim=1))
            hr = hr[:, 0, :, :, :] #####################

            #lr, hr = self.prepare(lr, hr)
            lr = [x.to(self.device) for x in lr]
            hr = hr.to(self.device)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.args.n_sequence>1:
                sr, loss_mc_mse, loss_mc_huber, loss_soft_mc_mse= self.model(lr, idx_scale)
                loss_mc = 0.005 * loss_mc_mse + 0.0005 * loss_mc_huber + 0.01 * loss_soft_mc_mse
                loss_sr = self.loss(sr, hr)
                loss = loss_sr + loss_mc
            else:
                sr = self.model(lr, idx_scale)
                loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                #d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr = list(torch.split(lr, 1, dim=1))
                    if not self.args.test_only:
                        hr = hr[:, 0, :, :, :]  #####################

                    #lr, hr = self.prepare(lr, hr)
                    lr = [x.to(self.device) for x in lr]
                    hr = hr.to(self.device)

                    if self.args.n_sequence > 1:
                        sr, _, _, _ = self.model(lr, idx_scale)
                    else:
                        sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs




'''import os
import math
import time
import imageio
import decimal

import numpy as np
from scipy import misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from tqdm import tqdm

import utils


class Trainer_VSR:
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        self.loss = nn.MSELoss()

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        print("weight decay: ",self.args.weight_decay)
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        print("VSR training")
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))

        self.model.train()
        self.ckp.start_log()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            #lr: [batch_size, n_seq, 3, patch_size, patch_size]
            if self.args.n_colors == 1 and lr.size()[2] == 3:
                lr = lr[:, :, 0:1, :, :]
                hr = hr[:, :, 0:1, :, :]

            # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> n_sequence * [N, 1, n_colors, H, W]    
            lr = list(torch.split(lr, self.args.n_colors, dim = 1))
            
            # target frame = middle HR frame [N, n_colors, H, W]
            hr = hr[:, int(hr.shape[1]/2), : ,: ,:]
            
            #lr = lr.to(self.device)
            lr = [x.to(self.device) for x in lr]
            hr = hr.to(self.device)
            self.optimizer.zero_grad()
            # output frame = single HR frame [N, n_colors, H, W]
            if self.model.get_model().name == 'ESPCN_mf':
                sr = self.model(lr)
                loss = self.loss(sr, hr)
            elif self.model.get_model().name == 'VESPCN':
                sr, loss_mc_mse, loss_mc_huber = self.model(lr)
                loss_mc = self.args.beta * loss_mc_mse + self.args.lambd * loss_mc_huber
                loss_espcn = self.loss(sr, hr)
                loss = loss_espcn + loss_mc
                
            self.ckp.report_log(loss.item())
            loss.backward()
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {:.5f}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)))
                #print(loss_mc.item(), loss_espcn.item())
        self.ckp.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                ycbcr_flag = False
                filename = filename[0][0]
                # lr: [batch_size, n_seq, 3, patch_size, patch_size]
                if self.args.n_colors == 1 and lr.size()[2] == 3:
                    # If n_colors is 1, split image into Y,Cb,Cr
                    ycbcr_flag = True
                    # for CbCr, select the middle frame
                    lr_center_y = lr[:, int(lr.shape[1]/2), 0:1, :, :].to(self.device)
                    lr_cbcr = lr[:, int(hr.shape[1]/2), 1:, :, :].to(self.device)
                    hr_cbcr = hr[:, int(hr.shape[1]/2), 1:, :, :].to(self.device)
                    # extract Y channels (lr should be group, hr should be the center frame)
                    lr = lr[:, :, 0:1, :, :]
                    hr = hr[:, int(hr.shape[1]/2), 0:1, :, :]

                # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> n_sequence * [N, 1, n_colors, H, W]    
                lr = list(torch.split(lr, self.args.n_colors, dim = 1))

                #lr = lr.to(self.device)
                lr = [x.to(self.device) for x in lr]
                hr = hr.to(self.device)

                # output frame = single HR frame [N, n_colors, H, W]
                if self.model.get_model().name == 'ESPCN_mf':
                    sr = self.model(lr)
                elif self.model.get_model().name == 'VESPCN':
                    sr, _, _ = self.model(lr)
                PSNR = utils.calc_psnr(self.args, sr, hr)
                self.ckp.report_log(PSNR, train=False)
                hr, sr = utils.postprocess(hr, sr, rgb_range=self.args.rgb_range,
                                               ycbcr_flag=ycbcr_flag, device=self.device)

                if self.args.save_images and idx_img%30 == 0:
                    if ycbcr_flag:
                        [lr_center_y] = utils.postprocess(lr_center_y, rgb_range=self.args.rgb_range,
                                                        ycbcr_flag=ycbcr_flag, device=self.device)
                        lr = torch.cat((lr_center_y, lr_cbcr), dim=1)
                        hr = torch.cat((hr, hr_cbcr), dim=1)
                        sr = torch.cat((sr, hr_cbcr), dim=1)

                    save_list = [lr, hr, sr]

                    self.ckp.save_images(filename, save_list, self.args.scale)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                self.args.data_test, self.ckp.psnr_log[-1],
                                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs '''