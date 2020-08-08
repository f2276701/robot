#!/usr/bin/env python
#coding:utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from collections import OrderedDict
import os

import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from utils import Logger, make_result_img


class Model:
    def __init__(self, args):
        """Training model for VAE and CVAE

        Attributes:
            prior_dist: string, indicating the prior distribution of input image,
                        should be either 'B': Bernoulli or 'G': Gaussian distribution
            rec_num: int, indicating how many images should be saved per log_iter 
        """

        self.name = args.net
        self.epoch = args.epoch
        self.lr = args.lr
        self.beta = args.beta
        self.prior_dist = args.prior_dist

        self.model_save_path = args.model_save_path 
        self.model_save_iter= args.model_save_iter
        self.log_path = args.log_path
        self.log_iter = args.log_iter
        self.rec_num = args.rec_num

        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        if args.tensorboard:
            self.writer = SummaryWriter(os.path.join(self.log_path + "tensorboard_log"))

        self.loss_list = OrderedDict([('total_loss', .0),
                                      ('rec_loss', .0),
                                      ('kld_loss', .0)])
        
        if args.validate:
            self.vali_loss_list = OrderedDict([('total_loss', .0),
                                               ('rec_loss', .0),
                                               ('kld_loss', .0)])

    def build(self, net, *args, **kwargs):
        self.net = net(*args, **kwargs).to(self.device)
        optim_params = (param for param in self.net.parameters() if param.requires_grad)
        self.optim = optim.Adam(optim_params, lr=self.lr)
        self.logger = Logger(self)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, 
                                                   step_size=max(50, self.epoch // 2), 
                                                   gamma=0.1)

    def loss(self, outputs, targets, mu, log_var):
        if self.prior_dist == "G":
            rec_loss = F.mse_loss(outputs, targets, reduction="none").sum([1,2,3]).mean()
        elif self.prior_dist == "B":
            rec_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none").sum([1,2,3]).mean()
        kld_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(1).mean()
        total_loss = rec_loss + self.beta * kld_loss
        return total_loss, rec_loss, kld_loss

    def load(self, pt_file):
        #
        # Using pretrained VAE and freezing encoder and drop the fc layer to train the CVAE
        #

        pre_params = torch.load(pt_file)
        state_dict = self.net.state_dict()
        for name, param in pre_params.items():
            if "fc" in name:
                pre_params.pop(name)
            #if "dec_block" in name:
            #    pre_params.pop(name)
        state_dict.update(pre_params)
        self.net.load_state_dict(state_dict)
        #for name, param in self.net.named_parameters():
        #    if "enc_block" in name:
        #        param.requires_grad = False

    def train(self, train_loader, vali_loader=None):
        self.net.train()
        for epoch in range(1, self.epoch + 1):
            total_sum, rec_sum, kld_sum = .0, .0, .0
            for i, batch in enumerate(train_loader, 1):
                batch_inputs = [b.to(self.device) for b in batch[:-1]]
                batch_target = batch[-1].to(self.device)

                self.optim.zero_grad()
                logits, outputs, mu, log_var = self.net(*batch_inputs) 
                if self.prior_dist == "G":
                    total_loss, rec_loss, kld_loss = self.loss(outputs, batch_target, 
                                                               mu, log_var)
                elif self.prior_dist == "B":
                    total_loss, rec_loss, kld_loss = self.loss(logits, batch_target, 
                                                               mu, log_var)

                total_sum += total_loss.item()
                rec_sum += rec_loss.item()
                kld_sum += kld_loss.item()

                total_loss.backward()
                self.optim.step()
            self.scheduler.step()

            self.loss_list['total_loss'] = total_sum / i
            self.loss_list['rec_loss'] = rec_sum / i
            self.loss_list['kld_loss'] = kld_sum / i
            
            if epoch % self.log_iter == 0:
                if vali_loader is not None:
                    self.test(vali_loader)
                data = iter(train_loader).next()
                imgs = [b.to(self.device) for b in data[:-1]]
                with torch.no_grad():
                    logits, outputs, _, _ = self.net(*imgs) 
                imgs = [b[:self.rec_num] for b in data]
                if self.prior_dist == "G":
                    imgs.append(outputs.detach().cpu()[:self.rec_num])
                    self.result_img = make_result_img(imgs, normalize=True, range=(-1., 1.))
                elif self.prior_dist == "B":
                    imgs.append(torch.sigmoid(logits.detach().cpu())[:self.rec_num])
                    self.result_img = make_result_img(imgs)

                self.logger(epoch)

            if epoch % self.model_save_iter == 0:
                torch.save(self.net.state_dict(), 
                           self.model_save_path + "{}_{:0>6}.pt".format(self.name, epoch))

    def test(self, data_loader):
        with torch.no_grad():
            total_sum, rec_sum, kld_sum = .0, .0, .0
            for i, batch in enumerate(data_loader, 1):
                batch_inputs = [b.to(self.device) for b in batch[:-1]]
                batch_target = batch[-1].to(self.device)

                logits, outputs, mu, log_var = self.net(*batch_inputs)
                if self.prior_dist == "G":
                    total_loss, rec_loss, kld_loss = self.loss(outputs, batch_target, 
                                                               mu, log_var)
                elif self.prior_dist == "B":
                    total_loss, rec_loss, kld_loss = self.loss(logits, batch_target, 
                                                               mu, log_var)

                total_sum += total_loss.item()
                rec_sum += rec_loss.item()
                kld_sum += kld_loss.item()

            self.vali_loss_list['total_loss'] = total_sum / i
            self.vali_loss_list['rec_loss'] = rec_sum / i
            self.vali_loss_list['kld_loss'] = kld_sum / i
        return outputs if self.prior_dist == "G" else torch.sigmoid(logits)
