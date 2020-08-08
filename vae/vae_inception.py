#!/usr/bin/env python
#coding:utf-8


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from collections import OrderedDict
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

__all__ = ["VAE", "CVAE"]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        init.normal_(m.bias)
    elif isinstance(m, ConvLayer):
        for name, param in m.conv.named_parameters():
            if "weight" in name:
                nonlinear = m.act.__name__ if m.act is not None else "linear"
                init.kaiming_normal_(param, nonlinearity=nonlinear)
            elif "bias" in name:
                init.normal_(param)
    elif isinstance(m, DeconvLayer):
        for name, param in m.deconv.named_parameters():
            if "weight" in name:
                nonlinear = m.act.__name__ if m.act is not None else "linear"
                init.kaiming_normal_(param, nonlinearity=nonlinear)
            elif "bias" in name:
                init.normal_(param)


class ViewLayer(nn.Module):
    def __init__(self, size):
        super(ViewLayer, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, 
                 act=F.leaky_relu):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, inputs):
        outputs = self.bn(self.conv(inputs))
        return self.act(outputs) if self.act else outputs


class ConvInceptionLayerV1(ConvInceptionLayerV2):
    """inception module, see 'google net v1'
    """
    def __init__(self, channel):
        super(ConvInceptionLayerV1, self).__init__(channel)
        bn_channel = channel // 4
        self.conv1 = nn.Sequential(
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                       ConvLayer(bn_channel, bn_channel, 3, 1, 1),
                       ConvLayer(bn_channel, bn_channel, 5, 1, 2),
                     )


class ConvInceptionLayerV2(nn.Module):
    """inception module, see 'google net v2'
    """
    def __init__(self, channel):
        super(ConvInceptionLayerV2, self).__init__()
        bn_channel = channel // 4
        self.conv1 = nn.Sequential(
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                       ConvLayer(bn_channel, bn_channel, 3, 1, 1),
                       ConvLayer(bn_channel, bn_channel, 3, 1, 1),
                     )
        self.conv2 = nn.Sequential(
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                       ConvLayer(bn_channel, bn_channel, 3, 1, 1),
                     )
        self.conv3 = nn.Sequential(
                       nn.AvgPool2d(3, 1, 1),
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                     )
        self.conv4 = ConvLayer(channel, bn_channel, 1, 1, 0)

    def forward(self, inputs):
        outputs = torch.cat([self.conv1(inputs), 
                             self.conv2(inputs), 
                             self.conv3(inputs), 
                             self.conv4(inputs)], 1)
        return outputs


class EfficientDownLayer(nn.Module):
    """downsampling module, see 'google net v2'
    """
    def __init__(self, channel):
        super(EfficientDownLayer, self).__init__()
        bn_channel = channel // 2
        self.conv1 = nn.Sequential(
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                       ConvLayer(bn_channel, bn_channel, 3, 1, 1),
                       ConvLayer(bn_channel, bn_channel, 3, 2, 1),
                     )
        self.conv2 = nn.Sequential(
                       ConvLayer(channel, bn_channel, 1, 1, 0),
                       ConvLayer(bn_channel, bn_channel, 3, 2, 1),
                     )
        self.pool = nn.AvgPool2d(3, 2, 1)

    def forward(self, inputs):
        outputs = torch.cat([self.conv1(inputs), self.conv2(inputs), self.pool(inputs)], 1)
        return outputs


class DeconvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, act=F.leaky_relu):
        super(DeconvLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,
                                         out_channels=out_channel,
                                         kernel_size=kernel_size, 
                                         stride=stride,
                                         padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, inputs):
        outputs = self.bn(self.deconv(inputs))
        return self.act(outputs) if self.act else outputs


class BaseVAE(nn.Module):
    def __init__(self, img_size, z_dim):
        super(BaseVAE, self).__init__()

        self.img_size = img_size
        self.z_dim = z_dim

        down_factory = 4
        self.conv_output_size = [64 * (2 ** down_factory)] + [size // (2 ** down_factory) for size in img_size[-2:]]
        print(self.conv_output_size)

        self.enc_block = nn.Sequential(OrderedDict([
                            ("conv_layer_00", ConvLayer(img_size[0], 64, 3, 1, 1)),
                            ("conv_layer_01", ConvLayer(64, 64, 3, 1, 1)),
                            ("down_layer_02", ConvLayer(64, 128, 3, 2, 1)),
                            ("inception_v1_layer_10", ConvInceptionLayerV1(128)),
                            ("inception_v1_layer_11", ConvInceptionLayerV1(128)),
                            ("efficient_down_layer_12", EfficientDownLayer(128)),
                            ("inception_v2_layer_20", ConvInceptionLayerV2(256)),
                            ("inception_v2_layer_21", ConvInceptionLayerV2(256)),
                            ("efficient_down_layer_22", EfficientDownLayer(256)),
                            ("inception_v2_layer_30", ConvInceptionLayerV2(512)),
                            ("inception_v2_layer_31", ConvInceptionLayerV2(512)),
                            ("efficient_down_layer_32", EfficientDownLayer(512)),
                            ("view_layer", ViewLayer([-1, np.prod(self.conv_output_size)])),
                        ]))

        self.enc_fc = nn.Sequential(OrderedDict([
                        ('enc_fc_layer_0', nn.Linear(np.prod(self.conv_output_size), 256)),
                        ('relu_0', nn.LeakyReLU(inplace=True)),
                        ('enc_fc_layer_1', nn.Linear(256, self.z_dim * 2))
                      ]))

        self.dec_fc = nn.Sequential(OrderedDict([
                        ('dec_fc_layer_0', nn.Linear(self.z_dim, 256)),
                        ('relu_0', nn.LeakyReLU(inplace=True)),
                        ('dec_fc_layer_1', nn.Linear(256, np.prod(self.conv_output_size))),
                        ('relu_1', nn.LeakyReLU(inplace=True)),
                      ]))

        self.dec_block = nn.Sequential(OrderedDict([
                            ("view_layer", ViewLayer([-1] + self.conv_output_size)),
                            ("upsample_layer_00", nn.Upsample(scale_factor=2)),
                            ("upsample_layer_01", ConvLayer(1024, 512, 1, 1, 0)),
                            ("inception_v2_layer_02", ConvInceptionLayerV2(512)),
                            ("inception_v2_layer_03", ConvInceptionLayerV2(512)),
                            ("upsample_layer_10", nn.Upsample(scale_factor=2)),
                            ("upsample_layer_11", ConvLayer(512, 256, 1, 1, 0)),
                            ("inception_v2_layer_12", ConvInceptionLayerV2(256)),
                            ("inception_v2_layer_13", ConvInceptionLayerV2(256)),
                            ("upsample_layer_20", nn.Upsample(scale_factor=2)),
                            ("upsample_layer_21", ConvLayer(256, 128, 1, 1, 0)),
                            ("inception_v1_layer_22", ConvInceptionLayerV2(128)),
                            ("inception_v1_layer_23", ConvInceptionLayerV1(128)),
                            ("upsample_layer_30", nn.Upsample(scale_factor=2)),
                            ("upsample_layer_31", ConvLayer(128, 64, 1, 1, 0)),
                            ("conv_layer_32", ConvLayer(64, 64, 3, 1, 1)),
                            ("conv_layer_33", ConvLayer(64, 64, 3, 1, 1)),
                            ("conv_layer_4", ConvLayer(64, img_size[0], 1, 1, 0, act=None)),
                        ]))

        self.apply(init_weights)

        #avoiding log_var exponentially increase
        init.uniform_(self.enc_fc.enc_fc_layer_1.weight, -0.02, 0.02)

    def encoder(*args, **kwargs):
        pass

    def decoder(*args, **kwargs):
        pass

    def reparam(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, *args, **kwargs):
        pass


class VAE(BaseVAE):
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

    def encoder(self, img):
        outputs = self.enc_block(img)
        outputs = self.enc_fc(outputs)
        mu, log_var = outputs[:, :self.z_dim], outputs[:, self.z_dim:]
        return mu, log_var

    def decoder(self, z):
        outputs = self.dec_fc(z)
        logits_outputs = self.dec_block(outputs)
        outputs = torch.tanh(logits_outputs)
        return logits_outputs, outputs

    def forward(self, img):
        mu, log_var = self.encoder(img)
        z = self.reparam(mu, log_var)
        logits, outputs = self.decoder(z)
        return logits, outputs, mu, log_var


class CVAE(BaseVAE):
    def __init__(self, *args, **kwargs):
        super(CVAE, self).__init__(*args, **kwargs)

        self.enc_fc = nn.Sequential(OrderedDict([
                        ('enc_fc_layer_0', nn.Linear(np.prod(self.conv_output_size) * 2, 256)),
                        ('relu_0', nn.LeakyReLU(True)),
                        ('enc_fc_layer_1', nn.Linear(256, self.z_dim * 2))
                      ]))

        self.dec_fc = nn.Sequential(OrderedDict([
                        ('dec_fc_layer_0', nn.Linear(self.z_dim * 2, 256)),
                        ('leaky_relu_0', nn.LeakyReLU(True)),
                        ('dec_fc_layer_2', nn.Linear(256, np.prod(self.conv_output_size))),
                        ('leaky_relu_2', nn.LeakyReLU(True)),
                      ]))

        self.cond_fc = nn.Sequential(OrderedDict([
                        ('cond_fc_layer_0', nn.Linear(np.prod(self.conv_output_size), 256)),
                        ('leaky_relu_0', nn.LeakyReLU(True)),
                        ('cond_fc_layer_2', nn.Linear(256, self.z_dim)),
                        ('leaky_relu_2', nn.LeakyReLU(True)),
                      ]))
        self.enc_fc.apply(init_weights)
        self.dec_fc.apply(init_weights)
        self.cond_fc.apply(init_weights)
        init.uniform_(self.enc_fc.enc_fc_layer_1.weight, -0.02, 0.02)

    def encoder(self, img, cond):
        outputs = self.enc_block(img)
        cond_outputs = self.enc_block(cond)
        outputs = torch.cat([outputs, cond_outputs], 1)
        outputs = self.enc_fc(outputs)
        mu, log_var = outputs[:, :self.z_dim], outputs[:, self.z_dim:]
        return mu, log_var

    def decoder(self, z, cond):
        cond = self.enc_block(cond)
        cond_embed = self.cond_fc(cond.detach())
        z = torch.cat([z, cond_embed], 1)
        outputs = self.dec_fc(z)
        logits_outputs = self.dec_block(outputs)
        outputs = torch.tanh(logits_outputs)
        return logits_outputs, outputs

    def forward(self, img, cond):
        mu, log_var = self.encoder(img, cond)
        z = self.reparam(mu, log_var)
        logits, outputs = self.decoder(z, cond)
        return logits, outputs, mu, log_var


if __name__ == "__main__":
    #from torchsummary import summary

    #vae = VAE([3, 96, 96], 20).cuda()
    #vae = VAE([3, 48, 48], 20).cuda()

    cvae = CVAE([3, 64, 64], 20).cuda()
    #summary(vae.conv_block, (3, 32, 32))
    #summary(vae.conv_block, (3, 96, 96))
    #summary(vae.enc_fc, (1024*6*6,))
    #icp = ConvInceptionLayer(256).cuda()
    #edp = EfficientDownLayer(256).cuda()
    #dp = ConvLayer(256, 512, 3, 2, 1).cuda()
    #dp = nn.MaxPool2d:g(3, 2, 1).cuda()
    #a = nn.Sequential(icp, dp)
    #b = nn.Sequential(edp)
    #summary(a, (256, 16, 16))
    #summary(b, (256, 16, 16))
    #summary(vae.enc_block, (3, 96, 96))
    #summary(vae.dec_block, (20,))
    #summary(vae, (3, 48, 48))
    #summary(icp, (256, 32, 32))
    #summary(edp, (256, 32, 32))
    #summary(dp, (256, 32, 32))
    #summary(vae.dec_fc, (20,))
    #summary(vae.dec_block, (1024, 6, 6))
    #summary(cvae, [(3, 64, 64), (3, 64, 64)])
	

