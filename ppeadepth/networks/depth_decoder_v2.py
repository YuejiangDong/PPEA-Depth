# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from manydepth.layers import *
from manydepth.networks import conv_bn_relu
# from layers import *
# from replknet import conv_bn_relu

class Adapter(nn.Module):
    def __init__(self, D_features_in, D_features_out, adpt_test=0, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int((D_features_in + D_features_out) / 2 * mlp_ratio)
        self.act = act_layer()
        
        # self.D_fc1 = nn.Conv2d(D_features_in, D_hidden_features, kernel_size=3, stride=1, padding=1)
        
        self.D_fc1 = nn.Linear(D_features_in, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features_out)
        self.test_id = adpt_test

        # init weights:
        for n, m in self.named_modules():
            if 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                
            # nn.init.constant_(m.weight, 0)
            # nn.init.constant_(m.bias, 0)
        
        

    def forward(self, x):
        # x.shape = B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # x is expected (B, HW, C)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        # xs = xs.flatten(2).permute(0, 2, 1)
        
        xs = self.D_fc2(xs)
        x = xs
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        return x
class Adapter_(nn.Module):
    def __init__(self, D_features_in, D_features_out, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        self.act = act_layer()
        
        self.D_fc1 = nn.Linear(D_features_in, D_features_out)
        # self.D_fc2 = #nn.ConvTranspose2d(D_features_out//2, D_features_out, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.D_fc
        # init weights:
        for n, m in self.named_modules():
            if 'D_fc1' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # x.shape = B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # x is expected (B, HW, C)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = xs.permute(0, 2, 1).view(B, -1, H, W)
        xs = upsample(xs)
        return xs



class DepthDecoderV2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), debug=False, num_output_channels=1, use_skips=True, dc=False, test_id=1):
        super(DepthDecoderV2, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        
        base_ch = num_ch_enc[0] // 4
        self.ch_in_disp = np.array([base_ch * 2 ** i for i in range(4)])
        # print("debug ", self.ch_in_disp)
        
        # self.ch_in_disp = np.array([32, 64, 128, 256])

        self.debug = debug

        # decoder
        self.upconvs_0 = nn.ModuleList()
        self.upconvs_1 = nn.ModuleList()
        for i in range(3, -1, -1):
            ch_in = num_ch_enc[i]
            ch_out = self.num_ch_enc[i] // 2

            self.upconvs_0.append(ConvBlock(ch_in, ch_out))
            if i == 0:
                ch_in = ch_out
            self.upconvs_1.append(ConvBlock(ch_in, ch_out))
        
        add_ch_0 = num_ch_enc[0] // 2
        add_ch_1 = add_ch_0 // 2 
        assert(add_ch_0 == base_ch * 2)
        assert(add_ch_1 == base_ch)

        self.upconvs_0.append(ConvBlock(add_ch_0, add_ch_1))
        self.upconvs_1.append(ConvBlock(add_ch_1, add_ch_1))
        

        self.disp_convs = nn.ModuleList()
        # sclm=0
        self.disp_convs.append(Conv3x3(self.ch_in_disp[0], self.num_output_channels))
        # for s in self.scales:
        #     self.disp_convs.append(Conv3x3(self.ch_in_disp[s], self.num_output_channels))

        self.sigmoid = nn.Sigmoid()
        
        # options for scale finetuning
        self.dc = dc
        self.test_id = test_id
        
        if self.dc:
            # # design 1: a general adapter 
            if test_id == 1 or test_id == 5 or test_id ==6 or test_id == 7:
                self.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.ch_in_disp[0])
                self.deconv_adpt = nn.ConvTranspose2d(self.ch_in_disp[0], self.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            # design 2: more input info 
            elif test_id == 2:
                self.adapter = Adapter(num_ch_enc[3]+num_ch_enc[2]+num_ch_enc[1]+num_ch_enc[0], self.ch_in_disp[0])
                self.deconv_adpt = nn.ConvTranspose2d(self.ch_in_disp[0], self.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            # design 3: only deepest feature
            elif test_id == 3:
                self.adapter = Adapter(num_ch_enc[-1], self.ch_in_disp[0])
                self.deconv_adpt = nn.ConvTranspose2d(self.ch_in_disp[0], self.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            elif test_id == 4:
                self.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.ch_in_disp[0])
                self.deconv_adpt = nn.ConvTranspose2d(self.ch_in_disp[0], self.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
                self.deconv_adpt2 = nn.ConvTranspose2d(self.ch_in_disp[0], self.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            elif test_id == 8:
                self.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.ch_in_disp[0])
                
            elif test_id == 10:
                self.adapters = nn.ModuleList()
                for i in range(3):
                    self.adapters.append(Adapter_(self.num_ch_enc[3-i],self.num_ch_enc[2-i]))#,3,2,1))
                self.adapters.append(Adapter_(self.num_ch_enc[0],self.num_ch_enc[0]//2))#,3,2,1))
                
            # elif test_id == 11:
            #     self.adapters = nn.ModuleList()
            #     for i in range(3):
            #         self.adapters.append(Adapter_())
                 
        for n, m in self.named_modules():
            if 'deconv_adpt' in n:
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        if self.dc:
            # design 1
            if self.test_id == 1 or self.test_id == 4 or self.test_id == 5 or self.test_id == 6 or self.test_id == 7:
                x_up = nn.functional.interpolate(x, scale_factor=8, mode='nearest')
                adpt_out = self.deconv_adpt(self.adapter(torch.cat([input_features[0], x_up], 1)))
            # design 2
            elif self.test_id ==2:
                x_3 = nn.functional.interpolate(x, scale_factor=8, mode='nearest')
                x_2 = nn.functional.interpolate(input_features[-2], scale_factor=4, mode='nearest')
                x_1 = nn.functional.interpolate(input_features[1], scale_factor=2, mode='nearest')
                adpt_out = self.deconv_adpt(self.adapter(torch.cat([input_features[0], x_3, x_2, x_1], 1)))
            
            # design 3
            elif self.test_id == 3:
                x_up = nn.functional.interpolate(x, scale_factor=8, mode='nearest')
                adpt_out = self.deconv_adpt(self.adapter(x_up))
            
            # design 4
            # elif self.test_id == 4:
            #     x_up = nn.functional.interpolate(x, scale_factor=8, mode='nearest')
            #     adpt_out = self.deconv_adpt(self.adapter(x_up))
            elif self.test_id == 8:
                x_up = nn.functional.interpolate(x, scale_factor=8, mode='nearest')
                adpt_out = upsample(self.adapter(torch.cat([input_features[0], x_up], 1)))
            
            # print(adpt_out.shape)
        for i in range(4):
            # print("0 ", x.shape)
            if self.dc and self.test_id >= 10:
                adpt_out = self.adapters[i](x)
            x = self.upconvs_0[i](x)
            # if self.debug:
            # print("1 ", x.shape)
            x = [upsample(x)]
            if i < 3:
                x += [input_features[2 - i]]
            x = torch.cat(x, 1)
            
            x = self.upconvs_1[i](x)
            # print(x.shape)
            if self.dc and self.test_id >= 10:
                # print(x.shape, adpt_out.shape)
                x = x + 0.01*adpt_out
            
            # sclm = 0
            # if i > 0:
            #     self.outputs[("disp", 4-i)] = self.sigmoid(self.disp_convs[4-i](x))

        x = upsample(self.upconvs_0[-1](x))
        # print("upsample" , x.shape)
        x = self.upconvs_1[-1](x)
        # print(x.shape)
        if self.dc:
            if self.test_id < 4 or self.test_id == 5 or self.test_id == 6 or self.test_id == 7 or self.test_id == 8:
                adpt_out = nn.functional.interpolate(adpt_out, scale_factor=2)
                x = x + adpt_out
                
            elif self.test_id == 4:
                adpt_out = self.deconv_adpt2(adpt_out)
                x = x + adpt_out
            
        self.outputs[("disp", 0)] = self.sigmoid(self.disp_convs[0](x))
        
        # if self.dc:
        #     scale = self.sigmoid(self.scale_head(input_features[-1]).flatten(1).mean(dim=1))
        #     self.outputs[("scale", 0)] = scale
        
        return self.outputs

if __name__ == "__main__":
    # x = [torch.randn(2, 8, 3, 4)]
    x = [torch.randn(2, 64, 48, 160), torch.randn(2, 128, 24, 80), torch.randn(2, 256, 12, 40), torch.randn(2, 512, 6, 20)]
    model = DepthDecoderV2([64, 128, 256, 512], dc=True, test_id=8)
    y = model(x)
