import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

from .replk_matching_adapter import RepLKMatchingAdapter
from .replk_matching import RepLKMatching
from .replknet import create_RepLKNet31B, RepLKNet, create_RepLKNet31L
from .replknet_adapter import create_RepLKNet31B_Adapter, RepLKNetAdapter, create_RepLKNet31L_Adapter, create_RepLKNetXL_Adapter
from .depth_decoder_v2 import DepthDecoderV2, Adapter, Adapter_
from .pose_decoder import PoseDecoder
from .resnet_encoder import ResnetEncoder, ResnetEncoderDYJ
from .layers import transformation_from_parameters, disp_to_depth
from .rigid_warp import forward_warp
from .replknet_pose import RepLKPose
from .pose_cnn import PoseCNN
from .pose_vit import PoseViT
from .pose_rep import PoseRep, PoseRepAdapter


class RepDepth(nn.Module):
    def __init__(self, opt):
        super(RepDepth, self).__init__()
        
        self.opt = opt
        
        if opt.adapter:
            self.encoder = RepLKMatchingAdapter(self.opt.rep_size, self.opt.use_checkpoint, self.opt.trans, self.opt.input, self.opt.adpt_test,
                                                g_blk=self.opt.g_blk, g_ffn=self.opt.g_ffn, ratio=self.opt.ratio,
                                                adaptive_bins=not self.opt.notadabins, min_depth_bin=0.1, max_depth_bin=20.0,
                                                depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        else:
            self.encoder = RepLKMatching(self.opt.rep_size, self.opt.use_checkpoint,
                                        input_height=self.opt.height, input_width=self.opt.width,
                                        adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                                        depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)

        if self.opt.rep_size == 'b':
            num_ch_enc = np.array([128, 256, 512, 1024]) # Base model
        elif self.opt.rep_size == 'l':
            num_ch_enc = np.array([192, 384, 768, 1536]) # Large Model
        else:
            num_ch_enc = np.array([256, 512, 1024, 2048]) # Large Model
        
        self.depth = DepthDecoderV2(
            num_ch_enc, self.opt.scales, self.opt.debug, dc=self.opt.dc, test_id=self.opt.dec_id
        )
        
        if self.opt.adapter and not self.opt.fullft_reb:
            for name, param in self.encoder.named_parameters():
                if 'adpt' not in name and 'adapter' not in name and 'reduce' not in name and 'bn' not in name:
                    param.requires_grad = False
                if self.opt.dc:
                    if self.opt.dec_id == 5:
                        if 'adapter' in name:
                            if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.3' in name:
                                pass
                            else:
                                param.requires_grad = False
                    elif self.opt.dec_id == 6:
                        if 'adapter' in name:
                            if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.3' in name or '3.blocks.2' in name or '2.blocks.34' in name or '1.blocks.2' in name or '0.blocks.2' in name:
                                pass
                            else:
                                param.requires_grad = False
                    # elif self.opt.dec_id == 8:
                    #     if 'adapter' in name:
            if self.opt.dec_only:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False           
            if self.opt.dc and self.opt.dec_id == 7:
                # 1/3
                # disable_id = [0, 1, 3, 5, 7, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 40, 41, 42, 43, 46]# [3, 5, 7, 12, 15, 16, 17, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 41, 42, 43]# [0, 7, 16, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 41, 43, 45, 47]
                # 1/2
                disable_id = [3, 5, 7, 12, 15, 16, 17, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 41, 42, 43]
                # 2/3 
                # disable_id = [3, 5, 7, 12, 16, 20, 23, 25, 27, 29, 33, 35, 37, 39, 41, 43]
                
                blk_cnter = 0
                # try drop path rate zero
                for i, layer in enumerate(self.encoder.replk.stages):
                    for j, blk in enumerate(layer.blocks):
                        if blk_cnter in disable_id:
                            # blk.test_id = -1
                            for name, param in blk.named_parameters():
                                if 'adapter' in name:
                                    param.requires_grad = False
                        blk_cnter += 1
                        
                            
            # if self.opt.dc and self.opt.dec_id == 8:
            #     enable_id = [2, 3, 6, 7, 26, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41, 42, 43, 46, 47]
            #     blk_cnter = 0
            #     # try drop path rate zero
            #     for i, layer in enumerate(self.encoder.replk.stages):
            #         for j, blk in enumerate(layer.blocks):
            #             if blk_cnter not in enable_id:
            #                 # blk.test_id = -1
            #                 for name, param in blk.named_parameters():
            #                     if 'adapter' in name:
            #                         param.requires_grad = False
            #             blk_cnter += 1
                        
            #     # if self.opt.adpt_test == 8:
            #     #     if 'small_conv' in name:
            #     #         param.requires_grad = True
        
        
        if self.opt.adapter:
            if self.opt.rep_size == 'b':
                mono_enc_class = create_RepLKNet31B_Adapter
            elif self.opt.rep_size == 'l':
                mono_enc_class = create_RepLKNet31L_Adapter
            else:
                mono_enc_class = create_RepLKNetXL_Adapter
        else:
            if self.opt.rep_size == 'b':
                mono_enc_class = create_RepLKNet31B
            elif self.opt.rep_size == 'l':
                mono_enc_class = create_RepLKNet31L
            else:
                mono_enc_class = create_RepLKNetXL
        
        if self.opt.rep_size == 'b':
            replk_path = "RepLKNet-31B_ImageNet-1K_224.pth"
            num_ch_enc = np.array([128, 256, 512, 1024]) # Base model
            
        elif self.opt.rep_size == 'l':
            replk_path = "RepLKNet-31L_ImageNet-22K.pth"
            num_ch_enc = np.array([192, 384, 768, 1536]) # Large Model
        else:
            replk_path = "RepLKNet-XL_MegData73M_pretrain.pth"
            num_ch_enc = np.array([256, 512, 1024, 2048]) # XLarge Model
            
        if self.opt.adapter:
            mono_enc_opts = dict(
                drop_path_rate=0.3,
                num_classes=None,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=self.opt.use_checkpoint,
                small_kernel_merged=False,
                pretrained=replk_path,
                use_sync_bn=False,
                g_blk=self.opt.g_blk, g_ffn=self.opt.g_ffn, ratio=self.opt.ratio, trans_adpt=self.opt.mono_trans, input_adpt=self.opt.mono_input, adpt_test=self.opt.adpt_test
            )
        else:
            mono_enc_opts = dict(
                drop_path_rate=0.3,
                num_classes=None,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=self.opt.use_checkpoint,
                small_kernel_merged=False,
                pretrained=replk_path,
                use_sync_bn=False
            )
        
        self.mono_encoder = mono_enc_class(**mono_enc_opts)
            
        # num_ch_enc = np.array([128, 256, 512, 1024]) # Base Model
        self.mono_depth = DepthDecoderV2(
            num_ch_enc, self.opt.scales, self.opt.debug, dc=self.opt.dc, test_id=self.opt.dec_id
        )
        if self.opt.adapter and not self.opt.fullft_reb:
            for name, param in self.mono_encoder.named_parameters():
                if 'adpt' not in name and 'adapter' not in name and 'bn' not in name:
                    param.requires_grad = False
                if self.opt.dc:
                    if self.opt.dec_id == 5:
                        if 'adapter' in name:
                            if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.3' in name:
                                pass
                            else:
                                param.requires_grad = False
                    elif self.opt.dec_id == 6:
                        if 'adapter' in name:
                            if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.3' in name or '3.blocks.2' in name or '2.blocks.34' in name or '1.blocks.2' in name or '0.blocks.2' in name:
                                pass
                            else:
                                param.requires_grad = False
            if self.opt.dec_only:
                for name, param in self.mono_encoder.named_parameters():
                    param.requires_grad = False
            if self.opt.dc and self.opt.dec_id == 7:
                # 1/3
                # disable_id = [0, 1, 3, 5, 7, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 40, 41, 42, 43, 46]# [3, 5, 7, 12, 15, 16, 17, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 41, 42, 43]# [0, 7, 16, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 41, 43, 45, 47]
                
                disable_id = [3, 5, 7, 12, 15, 16, 17, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 39, 41, 42, 43]
                # 2/3 
                # disable_id = [3, 5, 7, 12, 16, 20, 23, 25, 27, 29, 33, 35, 37, 39, 41, 43]
                
                blk_cnter = 0
                # try drop path rate zero
                for i, layer in enumerate(self.mono_encoder.stages):
                    for j, blk in enumerate(layer.blocks):
                        if blk_cnter in disable_id:
                            # blk.test_id = -1
                            for name, param in blk.named_parameters():
                                if 'adapter' in name:
                                    param.requires_grad = False
                        blk_cnter += 1
                        
                            
            # if self.opt.dc and self.opt.dec_id == 8:
            #     enable_id = [2, 3, 6, 7, 26, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41, 42, 43, 46, 47]
            #     blk_cnter = 0
            #     # try drop path rate zero
            #     for i, layer in enumerate(self.mono_encoder.stages):
            #         for j, blk in enumerate(layer.blocks):
            #             if blk_cnter not in enable_id:
            #                 # blk.test_id = -1
            #                 for name, param in blk.named_parameters():
            #                     if 'adapter' in name:
            #                         param.requires_grad = False
            #             blk_cnter += 1
            
            # if self.opt.dc and self.opt.dec_id == 7:
            #     disable_id = [3, 9, 12, 15, 17, 19, 20, 21, 22, 23, 25, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 42, 45, 47]
            #     blk_cnter = 0
            #     # try drop path rate zero
            #     for i, layer in enumerate(self.mono_encoder.stages):
            #         for j, blk in enumerate(layer.blocks):
            #             if blk_cnter in disable_id:
            #                 blk.test_id = -1
            #                 for name, param in blk.named_parameters():
            #                     param.requires_grad = False
            #             blk_cnter += 1
                
        if self.opt.lps2:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.mono_encoder.named_parameters():
                param.requires_grad = False
                  
        # posenet
        self.need_pose_dec = False
        
        if self.opt.pose_attn_adpt:
            self.pose_encoder = PoseRepAdapter(3, "res", 0.1)
        elif self.opt.pose_attn:
            self.pose_encoder = PoseRep(3, self.opt.rep_size)
        elif self.opt.pose_vit:
            self.pose_encoder = PoseViT(self.opt.height, self.opt.width, 3, self.opt.vit_size)
        elif self.opt.pose_replk:
            self.pose_encoder = RepLKPose(self.opt.rep_size, False,
                                        self.opt.trans, self.opt.input, self.opt.adpt_test,
                                        g_blk=self.opt.g_blk, g_ffn=self.opt.g_ffn)
            self.need_pose_dec = True
        
        elif self.opt.pose_test:
            self.pose_encoder = ResnetEncoderDYJ(18, self.opt.weights_init == "pretrained",
                                num_input_images=3)
        elif self.opt.pose_cnn:
            self.pose_encoder = PoseCNN(num_input_frames=3)
        else:
            self.pose_encoder = ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                num_input_images=2)
            self.need_pose_dec = True
        # if self.opt.adapter:
        #     if self.opt.pose_attn:
        #         pass
        #     elif self.opt.pose_attn_adpt or self.opt.pose_cnn:
        #         pass
        #         # for name, param in self.pose_encoder.named_parameters():
        #         #     if 'replk' not in name:
        #         #         continue
        #         #     if 'adpt' not in name and 'adapter' not in name and 'bn' not in name:
        #         #         param.requires_grad = False
        #     elif self.need_pose_dec and not self.opt.pose_replk:
        #         pass
        #     else:
        #         for name, param in self.pose_encoder.named_parameters():
        #             if 'adpt' not in name and 'adapter' not in name and 'bn' not in name and 'temporal_embedding' not in name and 'ln_post' not in name:
        #                 param.requires_grad = False
           
        if self.need_pose_dec:
            self.pose = PoseDecoder(self.pose_encoder.num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)
        
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
        
        self.freeze_tp = False
        self.freeze_pose = False
        self.dc = self.opt.dc
        
        if self.opt.perf:
            disable_id = [7, 20, 21, 23, 24, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 47]

            # disable_id = [0, 7, 16, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 41, 43, 45, 47]# [0, 7, 16, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 41, 43, 45, 47]
            # [0, 1, 2, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47]# [3, 9, 12, 15, 17, 19, 20, 21, 22, 23, 25, 27, 29, 30, 31, 32, 33, 35, 37, 39, 40, 42, 45, 47]#[0, 3, 12, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47]# [0, 1, 3, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47]
            blk_cnter = 0
            # try drop path rate zero
            for i, layer in enumerate(self.encoder.replk.stages):
                    for j, blk in enumerate(layer.blocks):
                        if blk_cnter in disable_id: # != 46 and blk_cnter != 4 and blk_cnter != 5:#in disable_id:
                            blk.test_id = -1
                            for name, param in blk.named_parameters():
                                param.requires_grad = False
                        blk_cnter += 1
                        
            blk_cnter = 0
            disable_id_mono = [1, 3, 9, 11, 17, 18, 19, 21, 23, 24, 25, 26, 28, 29, 40, 42]
    
            for i, layer in enumerate(self.mono_encoder.stages):
                    for j, blk in enumerate(layer.blocks):
                        if blk_cnter in disable_id_mono:
                            blk.test_id = -1
                            for name, param in blk.named_parameters():
                                if 'adapter' in name:
                                    param.requires_grad = False
                        blk_cnter += 1
    def dc_ft_init(self, adpt=True):
        # self.dc = True
        
        if adpt==False:
            return
        
        if self.opt.rep_size == 'b':
            num_ch_enc = np.array([128, 256, 512, 1024]) # Base model
        elif self.opt.rep_size == 'l':
            num_ch_enc = np.array([192, 384, 768, 1536]) # Large Model
        else:
            num_ch_enc = np.array([256, 512, 1024, 2048]) # Large Model
        
        print("dc check ", self.depth.dc, self.mono_depth.dc)
        # temporarily
        self.depth.dc = True
        self.depth.test_id = self.opt.dec_id
        
        # nn.init.constant_(self.depth.deconv_adpt2.weight, 0)
        # nn.init.constant_(self.depth.deconv_adpt2.bias, 0)
        
        # temporarily
        self.mono_depth.dc = True
        self.mono_depth.test_id = self.opt.dec_id
        
        # design 1:
        if self.opt.dec_id == 1 or self.opt.dec_id == 5 or self.opt.dec_id ==6:
            self.depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0], mlp_ratio=self.opt.dec_ratio)
            self.depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.mono_depth.ch_in_disp[0], mlp_ratio=self.opt.dec_ratio)
            self.mono_depth.deconv_adpt = nn.ConvTranspose2d(self.mono_depth.ch_in_disp[0], self.mono_depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # design 2: more input info 
        elif self.opt.dec_id == 2:
            self.depth.adapter = Adapter(num_ch_enc[3]+num_ch_enc[2]+num_ch_enc[1]+num_ch_enc[0], self.depth.ch_in_disp[0])
            self.depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.adapter = Adapter(num_ch_enc[3]+num_ch_enc[2]+num_ch_enc[1]+num_ch_enc[0], self.mono_depth.ch_in_disp[0])
            self.mono_depth.deconv_adpt = nn.ConvTranspose2d(self.mono_depth.ch_in_disp[0], self.mono_depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # design 3:
        elif self.opt.dec_id == 3:
            self.depth.adapter = Adapter(num_ch_enc[-1], self.depth.ch_in_disp[0])
            self.depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.adapter = Adapter(num_ch_enc[-1], self.mono_depth.ch_in_disp[0])
            self.mono_depth.deconv_adpt = nn.ConvTranspose2d(self.mono_depth.ch_in_disp[0], self.mono_depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # design 4:
        elif self.opt.dec_id ==4:
            self.depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0])
            self.depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.depth.deconv_adpt2 = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.mono_depth.ch_in_disp[0])
            self.mono_depth.deconv_adpt = nn.ConvTranspose2d(self.mono_depth.ch_in_disp[0], self.mono_depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.deconv_adpt2 = nn.ConvTranspose2d(self.mono_depth.ch_in_disp[0], self.mono_depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        elif self.opt.dec_id == 7:
            self.depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0])
            self.depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.deconv_adpt = nn.ConvTranspose2d(self.depth.ch_in_disp[0], self.depth.ch_in_disp[0], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.mono_depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0])
        
        elif self.opt.dec_id == 8:
            self.depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0])
            self.mono_depth.adapter = Adapter(num_ch_enc[-1]+num_ch_enc[0], self.depth.ch_in_disp[0])
        elif self.opt.dec_id == 10:
            self.depth.adapters = nn.ModuleList()
            self.mono_depth.adapters = nn.ModuleList()
            for i in range(3):
                self.depth.adapters.append(Adapter_(num_ch_enc[3-i],num_ch_enc[2-i]))#,3,2,1))
                self.mono_depth.adapters.append(Adapter_(num_ch_enc[3-i],num_ch_enc[2-i]))#,3,2,1))
            self.depth.adapters.append(Adapter_(num_ch_enc[0],num_ch_enc[0]//2))#,3,2,1))
            self.mono_depth.adapters.append(Adapter_(num_ch_enc[0],num_ch_enc[0]//2))#,3,2,1))
                # elif test_id == 11:
        #     self.depth.adapters = nn.ModuleList()
        #     self.mono_depth.adapters = nn.ModuleList()
        #     for i in range(3):
        #         self.depth.adapters.append(
        #             nn.Sequential(nn.Conv2d(self.depth.ch_in_disp[3-i],self.depth.ch_in_disp[2-i],3,2,1),
        #                             nn.GELU))
        #         self.mono_depth.adapters.append(
        #             nn.Sequential(nn.Conv2d(self.depth.ch_in_disp[3-i],self.depth.ch_in_disp[2-i],3,2,1),
        #                             nn.GELU))
                
        if self.opt.dec_id < 8:    
            nn.init.constant_(self.depth.deconv_adpt.weight, 0)
            nn.init.constant_(self.depth.deconv_adpt.bias, 0)
            nn.init.constant_(self.mono_depth.deconv_adpt.weight, 0)
            nn.init.constant_(self.mono_depth.deconv_adpt.bias, 0)
        
        # nn.init.constant_(self.mono_depth.deconv_adpt2.weight, 0)
        # nn.init.constant_(self.mono_depth.deconv_adpt2.bias, 0)
        
        
        # freeze parameters of depth and monodepth:
        for name, param in self.depth.named_parameters():
            if 'adpt' not in name and 'adapter' not in name:
                param.requires_grad = False
        for name, param in self.mono_depth.named_parameters():
            if 'adpt' not in name and 'adapter' not in name:
                param.requires_grad = False
        # for name, param in self.encoder.named_parameters():
        #     if 'adapter' in name:
        #         if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.0' in name:
        #             pass
        #             # param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #     else:
        #         param.requires_grad = False
                   
        # for name, param in self.mono_encoder.named_parameters():
        #     if 'adapter' in name:
        #         # if '3.blocks.3' in name or '2.blocks.35' in name or '1.blocks.3' in name or '0.blocks.0' in name:
        #         pass
        #             # param.requires_grad = True
        #         # else:
        #         #     param.requires_grad = False
        #     else:
        #         param.requires_grad = False
        
        ###### the code with bug
        # dict_load_depth = self.depth.state_dict().copy()
        # dict_load_depth_m = self.mono_depth.state_dict().copy()
        
        # self.depth = DepthDecoderV2(
        #     num_ch_enc, self.opt.scales, self.opt.debug, dc=True
        # )
        # self.mono_depth = DepthDecoderV2(
        #     num_ch_enc, self.opt.scales, self.opt.debug, dc=True
        # )
        
        # self.depth.load_state_dict(dict_load_depth, strict=False)
        # self.mono_depth.load_state_dict(dict_load_depth_m, strict=False)
    
    def cross_load_kitti(self, pretrained_folder='./ckpt/l_dprp3_s72000'):
        whole_model = torch.load(pretrained_folder+'/model.pth', map_location='cpu')
        self.load_state_dict(whole_model.state_dict(), strict=False)

    def load_drop_path_blank(self, pretrained_folder='./ckpt/blkc3_s57000'):
        if self.opt.adpt_test == 4:
            pretrained_folder = './ckpt/clcbfte6+_s15000'
        
        whole_encoder = torch.load(pretrained_folder+'/encoder.pth', map_location='cpu')
        # self.encoder.load_state_dict(whole_encoder.state_dict(), strict=False)
        
        whole_mono_encoder = torch.load(pretrained_folder+'/mono_encoder.pth', map_location='cpu')
        # self.mono_encoder.load_state_dict(whole_mono_encoder.state_dict(), strict=False)
        
        # plan b:
        # assign drop path of whole model to self.encoder
        for i, layer in enumerate(self.encoder.replk.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_encoder.replk.stages[i].blocks[j].drop_path
        
        # assign drop path of whole model to self.monoencoder
        for i, layer in enumerate(self.mono_encoder.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_mono_encoder.stages[i].blocks[j].drop_path
        
        # self.mono_encoder.trans_drop_path = whole_mono_encoder.trans_drop_path
        
        # depth_dict = torch.load(pretrained_folder+'/depth.pth', map_location='cpu')
        # self.depth.load_state_dict(depth_dict, strict=False)
        # self.mono_depth.load_state_dict(torch.load(pretrained_folder+'/mono_depth.pth', map_location='cpu'), strict=False)
        # self.pose_encoder.load_state_dict(torch.load(pretrained_folder+'/pose_encoder.pth', map_location='cpu'))
        # self.pose.load_state_dict(torch.load(pretrained_folder+'/pose.pth', map_location='cpu'))
        
        # min_depth_bin = depth_dict.get('min_depth_bin')
        # max_depth_bin = depth_dict.get('max_depth_bin')
        # print(min_depth_bin, max_depth_bin)
        # depthbin_tracker.load(min_depth_bin, max_depth_bin)
    def load_drop_path_l(self, 
                         drp_path = './ckpt/l_dprp3_1c_drp_path'):
        
        whole_encoder = torch.load(drp_path +'/encoder.pth', map_location='cpu')
        whole_mono_encoder = torch.load(drp_path +'/mono_encoder.pth', map_location='cpu')
        
        # assign drop path of whole model to self.encoder
        for i, layer in enumerate(self.encoder.replk.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_encoder.replk.stages[i].blocks[j].drop_path
        
        # assign drop path of whole model to self.monoencoder
        for i, layer in enumerate(self.mono_encoder.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_mono_encoder.stages[i].blocks[j].drop_path
        
        
    def load_drop_path(self, depthbin_tracker, pretrained_folder='./ckpt/blkc3_s57000'): #'../manydepth2_seg/ckpt/clcbt2_s8000'):
        if self.opt.adpt_test == 4:
            pretrained_folder = './ckpt/clcbfte6+_s15000'
        print("debug ", pretrained_folder)
        whole_encoder = torch.load(pretrained_folder+'/encoder.pth', map_location='cpu')
        self.encoder.load_state_dict(whole_encoder.state_dict(), strict=False)
        
        whole_mono_encoder = torch.load(pretrained_folder+'/mono_encoder.pth', map_location='cpu')
        self.mono_encoder.load_state_dict(whole_mono_encoder.state_dict(), strict=False)
        
        # assign drop path of whole model to self.encoder
        for i, layer in enumerate(self.encoder.replk.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_encoder.replk.stages[i].blocks[j].drop_path
        
        # assign drop path of whole model to self.monoencoder
        for i, layer in enumerate(self.mono_encoder.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_mono_encoder.stages[i].blocks[j].drop_path
        
        # self.mono_encoder.trans_drop_path = whole_mono_encoder.trans_drop_path
        depth_dict = torch.load(pretrained_folder+'/depth.pth', map_location='cpu')
        self.depth.load_state_dict(depth_dict, strict=False)
        self.mono_depth.load_state_dict(torch.load(pretrained_folder+'/mono_depth.pth', map_location='cpu'), strict=False)
        self.pose_encoder.load_state_dict(torch.load(pretrained_folder+'/pose_encoder.pth', map_location='cpu'))
        self.pose.load_state_dict(torch.load(pretrained_folder+'/pose.pth', map_location='cpu'))
        
        min_depth_bin = depth_dict.get('min_depth_bin')
        max_depth_bin = depth_dict.get('max_depth_bin')
        if depthbin_tracker is not None:
            print(min_depth_bin, max_depth_bin)
            depthbin_tracker.load(torch.Tensor([min_depth_bin]), torch.Tensor([max_depth_bin]))
        else:
            return torch.Tensor([min_depth_bin]), torch.Tensor([max_depth_bin])
       
    def load_pretrained_singlecard(self, pretrained='./ckpt/rep0van_s88000'):
        def update_dict(model_dict, pretrained_dict): 
            model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
            return model_dict
        
        encoder_path = os.path.join(pretrained, "encoder.pth")
        enc_full = torch.load(encoder_path)
        
        mono_enc_path = os.path.join(pretrained, "mono_encoder.pth")
        mono_enc_full = torch.load(mono_enc_path)
        
        depth_path = os.path.join(pretrained, "depth.pth")
        mono_depth_path = os.path.join(pretrained, "mono_depth.pth")
        pose_enc_path = os.path.join(pretrained, "pose_encoder.pth")
        pose_path = os.path.join(pretrained, "pose.pth")
        
        self.depth.load_state_dict(update_dict(self.depth.state_dict(), torch.load(depth_path)))
        self.mono_depth.load_state_dict(update_dict(self.mono_depth.state_dict(), torch.load(mono_depth_path)))
        self.pose_encoder.load_state_dict(update_dict(self.pose_encoder.state_dict(), torch.load(pose_enc_path)))
        self.pose.load_state_dict(update_dict(self.pose.state_dict(), torch.load(pose_path)))
        
        self.encoder.load_state_dict(update_dict(self.encoder.state_dict(), enc_full.state_dict()))
        self.mono_encoder.load_state_dict(update_dict(self.mono_encoder.state_dict(), mono_enc_full.state_dict()))
                    
    def load_pretrained(self):
        if self.opt.rep_size == 'b':
            pretrained_path = "../DSformer/RepLKNet-31B_ImageNet-1K_224.pth"
        elif self.opt.rep_size == 'l':
            pretrained_path = "../DSformer/RepLKNet-31L_ImageNet-22K.pth"
        pretrained_weights = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cpu())
        
        
        main_dict = self.encoder.state_dict()
        for k, v in main_dict.items():
            print(k, v)
            break
        for k, v in main_dict.items():
            main_state_dict = {'.'.join(['encoder.replk', k]) : v for k, v in pretrained_weights.items()}
            
        mono_dict = self.mono_encoder.state_dict()
        for k, v in mono_dict.items():
            mono_state_dict = {'.'.join(['mono_encoder', k]) : v for k, v in pretrained_weights.items()}
        
        self.encoder.load_state_dict(main_state_dict, strict=False)
        self.mono_encoder.load_state_dict(mono_state_dict, strict=False)
        
        for k, v in self.encoder.state_dict().items():
            print(k, v)
            break            
    
    def freeze_tp_net(self):
        for name, param in self.mono_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.mono_depth.named_parameters():
            param.requires_grad = False
        for name, param in self.pose_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.pose.named_parameters():
            param.requires_grad = False
        self.freeze_tp = True
        
        num_param = sum(p.numel() for p in self.mono_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_encoder.parameters())
        print("for mono_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.mono_depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_depth.parameters())
        print("for mono_depth ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose.parameters())
        print("for pose ", num_param, num_total_param)
    
    def freeze_pose_net(self):
        for name, param in self.pose_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.pose.named_parameters():
            param.requires_grad = False
        self.freeze_pose = True
        
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose.parameters())
        print("for pose ", num_param, num_total_param)
        
            
    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            # f_i = -1, 1
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation = self.pose(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                
                # if self.dc:
                #     if f_i < 0:
                #         # Tt->t-1
                #         outputs[("cam_T_cam", -1, 0)] = (transformation_from_parameters)(
                #             axisangle[:, 0], translation[:, 0], invert=False)
                #     else:
                #        # Tt->t+1
                #         outputs[("cam_T_cam", 1, 0)] = (transformation_from_parameters)(
                #             axisangle[:, 0], translation[:, 0], invert=True) 

        # now we need poses for matching - compute without gradients
        to_iter = self.matching_ids # if not self.dc else [0, -1, 1]
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in to_iter}
        with torch.no_grad():
            to_iter = self.matching_ids[1:] #if not self.dc else [-1, 1]
            
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in to_iter:
                # fi = -1
                if fi < 0:
                    pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.pose(pose_inputs)
                    pose = (transformation_from_parameters)(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    # print("for -1 ", axisangle[:, 0], translation[:, 0])
                    # if self.dc:
                    #     # Tt->t-1
                    #     pose_inv = (transformation_from_parameters)(
                    #         axisangle[:, 0], translation[:, 0], invert=False)
                    

                    # now find 0->fi pose
                    if fi != -1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.pose(pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    
                    # if self.dc:
                    #     # Tt->t+1
                    #     pose_inv = (transformation_from_parameters)(
                    #         axisangle[:, 0], translation[:, 0], invert=True)
                    

                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0
                        # if self.dc:
                        #     pose_inc[batch_idx] *= 0

                inputs[('relative_pose', fi)] = pose
                # if self.dc:
                #     inputs[('relative_pose_inv', fi)] = pose
                
        return outputs    
     
    def print_num_param(self):
        num_param = sum(p.numel() for p in self.mono_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_encoder.parameters())
        print("for mono_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.encoder.parameters())
        print("for encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.depth.parameters())
        print("for depth ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.mono_depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_depth.parameters())
        print("for mono_depth ", num_param, num_total_param)
    
        
    def forward(self, inputs, min_depth_bin, max_depth_bin):
        mono_outputs = {}
        outputs = {}
        
        # predict poses
        if self.freeze_tp == False and self.freeze_pose == False:
            # if self.opt.pose_cnn:
            #     pose_pred = self.predict_poses_cnn(inputs)
            # else:
            pose_pred = self.predict_poses_vit(inputs) if not self.need_pose_dec else self.predict_poses(inputs)
        else:
            with torch.no_grad():
                # if self.opt.pose_cnn:
                #     pose_pred = self.predict_poses_cnn(inputs)
                # else:
                pose_pred = self.predict_poses_vit(inputs) if not self.need_pose_dec else self.predict_poses(inputs)
                
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)
        
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
        
        # if self.dc:
        #     # to predict D_{t-1}
        #     relative_poses_inv = [inputs[('relative_pose_inv', idx)] for idx in self.matching_ids[1:]]
        #     relative_poses_inv = torch.stack(relative_poses_inv, 1)
        #     lookup_frames_inv = [inputs[('color_aug', 0, 0)]]
        #     lookup_frames_inv = torch.stack(lookup_frames_inv, 1)  # batch x frames x 3 x h x w
            
        #     # to predict D_{t+1}
        #     relative_poses_inv_nxt = [inputs[('relative_pose_inv', -1)]]
        #     relative_poses_inv_nxt = torch.stack(relative_poses_inv_nxt, 1)
        #     lookup_frames_inv_nxt = [inputs[('color_aug', 0, 0)]]
        #     lookup_frames_inv_nxt = torch.stack(lookup_frames_inv_nxt, 1)
            

        self.device = inputs[('color_aug', 0, 0)].device
        # if self.dc:
        #     batch_size = 3 * len(lookup_frames)
        #     augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()

        #     input_cat = torch.cat([inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]], dim=0)
        #     lookup_frames_cat = torch.cat([lookup_frames, lookup_frames_inv, lookup_frames_inv_nxt], dim=0)
        #     r_pose_cat = torch.cat([relative_poses, relative_poses_inv, relative_poses_inv_nxt], dim=0)
            
        #     for batch_idx in range(batch_size):
        #         rand_num = random.random()
                
        #         frame_seq = int(batch_idx / len(lookup_frames))
        #         frame_id = 0 if frame_seq == 0 else -1 if frame_seq == 1 else 1
                
        #         # static camera augmentation -> overwrite lookup frames with current frame
        #         if rand_num < 0.25:
        #             replace_frames = \
        #                 [input_cat[batch_idx] for _ in self.matching_ids[1:]]
        #             replace_frames = torch.stack(replace_frames, 0)
        #             lookup_frames_cat[batch_idx] = replace_frames
                    
        #             augmentation_mask[batch_idx] += 1
        #         # missing cost volume augmentation -> set all poses to 0, the cost volume will
        #         # skip these frames
        #         elif rand_num < 0.5:
        #             r_pose_cat[batch_idx] *= 0
        #             augmentation_mask[batch_idx] += 1
        
        # else:
        batch_size = len(lookup_frames)
        
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        # matching augmentation
        for batch_idx in range(batch_size):
            rand_num = random.random()
            # static camera augmentation -> overwrite lookup frames with current frame
            if rand_num < 0.25:
                replace_frames = \
                    [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                replace_frames = torch.stack(replace_frames, 0)
                lookup_frames[batch_idx] = replace_frames
                
                augmentation_mask[batch_idx] += 1
            # missing cost volume augmentation -> set all poses to 0, the cost volume will
            # skip these frames
            elif rand_num < 0.5:
                relative_poses[batch_idx] *= 0
                augmentation_mask[batch_idx] += 1
    
        outputs['augmentation_mask'] = augmentation_mask
        
        
        
        # predict by teacher network
        if self.freeze_tp == False:
            # if self.dc:
            #     img_aug = torch.cat([inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]], dim=0)
            # else:
            img_aug = inputs["color_aug", 0, 0]
            feats = self.mono_encoder(img_aug)
            mono_outputs.update(self.mono_depth(feats))
        else:
            with torch.no_grad():
                # if self.dc:
                #     img_aug = torch.cat([inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]], dim=0)
                # else:
                img_aug = inputs["color_aug", 0, 0]
                feats = self.mono_encoder(img_aug)
                mono_outputs.update(self.mono_depth(feats))
        
        # update multi frame outputs dictionary with single frame outputs
        # aim to compute consistency loss
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]
        
        if self.opt.dyn:
            ############# warpping image based on teacher model predicted pose and depth ###############
            with torch.no_grad():
                _, teacher_depth = disp_to_depth(mono_outputs["disp", 0].detach().clone(), self.opt.min_depth, self.opt.max_depth)  # [12, 1, 192, 512]
                teacher_depth =teacher_depth.detach().clone()
                tgt_imgs = inputs["color", 0, 0].detach().clone()  # [12, 3, 192, 512]
                m1_pose = outputs[("cam_T_cam", 0, -1)][:, :3, :].detach().clone()  # [12, 3, 4]
                intrins = inputs[('K', 0)][:,:3,:3]  # [12, 3, 3]
                doj_mask = inputs["doj_mask"]  # [12, 1, 192, 512]
                tgt_imgs[doj_mask.repeat(1,3,1,1)==0] = 0
                img_w_m1, _, _ = forward_warp(tgt_imgs, teacher_depth, m1_pose, intrins, upscale=3, rotation_mode='euler', padding_mode='zeros')
                doj_maskm1 = inputs["doj_mask-1"].repeat(1,3,1,1)  # [12, 3, 192, 512]
                if self.opt.no_teacher_warp:
                    inputs['ori_color', -1, 0] = inputs["color", -1, 0].detach().clone()
                inputs["color", -1, 0][doj_maskm1==1] = 0
                if not self.opt.no_reproj_doj:
                    inputs["color", -1, 0][img_w_m1>0] = img_w_m1[img_w_m1>0]
                else:
                    inputs["color", -1, 0][img_w_m1>0] = 0
                inputs["color", -1, 0] = inputs["color", -1, 0].detach().clone()

                non_cv_aug = [augmentation_mask[:,0,0,0]==0][0]  # [12]
                if non_cv_aug.sum() > 0:
                    tgt_imgs_aug = inputs["color_aug", 0, 0].detach().clone()  # [12, 3, 192, 512]
                    tgt_imgs_aug[doj_mask.repeat(1,3,1,1)==0] = 0
                    imgaug_w_m1, _, _ = forward_warp(tgt_imgs_aug[non_cv_aug], teacher_depth[non_cv_aug], m1_pose[non_cv_aug], intrins[non_cv_aug], upscale=3, rotation_mode='euler', padding_mode='zeros')
                    warp_frame = lookup_frames[non_cv_aug][:,0,:,:,:].detach().clone()
                    warp_frame[doj_maskm1[non_cv_aug]==1] = 0
                    warp_frame[imgaug_w_m1>0] = imgaug_w_m1[imgaug_w_m1>0]
                    lookup_frames[non_cv_aug] = warp_frame.unsqueeze(1).detach().clone()

                p1_pose = outputs[("cam_T_cam", 0, 1)][:, :3, :].detach().clone()  # [12, 3, 4]
                img_w_p1, _, _ = forward_warp(tgt_imgs, teacher_depth, p1_pose, intrins, upscale=3, rotation_mode='euler', padding_mode='zeros')
                doj_maskp1 = inputs["doj_mask+1"].repeat(1,3,1,1)  # [12, 3, 192, 512]
                if self.opt.no_teacher_warp:
                    inputs['ori_color', 1, 0] = inputs["color", -1, 0].detach().clone()
                inputs["color", 1, 0][doj_maskp1==1] = 0
                if not self.opt.no_reproj_doj:
                    inputs["color", 1, 0][img_w_p1>0] = img_w_p1[img_w_p1>0]
                else:
                    inputs["color", 1, 0][img_w_p1>0] = 0
                inputs["color", 1, 0] = inputs["color", 1, 0].detach().clone()
        ####################################################################################################
        
        _, teacher_depth = disp_to_depth(mono_outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)

        # predict by main net using multi frames
        # if not self.dc:
            # if self.opt.dyn:
            #     features, lowest_cost, confidence_mask = self.encoder(
            #                                         inputs["color_aug", 0, 0],
            #                                         lookup_frames,
            #                                         relative_poses,
            #                                         inputs[('K', 2)],
            #                                         inputs[('inv_K', 2)],
            #                                         min_depth_bin=min_depth_bin,
            #                                         max_depth_bin=max_depth_bin,
            #                                         teacher_depth=teacher_depth,
            #                                         doj_mask=inputs["doj_mask"],
            #                                         cv_min=self.opt.cv_min=='true',
            #                                         aug_mask=augmentation_mask,
            #                                         set_1=self.opt.cv_set_1,
            #                                         pool=self.opt.cv_pool,
            #                                         pool_r=self.opt.cv_pool_radius,
            #                                         pool_th=self.opt.cv_pool_th)
            # else:
        features, lowest_cost, confidence_mask = self.encoder(
                                            inputs["color_aug", 0, 0],
                                            lookup_frames,
                                            relative_poses,
                                            inputs[('K', 2)],
                                            inputs[('inv_K', 2)],
                                            min_depth_bin=min_depth_bin,
                                            max_depth_bin=max_depth_bin)
                                            
        # else:
        #     K_cat = torch.cat([inputs[('K', 2)], inputs[('K', 2)], inputs[('K', 2)]], dim=0)
        #     invk_cat = torch.cat([inputs[('inv_K', 2)], inputs[('inv_K', 2)], inputs[('inv_K', 2)]], dim=0)
            
        #     features_cat, lowest_cost_cat, confidence_mask_cat = self.encoder(
        #         input_cat, lookup_frames_cat, r_pose_cat,
        #         K_cat, invk_cat,
        #         min_depth_bin=min_depth_bin,
        #         max_depth_bin=max_depth_bin)
        #     bs = inputs[('K', 2)].shape[0]
        #     lowest_cost = lowest_cost_cat[:bs]
        #     confidence_mask = confidence_mask_cat[:bs]
        #     features = features_cat
        
        outputs.update(self.depth(features))
        
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                            [self.opt.height, self.opt.width],
                                            mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]
        
        
        
        return mono_outputs, outputs
