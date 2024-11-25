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
from .resnet_encoder import ResnetEncoder
from .layers import transformation_from_parameters, disp_to_depth
from .pose_cnn import PoseCNN


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
            if self.opt.dec_only:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False           
        
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
                raise NotImplementedError
        
        if self.opt.rep_size == 'b':
            replk_path = "./pretrained/RepLKNet-31B_ImageNet-1K_224.pth"
            num_ch_enc = np.array([128, 256, 512, 1024]) # Base model
            
        elif self.opt.rep_size == 'l':
            replk_path = "./pretrained/RepLKNet-31L_ImageNet-22K.pth"
            num_ch_enc = np.array([192, 384, 768, 1536]) # Large Model
        else:
            raise NotImplementedError
            
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
            
                
        if self.opt.lps2:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.mono_encoder.named_parameters():
                param.requires_grad = False
                  
        # posenet
        self.need_pose_dec = False
        
        if self.opt.pose_cnn:
            self.pose_encoder = PoseCNN(num_input_frames=3)
        else:
            self.pose_encoder = ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                num_input_images=2)
            self.need_pose_dec = True
        
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
        
        
    def dc_ft_init(self, adpt=True):
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
        
        
    def load_drop_path(self, depthbin_tracker, pretrained_folder='./ckpt/blkc3_s57000'): 
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
            pretrained_path = "./pretrained/RepLKNet-31B_ImageNet-1K_224.pth"
        elif self.opt.rep_size == 'l':
            pretrained_path = "./pretrained/RepLKNet-31L_ImageNet-22K.pth"
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
                    
                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0

                inputs[('relative_pose', fi)] = pose
                
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
        
        self.device = inputs[('color_aug', 0, 0)].device
        
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
            img_aug = inputs["color_aug", 0, 0]
            feats = self.mono_encoder(img_aug)
            mono_outputs.update(self.mono_depth(feats))
        else:
            with torch.no_grad():
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
        
        
        _, teacher_depth = disp_to_depth(mono_outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)

        
        features, lowest_cost, confidence_mask = self.encoder(
                                            inputs["color_aug", 0, 0],
                                            lookup_frames,
                                            relative_poses,
                                            inputs[('K', 2)],
                                            inputs[('inv_K', 2)],
                                            min_depth_bin=min_depth_bin,
                                            max_depth_bin=max_depth_bin)
                                            
        
        outputs.update(self.depth(features))
        
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                            [self.opt.height, self.opt.width],
                                            mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]
        
        
        
        return mono_outputs, outputs
