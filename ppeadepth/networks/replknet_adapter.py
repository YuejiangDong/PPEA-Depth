# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# Based on ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
import sys
import os
from typing import Optional

class Adapter(nn.Module):
    def __init__(self, D_features, adpt_test, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        self.feats = D_features
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        # if adpt_test == 2:
        #     self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=1, padding=1)
        #     self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=1, padding=1)
        # else:
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.test_id = adpt_test
        

    def forward(self, x):
        # x.shape = B, C, H, W
        # if self.test_id != 2:
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        # if self.test_id != 2:
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        return x

class B_Adapter(nn.Module):
    def __init__(self, D_features, adpt_test, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        self.feats = D_features
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        # print("in B_adapter ", adpt_test)
        if adpt_test == 1 or adpt_test == 2:
            self.D_fc1 = nn.Linear(D_features, D_hidden_features)
            self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        elif adpt_test == 4:
            self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=1, padding=1)
            # self.bn1 = nn.BatchNorm2d(D_hidden_features)
            self.D_fc2 = nn.Linear(D_hidden_features, D_features) #, kernel_size=3, stride=1, padding=1)
            
        # elif adpt_test == 4:
        #     self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=1, padding=1)
        #     self.bn1 = nn.BatchNorm2d(D_hidden_features)
        #     self.D_fc2 = nn.Linear(D_hidden_features, D_features) #, kernel_size=3, stride=1, padding=1)
        # elif adpt_test == 7:
        #     self.D_fc0 = nn.Linear(D_features, int(D_features*1.5))
        #     self.D_fc1 = nn.Linear(int(D_features*1.5), D_hidden_features) #, kernel_size=3, stride=1, padding=1)
        #     self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=1, padding=1)
        
        # elif adpt_test == 6:
        #     wn = lambda x: torch.nn.utils.weight_norm(x)
        #     self.D_fc1 = wn(nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=1, padding=1))
        #     self.D_fc2 = wn(nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=1, padding=1))
        
        else:  
            self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=1, padding=1)
            self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=1, padding=1)
        
        self.test_id = adpt_test
        # if adpt_test == 1:
        #     self.bn1 = nn.BatchNorm2d(D_hidden_features)
        #     self.bn2 = nn.BatchNorm2d(D_features)

    def forward(self, x):
        # x.shape = B, C, H, W
        B, C, H, W = x.shape
        if self.test_id == 1 or self.test_id == 2:
            x = x.flatten(2).permute(0, 2, 1)
            
        xs = self.D_fc1(x)
        if self.test_id == 4:
            # xs = self.bn1(xs)
            xs = xs.flatten(2).permute(0, 2, 1)
            
        xs = self.act(xs)
        # if self.test_id == 4:
        #     xs = self.D_fc2()
        #     xs = xs.permute(0, 2, 1).view(B, -1, H, W)
        # else:
        xs = self.D_fc2(xs)
        if self.test_id > 0:
            xs = xs.permute(0, 2, 1).view(B, -1, H, W)
            
        # if self.test_id == 1:
        #     xs = self.bn2(xs)
        return xs


class D_Adapter(nn.Module):
    def __init__(self, in_features, out_features, adpt_test, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        self.feats = in_features
        D_hidden_features = int(in_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(in_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, out_features)
        self.test_id = adpt_test
        

    def forward(self, x):
        # x.shape = B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs.permute(0, 2, 1).view(B, -1, H, W)
        return x
      
class InputAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, kernel_size=3, stride=2, padding=1)
        self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=1, padding=1)
        self.act = act_layer()
        self.bn1 = nn.BatchNorm2d(D_hidden_features)
        self.bn2 = nn.BatchNorm2d(D_features)
        

    def forward(self, x):
        xs = self.bn1(self.D_fc1(x))
        xs = self.act(xs)
        xs = self.bn2(self.D_fc2(xs))
        return xs

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path, gamma=1.0, adpt_test=0, ratio=0.25):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()
        mlp_ratio = 0.5 if adpt_test == 2 else 0.25
        
        if adpt_test >= 0:
            self.mlp_adapter = Adapter(D_features=in_channels, adpt_test=adpt_test, mlp_ratio=mlp_ratio)
        # else:
        #     self.mlp_adapter = Adapter(D_features=in_channels, adpt_test=adpt_test, mlp_ratio=0.1)
            
        self.gamma = gamma
        self.test_id = adpt_test

    def forward(self, x):
        out = self.preffn_bn(x)
        adpt_output = self.mlp_adapter(out) if self.test_id >= 0 else torch.zeros_like(out)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out) + self.gamma * adpt_output


class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, gamma=1.0, small_kernel_merged=False, adpt_test=0, ratio=0.25):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', drop_path)
        if adpt_test >= 0:
            # self.lk_adapter = B_Adapter(D_features=dw_channels, adpt_test=adpt_test)
            # self.ffn_adapter = D_Adapter(in_features=dw_channels, out_features=in_channels, adpt_test=adpt_test)
        # else:
            self.adapter = B_Adapter(D_features=in_channels, adpt_test=adpt_test, mlp_ratio=ratio)
        # else:
        #     self.adapter = B_Adapter(D_features=in_channels, adpt_test=adpt_test, mlp_ratio=0.1)
            
        self.gamma = gamma
        self.test_id = adpt_test

    def forward(self, x):
        out = self.prelkb_bn(x)
        adpt_out = self.adapter(out) if self.test_id >= 0 else 0
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        # if self.test_id == 8:
        #     adpt_out2 = self.ffn_adapter(out)
        #     out = out + self.gamma * adpt_out2
        # out = self.adapter(out)
        return x + self.drop_path(out) + self.gamma * adpt_out


class RepLKNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, g_blk=1.0, g_ffn=1.0, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False, adpt_test=0, ratio=0.25):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        adpt_test_r = adpt_test
        adpt_test_c = adpt_test
        for i in range(num_blocks):
            if adpt_test == 5:
                adpt_test_r = -1
                adpt_test_c = 1
            if adpt_test == 6:
                adpt_test_c = -1 # convffn
                adpt_test_r = 4 # replkblock
            
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, gamma=g_blk, small_kernel_merged=small_kernel_merged, adpt_test=adpt_test_r, ratio=ratio)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path, gamma=g_ffn, adpt_test=adpt_test_c, ratio=ratio)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)    #   Only use this with RepLKNet-XL on downstream tasks
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class RepLKNetAdapter(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,                 
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False,
                 small_kernel_merged=False,
                 use_sync_bn=True,
                 norm_intermediate_features=False,      # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 pretrained=None,
                 g_blk=1, g_ffn=1, trans_adpt= False, input_adpt=False, adpt_test=0,ratio=0.25,
                 num_input_images=1
                 ):
        super().__init__()

        # if num_classes is None and out_indices is None:
        #     raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        if num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()
        self.channels = channels

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.num_input_images = num_input_images
        if num_input_images == 1:
            self.stem = nn.ModuleList([
                conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        else:
            self.stem = nn.ModuleList([
                conv_bn_relu(in_channels=in_channels*num_input_images, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
                conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        self.input_adpt = input_adpt
        if input_adpt:
            self.input_adapter = InputAdapter(D_features=base_width)
        
        self.trans_adpt = trans_adpt
        if trans_adpt:
            self.trans_adpt = nn.ModuleList()
            self.trans_drop_path = nn.ModuleList()
        adpt_test_ori = adpt_test
        for stage_idx in range(self.num_stages):
            # if stage_idx <= 1:
            #     adpt_test = -1
            # else:
            #     adpt_test = adpt_test_ori
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, g_blk=g_blk, g_ffn=g_ffn, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features, adpt_test=adpt_test, ratio=ratio)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)
                if trans_adpt:
                    self.trans_adpt.append(Adapter(D_features=channels[stage_idx + 1], adpt_test=adpt_test))
                    print("trans drop rate ", dpr[sum(layers[:stage_idx])])
                    self.trans_drop_path.append(DropPath(dpr[sum(layers[:stage_idx])]))
                

        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)
        
        
        if pretrained:
            print('============= load pretrained backbone from ', pretrained)
            weights = torch.load(pretrained, map_location='cpu')
            if 'model' in weights:
                weights = weights['model']
            if 'state_dict' in weights:
                weights = weights['state_dict']
                
            if self.num_input_images == 2:
                weights['stem.0.conv.weight'] = torch.cat(
                    [weights['stem.0.conv.weight']] * num_input_images, 1) / num_input_images
            self.load_state_dict(weights, strict=False)
            
        
        # initialize weights for adapters:
        for n, m in self.named_modules():
            if 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
                            # print(n)
                            
                        elif isinstance(m2, nn.Conv2d):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
                            # print(n)
                            
                        # if isinstance(m, nn.Linear):
                        #     trunc_normal_(m.weight, std=.02)
                        #     if isinstance(m, nn.Linear) and m.bias is not None:
                        #         nn.init.constant_(m.bias, 0)
                    # else:
                    #     if isinstance(m2, nn.BatchNorm2d):
                    #         # print(m2.named_parameters()[0])
                    #         nn.init.constant_(m2.bias, 0)
                    #         nn.init.constant_(m2.weight, 1.0)
                    #     elif isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear):
                    #         trunc_normal_(m2.weight, std=.02)
                    #         if m2.bias is not None:
                    #             nn.init.constant_(m2.bias, 0)
        

    def forward_features(self, x):
        x = self.stem[0](x)
        if self.input_adpt:
            adpt_out = self.input_adapter(x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)
        if self.input_adpt:
            x = x + adpt_out

        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
                    if self.trans_adpt:
                        x = x + self.trans_drop_path[stage_idx](self.trans_adpt[stage_idx](x))
                    
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        # if self.out_indices:
        return x #, tokens
        # else:
        #     x = self.norm(x)
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     x = self.head(x)
        #     return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()

    def init_token(self, token_num, embed_dim=256, pos_dim=256, 
                   hidden_dim=256, num_attn_layers=8, nheads=8):
        self.token_num = token_num
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads

        self.seg_token = nn.Parameter(torch.zeros(1, token_num, self.hidden_dim))
        self.seg_token = trunc_normal_(self.seg_token, std=.02)
        self.token_pos_embed = nn.Parameter(torch.zeros(1, token_num, self.pos_dim))
        self.token_pos_embed = trunc_normal_(self.token_pos_embed, std=.02)

        self.num_attn_layers = num_attn_layers
        # self.cross_attn_layers = nn.ModuleList()
        # self.self_attn_layers = nn.ModuleList()
        # self.ffn_layers = nn.ModuleList()

        self.dim_adjust_layers = nn.ModuleList()
        self.feature_dims = [embed_dim * 2**i for i in range(self.num_stages)]
        
        self.attn_level = 3
        self.cross_attn_layer = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=False,
        )
    
        self.self_attn_layer = SelfAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=False,
        )
        self.ffn_layer = FFNLayer(
            d_model=hidden_dim,
            dim_feedforward=256,
            dropout=0.0,
            normalize_before=False,
        )
        
        self.dim_adjust_layer = nn.Sequential(
            nn.Conv2d(self.feature_dims[self.attn_level], self.hidden_dim, kernel_size=1),
            nn.GroupNorm(32, self.hidden_dim)
        )


def create_RepLKNet31B_Adapter(drop_path_rate=0.3, num_classes=1000, num_input_images=1, out_indices=(0, 1, 2, 3), use_checkpoint=True, small_kernel_merged=False, pretrained=None, use_sync_bn=True, g_blk=1.0, g_ffn=1.0, ratio=0.25, trans_adpt=False, input_adpt=False, adpt_test=0):
    return RepLKNetAdapter(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, out_indices=out_indices, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged, use_sync_bn=use_sync_bn, pretrained=pretrained, g_blk=g_blk, g_ffn=g_ffn, trans_adpt=trans_adpt, input_adpt=input_adpt, adpt_test=adpt_test, ratio=ratio, num_input_images=num_input_images)

def create_RepLKNet31L_Adapter(drop_path_rate=0.3, num_classes=1000, num_input_images=1, out_indices=(0, 1, 2, 3), use_checkpoint=True, small_kernel_merged=False, pretrained=None, use_sync_bn=True, g_blk=1.0, g_ffn=1.0, ratio=0.25, trans_adpt=False, input_adpt=False, adpt_test=0):
    return RepLKNetAdapter(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, out_indices=out_indices, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged, use_sync_bn=use_sync_bn, pretrained=pretrained, g_blk=g_blk, g_ffn=g_ffn, trans_adpt=trans_adpt, input_adpt=input_adpt, adpt_test=adpt_test, ratio=ratio, num_input_images=num_input_images)

def create_RepLKNetXL_Adapter(drop_path_rate=0.3, num_classes=1000, num_input_images=1, out_indices=(0, 1, 2, 3), use_checkpoint=True, small_kernel_merged=False, pretrained=None, use_sync_bn=True, g_blk=1.0, g_ffn=1.0, ratio=0.25, trans_adpt=False, input_adpt=False, adpt_test=0):
    return RepLKNetAdapter(large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], channels=[256,512,1024,2048],
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
                    num_classes=num_classes, out_indices=out_indices, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged, use_sync_bn=use_sync_bn, pretrained=pretrained, g_blk=g_blk, g_ffn=g_ffn, trans_adpt=trans_adpt, input_adpt=input_adpt, adpt_test=adpt_test, ratio=ratio, num_input_images=num_input_images)

if __name__ == '__main__':
    model = create_RepLKNet31B_Adapter(
        drop_path_rate=0.5,
        num_classes=None,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        small_kernel_merged=False,
        pretrained=None,#'../../../DSformer/RepLKNet-31L_ImageNet-22K.pth',
        use_sync_bn=False,
        trans_adpt=True,
        adpt_test=7)
    for name, param in model.named_parameters():
        if 'adapter' in name:
            print(name, param.shape)
    exit(0)
    model.eval()
    print('------------------- training-time model -------------')
    # print(model)
    x = torch.randn(2, 3, 224, 224)
    # x = x.cuda()
    # model.cuda()
    origin_y = model(x)
    for item in origin_y:
        print(item.shape)
    # model.structural_reparam()
    # print('------------------- after re-param -------------')
    # # print(model)
    # reparam_y = model(x)
    # print('------------------- the difference is ------------------------')
    # print((origin_y - reparam_y).abs().sum())


