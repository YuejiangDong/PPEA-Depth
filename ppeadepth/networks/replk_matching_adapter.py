import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np

from manydepth.layers import BackprojectDepth, Project3D
from manydepth.networks import create_RepLKNet31B_Adapter, create_RepLKNet31L_Adapter, create_RepLKNetXL_Adapter
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

    def forward(self, x):
        # x.shape = B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        return x

class RepLKMatchingAdapter(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, rep_size, use_checkpoint, trans_adpt, input_adpt, adpt_test, g_blk, g_ffn,ratio,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(RepLKMatchingAdapter, self).__init__()
        
        if rep_size == 'b': 
            pretrained_path = "RepLKNet-31B_ImageNet-1K_224.pth"
            class_name = create_RepLKNet31B_Adapter
            self.num_ch_enc = np.array([128, 256, 512, 1024])
            
        elif rep_size == 'l':
            pretrained_path = "RepLKNet-31L_ImageNet-22K.pth"
            class_name = create_RepLKNet31L_Adapter
            self.num_ch_enc = np.array([192, 384, 768, 1536]) # Large Model
            
        else: # xl:
            pretrained_path = "RepLKNet-XL_MegData73M_pretrain.pth"
            class_name = create_RepLKNetXL_Adapter
            self.num_ch_enc = np.array([256, 512, 1024, 2048]) # XL Model
        
        # if load_full_model:
            # pretrained_path = None
        
        self.replk = class_name(
            drop_path_rate=0.3,
            num_classes=None,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=use_checkpoint,
            small_kernel_merged=False,
            pretrained=pretrained_path,
            use_sync_bn=True,
            g_blk=g_blk,
            g_ffn=g_ffn,
            ratio=ratio,
            trans_adpt=trans_adpt,
            input_adpt=input_adpt,
            adpt_test=adpt_test
        )   
        
        self.trans_adpt = trans_adpt
        self.input_adpt = input_adpt
            
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        

        # self.num_ch_enc = np.array([128, 256, 512, 1024]) # Base Model
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        # self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        # resnets = {18: models.resnet18,
        #            34: models.resnet34,
        #            50: models.resnet50,
        #            101: models.resnet101,
        #            152: models.resnet152}

        # if num_layers not in resnets:
        #     raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # encoder = resnets[num_layers](pretrained)
        # self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        # self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        # self.layer2 = encoder.layer2
        # self.layer3 = encoder.layer3
        # self.layer4 = encoder.layer4

        # if num_layers > 34:
        #     self.num_ch_enc[1:] *= 4

        # self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
        #                                       height=self.matching_height,
        #                                       width=self.matching_width)
        # self.projector = Project3D(batch_size=self.num_depth_bins,
        #                            height=self.matching_height,
        #                            width=self.matching_width)

        # self.compute_depth_bins(min_depth_bin, max_depth_bin)

        # self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
        #                                                 kernel_size=1, stride=1, padding=0),
        #                                       nn.ReLU(inplace=True)
        #                                       )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[0] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[0],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )
        # self.adpt = Adapter(D_features=self.replk.channels[1])

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        elif self.depth_binning == 'log':
            base = torch.log(min_depth_bin)
            it = torch.log(max_depth_bin / min_depth_bin)
            
            exp_base = [base + it * i / self.num_depth_bins for i in range(self.num_depth_bins)]
            self.depth_bins = torch.exp(torch.Tensor(exp_base))
        else:
            raise NotImplementedError
        # self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        self.warp_depths = self.warp_depths.to(self.device)
    
    def match_features_dyn(self, current_feats, lookup_feats, relative_poses, K, invK, lookup_images, cv_min, aug_mask, set_1, pool, pool_r, pool_th):
        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        occ_batch = F.interpolate((lookup_images.sum(1).unsqueeze(1)<0.15).float(), [48, 128]) # occluded areas after DOMD module will be black, so we assume their RGB value < 0.15
        
        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            if cv_min:
                cost_volume = torch.ones(volume_shape, dtype=torch.float, device=current_feats.device)
            else:
                cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
                counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear', align_corners=True)
                num_channels = warped.shape[1]
                if aug_mask[batch_idx][0][0][0] == 0:
                    if set_1 or pool:
                        occ_mask = (occ_batch[batch_idx]>0).unsqueeze(0).repeat([96,num_channels,1,1])
                        mask = (F.grid_sample(occ_mask.float(), pix_locs, padding_mode='zeros', mode='bilinear', align_corners=True) > pool_th).detach() # project the occlusion mask of the image to each layers of the cost volume
                        # mask = mask.permute(0, 2, 3, 1)
                        # print("debug  ", warped.shape, mask.shape)
                        if set_1: # Set all occluded cost to be 1.0
                            warped[mask] = 1.0
                        elif pool: # Use nearby non-occluded area cost value to replace occluded ones, as mentioned in the paper.
                            x = warped.clone()
                            x[mask] = 0
                            x = F.max_pool3d(x.permute(1, 0, 2, 3), pool_r*2+1, stride=1, padding=pool_r).permute(1, 0, 2, 3)
                            warped[mask] = x[mask]

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask

                # integrate into cost volume
                if cv_min:
                    diffs[diffs==0] = 1.0
                    cost_volume = torch.minimum(diffs, cost_volume)
                else:
                    cost_volume = cost_volume + diffs
                    counts = counts + (diffs > 0).float()
            # average over lookup images
            if cv_min:
                cost_volume[cost_volume==1] = 0
            else:
                cost_volume = cost_volume / (counts + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks


    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]
            
            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask

                # integrate into cost volume
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        # image = (image - 0.45) / 0.225  # imagenet normalisation
        x = self.replk.stem[0](image)
        if self.input_adpt:
            adpt_out = self.replk.input_adapter(x)
        for stem_layer in self.replk.stem[1:]:
            if self.replk.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)
        if self.input_adpt:
            x = x + adpt_out
        
        x = self.replk.stages[0](x)

        ### TODO norm or transitions?
        out = self.replk.stages[0].norm(x)
        # x = self.replk.transitions[0](x)

        # for stage_idx in range(self.num_stages):
        #     x = self.stages[stage_idx](x)
        #     if stage_idx in self.out_indices:
        #         outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
        #     if stage_idx < self.num_stages - 1:
        #         x = self.transitions[stage_idx](x)
        return x, [out]


    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None,
                teacher_depth=None,
                doj_mask=None,
                cv_min=None,
                aug_mask=None,
                set_1=None,
                pool=None,
                pool_r=None,
                pool_th=None
                ):
        
        self.dyn = True if teacher_depth != None else False

        B, _, H, W = current_image.shape
        
        self.matching_height, self.matching_width = H // 4, W // 4
        
        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        self.device = current_image.device
        self.backprojector.to(self.device)
        self.projector.to(self.device)
        
        self.compute_depth_bins(min_depth_bin, max_depth_bin)
        
        
        # debug 
        # print("forward input shape: ", poses.shape, K.shape, invK.shape)

        # feature extraction
        current_feats, self.features = self.feature_extraction(current_image, return_all_feats=True)
    
        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)

            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
            lookup_feats, _ = self.feature_extraction(lookup_images,
                                                   return_all_feats=False)
            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

            # warp features to find cost volume
            if self.dyn:
                cost_volume, missing_mask = \
                self.match_features_dyn(self.features[-1], lookup_feats, poses, K, invK, lookup_images, cv_min=cv_min, aug_mask=aug_mask,
                set_1=set_1, pool=pool, pool_r=pool_r, pool_th=pool_th)
            else:
                cost_volume, missing_mask = \
                    self.match_features(self.features[-1], lookup_feats, poses, K, invK)
            confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                           (1 - missing_mask.detach()))

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)
        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], cost_volume], 1))
        # post_matching_feats.shape = B, 128, H/4, W/4
        # test_id = self.replk.stages[0].blocks[1].adapter.test_id
        x = self.replk.transitions[0](post_matching_feats)
        if self.trans_adpt:
            x = x + self.replk.trans_drop_path[0](self.replk.trans_adpt[0](x))
            
        # x = self.adpt(x)
        for stage_idx in range(1, self.replk.num_stages):
            x = self.replk.stages[stage_idx](x)
            if stage_idx in self.replk.out_indices:
                self.features.append(self.replk.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
            if stage_idx < self.replk.num_stages - 1:
                x = self.replk.transitions[stage_idx](x)
                if self.trans_adpt:
                    x = x + self.replk.trans_drop_path[stage_idx](self.replk.trans_adpt[stage_idx](x))
                    
                # x = self.replk.trans_adpt[stage_idx](x)
        
        return self.features, lowest_cost, confidence_mask

    def cuda(self):
        super().cuda()
        # self.backprojector.cuda()
        # self.projector.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError

if __name__ == "__main__":
    model = RepLKMatchingAdapter(rep_size='b', 
                                 use_checkpoint=True, 
                                 trans_adpt=False, 
                                 adpt_test=0,
                                 g_blk=1, g_ffn=1,min_depth_bin=0.1, max_depth_bin=20)
    model.eval()
    x = torch.rand(2, 3, 192, 640)
    lookimgs = torch.rand(2, 1, 3, 192, 640)
    poses = torch.rand(2, 1, 4, 4)
    K = torch.rand(2, 4, 4)
    invk = torch.rand(2, 4, 4)
    y1, y2, y3 = model(x, lookimgs, poses, K, invk,min_depth_bin=0.1, max_depth_bin=20)

    # y1: list of features: dim=128, 256, 512, 1024; size=1/4, 1/8, 1/16, 1/32
    # y2.shape= B, H/4, W/4
    # y3.shape= B, H/4, W/4
    