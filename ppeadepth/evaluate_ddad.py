# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from .utils import readlines
from .options import MonodepthOptions
from manydepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
from tqdm import tqdm

from manydepth.vis import colorize
from PIL import Image
import matplotlib.pyplot as plt

def vis_depth(arr, name):
    d0 = colorize(arr, vmin=0, vmax=None)
    im_d0 = Image.fromarray(d0)
    im_d0.save(os.path.join('./debugout', name+'.jpg'))
    
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def validate(model, opt, min_depth_bin, max_depth_bin, val_frames_to_load=[0, -1]):
    """
        Validate the model on a single minibatch
    """
    val_dataset = datasets.DDADDataset(4, is_train=False)
    val_loader = DataLoader(
        val_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    """Validate the model on a single minibatch
    """
    errors = []
    ratios = []
            
    MIN_VAL = 1e-3
    MAX_VAL = 80
    
    mono_flag = opt.eval_teacher
    if mono_flag:
        pred_disps_mono = []
        errors_mono = []
        ratios_mono = []

    with torch.no_grad():
        iterator = tqdm(val_loader)#, desc=f"Epoch: {self.epoch + 1}/{self.opt.
        i = 0
        for data in iterator:# self.val_loader: #tqdm(self.val_loader, desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Validation"):
            # if i == 600 and self.opt.ddad:
            #     break
            input_color = data[("color", 0, 0)].cuda()
            # if torch.cuda.is_available():
            #     input_color = input_color.cuda()
            if opt.static_camera:
                for f_i in val_frames_to_load:
                    data["color", f_i, 0] = data[('color', 0, 0)]

            # predict poses
            pose_feats = {f_i: data["color", f_i, 0] for f_i in val_frames_to_load}
            if torch.cuda.is_available():
                pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in val_frames_to_load[1:]:
                if fi < 0:
                    pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    pose_inputs = [model.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = model.pose(pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)

                    # now find 0->fi pose
                    if fi != -1:
                        pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [model.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = model.pose(pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)

                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                data[('relative_pose', fi)] = pose
                
            lookup_frames = [data[('color', -1, 0)]] #  for idx in self.val_frames_to_load[1:]]
            lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

            relative_poses = [data[('relative_pose', -1)]] # for idx in self.val_frames_to_load[1:]]
            relative_poses = torch.stack(relative_poses, 1)

            K = data[('K', 2)]  # quarter resolution for matching
            invK = data[('inv_K', 2)]

            if torch.cuda.is_available():
                lookup_frames = lookup_frames.cuda()
                relative_poses = relative_poses.cuda()
                K = K.cuda()
                invK = invK.cuda()

            if opt.zero_cost_volume:
                relative_poses *= 0
            
            
            output, _, _ = model.encoder(input_color, lookup_frames, relative_poses, K, invK, min_depth_bin, max_depth_bin)
            output = model.depth(output)
            
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 200)
                
                # pred_disp = pred_disp # (b, 1, h, w)
                # vis_depth(pred_disp[0][0].cpu().numpy(), "pred")
                

            gt_depth = data['depth'].cpu().numpy()
            gt_height, gt_width = gt_depth.shape[-2:] 
            pred_disp = torch.nn.functional.interpolate(pred_disp, size=(gt_height, gt_width), mode='bilinear', align_corners=True).cpu()[:, 0].numpy()
                
            if mono_flag:
                output_mono = model.mono_depth(model.mono_encoder(input_color))
                pred_disp_mono, _ = disp_to_depth(output_mono[("disp", 0)], 0.1, 100)
                pred_disp_mono = torch.nn.functional.interpolate(pred_disp_mono, size=(gt_height, gt_width), mode='bilinear').cpu()[:, 0].numpy()
                    
            for b_idx in range(pred_disp.shape[0]):
                pred_ = 1./pred_disp[b_idx] #1/pred_disp[b_idx]
                gt_ = gt_depth[b_idx]
                # vis_depth(gt_, "gt")
                # img = Image.fromarray(np.uint8(255 * data[("color", 0, 0)][0].permute(1,2,0))) # no opencv required
                # img.save("./debugout/file.png")
                
                mask = np.logical_and(gt_ > 0.0, gt_ < 200)
                
                pred_ = pred_[mask]
                gt_ = gt_[mask]
                
                pred_ *= opt.pred_depth_scale_factor
                if not opt.disable_median_scaling:
                    ratio = np.median(gt_) / np.median(pred_)
                    ratios.append(ratio)
                    pred_ *= ratio
                
                pred_[pred_ < 0.0] = 0.0
                pred_[pred_ > 200] = 200

                errors.append(compute_errors(gt_, pred_))
                # print(errors[0])
                # exit(0)
                                
                if mono_flag:
                    pred_mono_ = 1./pred_disp_mono[b_idx]
                    
                    pred_mono_ = pred_mono_[mask]
                
                    ratio_mono = np.median(gt_) / np.median(pred_mono_)
                    ratios_mono.append(ratio_mono)
                    pred_mono_ *= ratio_mono
                    pred_mono_[pred_mono_ < MIN_VAL] = MIN_VAL
                    pred_mono_[pred_mono_ > 200] = 200

                    errors_mono.append(compute_errors(gt_, pred_mono_))
                    # print(errors_mono[0])
                    # exit(0)

            i += 1
                
            
        mean_errors = np.array(errors).mean(0)
        if mono_flag:
            mean_errors_mono = np.array(errors_mono).mean(0)
            if opt.debug:
                print(ratio, ratio_mono)
                    
    if mono_flag:
        return mean_errors, mean_errors_mono 
    else:
        return mean_errors
   

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    weight1 = opt.w1
    weight2 = opt.w2
    MIN_DEPTH = 0.001 #opt.min_depth
    MAX_DEPTH = 80 # opt.max_depth
    opt.min_depth = MIN_DEPTH
    opt.max_depth = MAX_DEPTH
    print(f"MIN_DEPTH = {MIN_DEPTH}")
    

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    opt.dataset = 'ddad' 
    opt.height = 320
    opt.width = 480 
    opt.split = 'ddad' 
    opt.eval_split = 'ddad'
        
    if opt.load_weights_folder is not None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Setup dataloaders
    if not opt.separate_load:
        model_path = os.path.join(opt.load_weights_folder, "model.pth")
        model_weights = torch.load(model_path, map_location='cpu') # .state_dict()
        model = networks.RepDepth(opt)
        model.load_state_dict(model_weights, strict=False)
        model.cuda()
        
        track_path = os.path.join(opt.load_weights_folder, "track.pth")
        track_dict = torch.load(track_path, map_location='cpu')
        min_depth_bin = track_dict.get('min_depth_bin')
        max_depth_bin = track_dict.get('max_depth_bin')
    else:
        model = networks.RepDepth(opt)
        
        pretrained_folder = opt.load_weights_folder
        whole_encoder = torch.load(pretrained_folder+'/encoder.pth', map_location='cpu')
        model.encoder.load_state_dict(whole_encoder.state_dict(), strict=False)
        
        whole_mono_encoder = torch.load(pretrained_folder+'/mono_encoder.pth', map_location='cpu')
        model.mono_encoder.load_state_dict(whole_mono_encoder.state_dict(), strict=False)
        
        # assign drop path of whole model to self.encoder
        for i, layer in enumerate(model.encoder.replk.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_encoder.replk.stages[i].blocks[j].drop_path
        
        # assign drop path of whole model to self.monoencoder
        for i, layer in enumerate(model.mono_encoder.stages):
            for j, blk in enumerate(layer.blocks):
                blk.drop_path = whole_mono_encoder.stages[i].blocks[j].drop_path
        
        # model.mono_encoder.trans_drop_path = whole_mono_encoder.trans_drop_path
        depth_dict = torch.load(pretrained_folder+'/depth.pth', map_location='cpu')
        model.depth.load_state_dict(depth_dict, strict=False)
        model.mono_depth.load_state_dict(torch.load(pretrained_folder+'/mono_depth.pth', map_location='cpu'), strict=False)
        model.pose_encoder.load_state_dict(torch.load(pretrained_folder+'/pose_encoder.pth', map_location='cpu'))
        model.pose.load_state_dict(torch.load(pretrained_folder+'/pose.pth', map_location='cpu'))
        
        min_depth_bin = torch.Tensor([depth_dict.get('min_depth_bin')])
        max_depth_bin = torch.Tensor([depth_dict.get('max_depth_bin')])   
        model.cuda() 
        print(min_depth_bin, max_depth_bin)
    model.eval()
    
    

    eval_res = validate(model, opt, min_depth_bin, max_depth_bin)
    print(eval_res)
    
    if opt.eval_teacher:
        errors, mono_errors = eval_res 
        print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*errors.tolist()) + "\\\\")
        print("------------------------------------------------------\n")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mono_errors.tolist()) + "\\\\")

    
    else:
        errors = eval_res
        print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*errors.tolist()) + "\\\\")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
