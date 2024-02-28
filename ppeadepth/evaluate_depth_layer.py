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
from torch import nn
from torch.utils.data import DataLoader

from .utils import readlines
from .options import MonodepthOptions
from manydepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
from tqdm import tqdm


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

def validate(model, opt, val_loader, min_depth_bin, max_depth_bin, gt_depths, teacher=False, val_frames_to_load=[0, -1]):
    """
        Validate the model on a single minibatch
    """
    
    
    
    model.eval()
    pred_disps = []
    
    mono_flag = opt.eval_teacher
    if mono_flag:
        pred_disps_mono = []

    iterator = tqdm(val_loader)
    
    with torch.no_grad():
        for data in iterator:

            input_color = data[("color", 0, 0)].cuda()
            if teacher:
                output_mono = model.mono_depth(model.mono_encoder(input_color))
                pred_disp, _ = disp_to_depth(output_mono[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
            
            else:
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
                
                lookup_frames = [data[('color', -1, 0)]] 
                lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                relative_poses = [data[('relative_pose', -1)]] 
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
                
                
                output, lowest_cost, costvol = model.encoder(input_color, lookup_frames,
                                                        relative_poses,
                                                        K,
                                                        invK,
                                                        min_depth_bin, max_depth_bin)
                output = model.depth(output)
                

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps.append(pred_disp)
                
            
                
        pred_disps = np.concatenate(pred_disps)

        
    errors = []
    ratios = []
    # if mono_flag:
    #     ratios_mono = []
    #     errors_mono = []
    #     pred_disps_mono = np.concatenate(pred_disps_mono)
    
    iter_id_st = 0
    iter_id_end = pred_disps.shape[0]
    
    ##### specific test for one video seq
    # for 0926_0048
    ###################

    for i in range(iter_id_st, iter_id_end):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]


        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > opt.min_depth, gt_depth < opt.max_depth)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        elif opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > opt.min_depth, gt_depth < opt.max_depth)
        else:
            mask = gt_depth > 0
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            
            # np.save("pred_wo_before.npy", pred_depth)
            
            ##### specific test for one video seq
            # print(i, ratio, np.median(gt_depth), np.median(pred_depth))
            ##########
            ratios.append(ratio)
            pred_depth *= ratio
            
            # np.save("gt.npy", gt_depth)
            
        
        pred_depth[pred_depth < opt.min_depth] = opt.min_depth
        pred_depth[pred_depth > opt.max_depth] = opt.max_depth

        errors.append(compute_errors(gt_depth, pred_depth))
        
        # if mono_flag:
        #     pred_disp_mono = pred_disps_mono[i]
        #     pred_disp_mono = cv2.resize(pred_disp_mono, (gt_width, gt_height))
        #     pred_depth_mono = 1 / pred_disp_mono
        #     if opt.eval_split == 'cityscapes':
        #         pred_depth_mono = pred_depth_mono[256:, 192:1856]
            
        #     pred_depth_mono = pred_depth_mono[mask]
        #     ratio_mono = np.median(gt_depth) / np.median(pred_depth_mono)
        #     ratios_mono.append(ratio_mono)
        #     pred_depth_mono *= ratio_mono
        #     pred_depth_mono[pred_depth_mono < opt.min_depth] = opt.min_depth
        #     pred_depth_mono[pred_depth_mono > opt.max_depth] = opt.max_depth

        #     errors_mono.append(compute_errors(gt_depth, pred_depth_mono))

    mean_errors = np.array(errors).mean(0)
    
    if opt.test_scale:
        # ratios_in_t_order = [ratios[i-iter_id_st] for i in id_t_order]
        # print(ratios_in_t_order)
        ######## general scale test across all video sequences
        # import pickle
        # # read video sequence 
        # with open('./splits/eigen/video_seq.pkl', 'rb') as f:
        #     video_id = pickle.load(f)
        # for k, v in video_id.items():
        #     ratio_s = [ratios[i] for i in v]
        #     # generate statistics for ratios and ratios_mono
        #     print(f'for main net: min = {np.min(ratio_s)}, max = {np.max(ratio_s)}, mean = {np.mean(ratio_s)}, std = {np.std(ratio_s)}')
        # if opt.eval_teacher:
        #     for k, v in video_id.items():
        #         ratio_mono_s = [ratios_mono[i] for i in v]
        #         print(f'for mono net: min = {np.min(ratio_mono_s)}, max = {np.max(ratio_mono_s)}, mean = {np.mean(ratio_mono_s)}, std = {np.std(ratio_mono_s)}')
        #############################
        print(f'for main net: min = {np.min(ratios)}, max = {np.max(ratios)}, mean = {np.mean(ratios)}, std = {np.std(ratios)}')
     
    # if mono_flag:
    #     mean_errors_mono = np.array(errors_mono).mean(0)
    #     # if opt.test_scale:
    #     #     print(ratios, ratios_mono)
    # if mono_flag:
    #     return mean_errors, mean_errors_mono 
    # else:
    return mean_errors


def evaluate(opt, disable_id, gt_depths, teacher=False):
    """Evaluates a pretrained model using a specified test set
    """
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

    if opt.eval_cs:
        opt.data_path = '/home/share/dyj' #'../nas/dataset/cityscapes'  
        opt.eval_split = 'cityscapes'
        opt.height=192
        opt.width=512
    
    if opt.load_weights_folder is not None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Setup dataloaders
    if not opt.separate_load:
        model_path = os.path.join(opt.load_weights_folder, "model.pth")
        
        # new ver : state_dict
        model = networks.RepDepth(opt)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        
        # old ver 
        # model = torch.load(model_path, map_location='cpu')

        model.cuda()
        track_path = os.path.join(opt.load_weights_folder, "track.pth")
        track_dict = torch.load(track_path, map_location='cpu')
        min_depth_bin = track_dict.get('min_depth_bin')
        max_depth_bin = track_dict.get('max_depth_bin')
    else:
        model = networks.RepDepth(opt)
        min_depth_bin, max_depth_bin = model.load_drop_path(None)   
        model.cuda() 
        print(min_depth_bin, max_depth_bin)
    val_frames_to_load=[0, -1]
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
    if opt.eval_split == 'cityscapes':
        dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                    opt.height, opt.width,
                                                    val_frames_to_load, 4,
                                                    is_train=False)

    else:
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width, #, WIDTH,#encoder_dict['height'], encoder_dict['width'],
                                            val_frames_to_load, 4,
                                            is_train=False)
    
    val_loader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    
    print("len val", len(val_loader))
    
    blk_cnter = 0
    # try drop path rate zero
    for i, layer in enumerate(model.encoder.replk.stages):
        for j, blk in enumerate(layer.blocks):
            if blk_cnter in disable_id:
                blk.test_id = -1
            blk_cnter += 1
        # assign drop path of whole model to self.monoencoder
    blk_cnter = 0
    for i, layer in enumerate(model.mono_encoder.stages):
        for j, blk in enumerate(layer.blocks):
            if blk_cnter in disable_id:
                blk.test_id = -1
            blk_cnter += 1
    #         blk.drop_path = nn.Identity()
    # model.mono_encoder.trans_drop_path = nn.Identity()

    eval_res = validate(model, opt, val_loader, min_depth_bin, max_depth_bin, gt_depths, teacher=teacher)
    
    # if opt.eval_teacher:
    #     errors, mono_errors = eval_res 
    #     print("\n  " + ("{:>8} | " * 7).format("abs_rel",
    #                                         "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    #     print(("&{: 8.3f}  " * 7).format(*errors.tolist()) + "\\\\")
    #     print("------------------------------------------------------\n")
    #     print("\n  " + ("{:>8} | " * 7).format("abs_rel",
    #                                         "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    #     print(("&{: 8.3f}  " * 7).format(*mono_errors.tolist()) + "\\\\")

    
    # else:
    errors = eval_res
    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                        "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*errors.tolist()) + "\\\\")
    return errors


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    
    all_blks = 48
    
    # import numpy as np
    # errs = evaluate(opt, [9, 12, 20, 22, 30])# [9, 12,20,22,26,30,34,41,45,47])
    gt_path = "./manydepth/gt_depths_val.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    
    # errs = evaluate(opt, [], gt_depths)
    # print(errs)
    # exit(0)
    # if opt.debug:
    if opt.eval_teacher:
        for i in range(all_blks):
            errs = evaluate(opt, [i], gt_depths, teacher=True)
            with open('repl_teacher.txt', 'a') as f:
                f.write(f'{i} {errs[0]} {errs[4]}')
                f.write('\n')
    else:
        for i in range(all_blks):#, -1, -1):
            errs = evaluate(opt, [i], gt_depths)
            with open('repl.txt', 'a') as f:
                f.write(f'{i} {errs[0]} {errs[4]}')
                f.write('\n')
# else:
    #     for i in range(all_blks):
    #         errs = evaluate(opt, [i], gt_depths)
    #         with open('1clcb_valset.txt', 'a') as f:
    #             f.write(f'{i} {errs[0]} {errs[4]}')
    #             f.write('\n')
            
    # print(evaluate(opt, [0, 1, 2, 3, 4, 5, 6]))
    # for i in range(0, all_blks, 2):
    #     errs = evaluate(opt, [i, i+1])
    #     with open('2blk.txt', 'a') as f:
    #         f.write(f'{i}, {i+1} {errs[0]} {errs[4]}')
    #         f.write('\n')
