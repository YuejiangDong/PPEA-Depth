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
from ppeadepth import datasets, networks
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

def validate(model, opt, min_depth_bin, max_depth_bin, val_frames_to_load=[0, -1]):
    """
        Validate the model on a single minibatch
    """
    if opt.eval_split != 'cityscapes':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    else:
        filenames = readlines(os.path.join(splits_dir, "cityscapes_preprocessed", "test_files.txt"))

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
    
    model.eval()
    pred_disps = []
    
    mono_flag = opt.eval_teacher
    if mono_flag:
        pred_disps_mono = []

    iterator = tqdm(val_loader)
    import time
    t1 = time.time()
    with torch.no_grad():
        for data in iterator:

            input_color = data[("color", 0, 0)].cuda()

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
            
            if mono_flag:
                output_mono = model.mono_depth(model.mono_encoder(input_color))
                pred_disp_mono, _ = disp_to_depth(output_mono[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp_mono = pred_disp_mono.cpu()[:, 0].numpy()
                pred_disps_mono.append(pred_disp_mono)
            
            # TODO TEST CODE
            # break
                
        pred_disps = np.concatenate(pred_disps)

    
    t2 = time.time()
    print("average inference time ", (t2-t1) / pred_disps.shape[0])
    
    if opt.eval_split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join("./splits", opt.eval_split, "gt_depths")
    else:
        gt_path = os.path.join('./splits', opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors = []
    ratios = []
    if mono_flag:
        ratios_mono = []
        errors_mono = []
        pred_disps_mono = np.concatenate(pred_disps_mono)
    
    iter_id_st = 0
    iter_id_end = pred_disps.shape[0]
    
    ##### specific test for one video seq
    # for 0926_0048
    # if opt.test_scale:
    #     iter_id_st = 0
    #     iter_id_end = 1
    #     # id_t_order = [225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246]
    #     # iter_id_st = 225
    #     # iter_id_end = 247        
    # ###################
    
    for i in range(pred_disps.shape[0]):

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
        
        with open('after.txt', 'a') as f:
            for i, item in enumerate(pred_depth):
                f.write(f'{abs(gt_depth[i]-item)}\n')

        errors.append(compute_errors(gt_depth, pred_depth))
        
        if mono_flag:
            pred_disp_mono = pred_disps_mono[i]
            pred_disp_mono = cv2.resize(pred_disp_mono, (gt_width, gt_height))
            pred_depth_mono = 1 / pred_disp_mono
            if opt.eval_split == 'cityscapes':
                pred_depth_mono = pred_depth_mono[256:, 192:1856]
            
            pred_depth_mono = pred_depth_mono[mask]
            ratio_mono = np.median(gt_depth) / np.median(pred_depth_mono)
            ratios_mono.append(ratio_mono)
            pred_depth_mono *= ratio_mono
            pred_depth_mono[pred_depth_mono < opt.min_depth] = opt.min_depth
            pred_depth_mono[pred_depth_mono > opt.max_depth] = opt.max_depth

            errors_mono.append(compute_errors(gt_depth, pred_depth_mono))

    mean_errors = np.array(errors).mean(0)
    
    if mono_flag:
        mean_errors_mono = np.array(errors_mono).mean(0)
        # if opt.test_scale:
        #     print(ratios, ratios_mono)
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

    if opt.eval_cs:
        opt.data_path = '/mnt/disk-2/alice/cityscapes' #'../nas/dataset/cityscapes'  
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
