# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random
import uuid
from datetime import datetime as dt
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

import json

from .utils import readlines, sec_to_hm_str
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors, Sobel
from .evaluate_depth import compute_errors
from ppeadepth import datasets, networks
import matplotlib.pyplot as plt

from accelerate import Accelerator
from torchmetrics import Metric
from collections import OrderedDict

from PIL import Image

class DepthBins(Metric):
    full_state_update: bool = True
    def __init__(self, opt_min_depth):
        super().__init__()
        self.add_state("min_depth", default=torch.tensor(0.1), dist_reduce_fx="min")
        self.add_state("max_depth", default=torch.tensor(10.0), dist_reduce_fx="max")
        self.opt_min_depth = opt_min_depth
        self.updated = False

    def update(self, mono_depth: torch.Tensor):
        self.updated = True
        min_depth = mono_depth.detach().min(-1)[0].min(-1)[0]
        max_depth = mono_depth.detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean()
        max_depth = max_depth.mean()
        
        min_depth = max(self.opt_min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1
        
        self.max_depth = self.max_depth * 0.99 + max_depth * 0.01
        self.min_depth = self.min_depth * 0.99 + min_depth * 0.01
    
    def load(self, min_depth: torch.Tensor, max_depth: torch.Tensor):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def compute(self):
        return self.min_depth.float(), self.max_depth.float()


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

PROJECT = "SMDE_many"
logging = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, options, acc):
        self.opt = options
        self.log_path = './ckpt'
        
        self.acc = acc # accelerator
        
        if self.opt.train_cs:
            self.opt.dataset = 'cityscapes_preprocessed' 
            self.opt.height = 192
            self.opt.width = 512 
            # self.opt.data_path = '../cs'
            self.opt.split = 'cityscapes_preprocessed' 
            self.opt.eval_split = 'cityscapes'
        
        if self.opt.ddad:
            self.opt.dataset = 'ddad' 
            self.opt.height = 384
            self.opt.width = 640 
            self.opt.split = 'ddad' 
            self.opt.eval_split = 'ddad'
        

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"


        self.device = acc.device

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        
        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            if idx not in frames_to_load:
                frames_to_load.append(idx)
        
        # frames_to_load for validation
        self.val_frames_to_load = [0]
        if self.opt.use_future_frame:
            self.val_frames_to_load.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            if idx not in self.val_frames_to_load:
                self.val_frames_to_load.append(idx)
        
        # print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.model = networks.RepDepth(self.opt)
        
        filtered_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model_optimizer = optim.Adam(filtered_params, self.opt.learning_rate)
        
        model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
            
        self.depth_bin_tracker = DepthBins(self.opt.min_depth)
        self.depth_bin_tracker.to(self.device)
        
        # TEST SCRIPT
        if self.opt.load_weights_folder is not None:
            self.load_model(transfer=self.opt.ktf)
            
            if self.opt.dc:
                self.model.dc_ft_init()
            print("finish load model")
        
        else:
            if self.opt.ktf and self.opt.rep_size == 'b':
                self.model.load_drop_path(depthbin_tracker=None)
            if self.opt.dc:
                self.model.dc_ft_init()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()
            
        self.freeze_tp = self.opt.freeze_teacher_and_pose
        if self.freeze_tp:
            self.model.freeze_tp_net()
            self.depth_bin_tracker.updated = False
            
        self.freeze_pose = self.opt.freeze_pose
        if self.freeze_pose:
            self.model.freeze_pose_net()
            self.opt.learning_rate = 1e-6
            model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        
        print("Training model named:\n  ", self.opt.model_name)
        print("Training is using:\n  ", self.device)

        # DATA
        if self.opt.ddad:
            train_dataset = datasets.DDADDataset(4, is_train=True)
            self.opt.eval_split = 'ddad'
            val_dataset = datasets.DDADDataset(4, is_train=False)
        else:
            datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                            "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                            "kitti_odom": datasets.KITTIOdomDataset,
                            "cityscapes_eval": datasets.CityscapesEvalDataset}
            self.dataset = datasets_dict[self.opt.dataset]

            fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("test"))
            img_ext = '.png' if self.opt.png else '.jpg'
            
            num_train_samples = len(train_filenames)
            self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                frames_to_load, 4, is_train=True, img_ext=img_ext)
            
            val_path = self.opt.data_path
            
            if self.opt.dataset != 'kitti':
                self.dataset = datasets_dict["cityscapes_eval"]
                val_path = self.opt.cs_eval_path
                
            val_dataset = self.dataset(
                val_path, val_filenames, self.opt.height, self.opt.width,
                self.val_frames_to_load, 4, is_train=False, img_ext=img_ext)
            
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)
        
        self.model, self.model_optimizer, self.train_loader, self.model_lr_scheduler = acc.prepare(
            self.model, self.model_optimizer, train_loader, model_lr_scheduler
        )
        
        
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in range(self.opt.sclm+1):
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        self.train_pose = True
        if self.opt.debug:
            print(self.count_parameters())
        self.is_main = acc.is_main_process
        self.depth_bin_tracker.to(self.device)
        
        if self.opt.eval:
            self.model.module.eval()
            metrics, metrics_mono = self.val(hard_test_mono = True)
            print(f"validate after load ", metrics, metrics_mono)
            exit(0)
        
        
    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if not self.train_pose:
                train_keys = ['mono_encoder', 'mono_depth', 'depth', 'encoder'] # if self.opt.train_sem else ['depth', 'encoder']
                if k in train_keys:
                    m.train()
            else:
                if self.train_teacher_and_pose:
                    if k == 'ins' or k == 'sem' or k == 'pan':# and self.opt.train_sem == False):
                        m.eval()
                    else:
                        m.train()
                else:
                    # if teacher + pose is frozen, then only use training batch norm stats for
                    # multi components
                    train_keys = ['depth', 'encoder'] # if self.opt.train_sem else ['depth', 'encoder']
                    if k in train_keys:
                        m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        global PROJECT
        experiment_name = self.opt.name
        if self.is_main:
            print(f"Training {experiment_name}")

            run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"
            name = f"{experiment_name}_{run_id}"

            tags = self.opt.tags.split(',') if self.opt.tags != '' else None
            if self.opt.dataset != 'kitti':
                PROJECT = PROJECT + f"-{self.opt.dataset}"
            wandb.init(project=PROJECT, name=name, config=self.opt, dir='.', tags=tags, notes='')
            self.model.module.print_num_param()
        
        self.best_loss = np.inf
        self.best_delta1 = 0
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        for self.epoch in range(self.opt.num_epochs):
            if self.is_main:
                wandb.log({"Epoch": self.epoch}, step=self.step)
                
            self.run_epoch()

    def count_parameters(self):
        params = 0
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(params)
        # exit(0)
    

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # print('Loading frames for val: {}'.format(self.val_frames_to_load))
        
        self.model.module.train()
        
        if self.is_main:
            iterator = tqdm(enumerate(self.train_loader), desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Train", total=len(self.train_loader)) 
        else:
            iterator = enumerate(self.train_loader)
        for batch_idx, inputs in iterator:
            outputs, losses = self.process_batch(inputs, is_train=True)
            
            # if not self.opt.rebalance:
            self.model_optimizer.zero_grad()
            self.acc.backward(losses["loss"])
            self.model_optimizer.step()

            if self.is_main and self.step % 50 == 0:
                # if self.step < 10 and batch_idx == 0:
                #     self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                wandb.log({f"Train/loss": losses["loss"]}, step=self.step)
                # wandb.log({f"Train/lr": self.model_lr_scheduler.get_last_lr()[0]}, step=self.step)
                
                for scale in range(self.opt.sclm+1):
                    wandb.log({f"Train/loss_{scale}": losses["loss/{}".format(scale)]}, step=self.step)
                    wandb.log({f"Train/loss_{scale}": losses["loss/{}".format(scale)]}, step=self.step)

            # early val check:
            mono_flag = self.train_teacher_and_pose and self.is_main # and self.opt.dataset == "cityscapes_preprocessed" 
            
            if self.step == 250:
                self.acc.wait_for_everyone()
                
                if self.is_main:
                    self.model.module.eval()
                    if self.freeze_tp:
                        metrics = self.val() if not self.opt.ddad else self.val_ddad()
                        print(f"early validate at step 30 ", metrics)
                    else:
                        metrics, metrics_mono = self.val() if not self.opt.ddad else self.val_ddad()
                        print(f"early validate at step 30 ", metrics, metrics_mono)
                    self.model.module.train()
                    print("cur lr = ", self.model_optimizer.param_groups[0]["lr"])
                    # print("cur lr = ", self.model_lr_scheduler.get_last_lr())# param_groups[0]["lr"])
                    save_folder = os.path.join("./ckpt", f"{self.opt.name}_drp")
                    self.save_drop_path(save_folder)
                    
                
            if self.step != 0 and self.step % self.opt.validate_every == 0 and self.step > self.opt.validate_from:
                # validate
                self.acc.wait_for_everyone()
                
                if self.is_main:
                    self.model.module.eval()
                    if self.freeze_tp:
                        metrics = self.val() if not self.opt.ddad else self.val_ddad()
                        print("self.step ", self.step, " validate ", metrics)
                    else:
                        metrics, metrics_mono = self.val() if not self.opt.ddad else self.val_ddad()
                        print("self.step ", self.step, " validate ", metrics)
                        print("validate teacher ", metrics_mono)
                    
                    self.model.module.train()

                    wandb.log({f"{k}": v for k, v in zip(self.depth_metric_names, metrics)}, step=self.step)
                    print(f"step {self.step} min depth {self.depth_bin_tracker.min_depth}, max depth {self.depth_bin_tracker.max_depth}")
                    
                    save_folder = os.path.join("./ckpt", f"{self.opt.name}_s{self.step}")
                    if not self.opt.saveoff: # and self.step == 2700:
                        # print(save_folder)
                        if self.step >= self.opt.save_until:
                            self.save_model_debug(save_folder)
                        # exit(0)
                
            # if self.step == self.opt.freeze_teacher_step:
            #     if self.opt.freeze_pose_only:
            #         self.freeze_pose()
            #     else:
            #         self.freeze_teacher()

            self.step += 1
            
        self.model_lr_scheduler.step()
            
    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        
        # predict poses for all frames
        if self.opt.notadabins:
            min_depth, max_depth = self.depth_bin_tracker.min_depth, self.depth_bin_tracker.max_depth 
        else:
            if self.depth_bin_tracker.updated:
                min_depth, max_depth = self.depth_bin_tracker.compute()
            else:
                min_depth, max_depth = torch.Tensor([self.depth_bin_tracker.min_depth]), torch.Tensor([self.depth_bin_tracker.max_depth]) 
            
        mono_outputs, outputs = self.model(inputs, min_depth, max_depth)
        
        # single frame path
        with self.acc.autocast():
            self.generate_images_pred(inputs, mono_outputs)
            
            mono_losses, mono_reproj = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        # aim to compute consistency loss
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]
                
        outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                        self.compute_matching_mask(outputs))
        
        with self.acc.autocast():
            self.generate_images_pred(inputs, outputs, is_multi=True)
            
            losses, main_reproj = self.compute_losses(inputs, outputs, is_multi=True)
            
        # update losses with single frame losses
        if self.freeze_tp == False:
            for key, val in mono_losses.items():
                losses[key] += val
        
        if self.freeze_tp == False and self.opt.notadabins == False:
            # update adaptive depth bins
            self.acc.wait_for_everyone()
            self.depth_bin_tracker.update(outputs[('mono_depth', 0, 0)])
            # if self.opt.debug:
            # #     self.acc.wait_for_everyone()
            # # if self.is_main:
            #     print("depth bin", self.depth_bin_tracker.compute())
        
        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01
    
    def val_ddad(self, hard_test_mono=False):
        """Validate the model on a single minibatch
        """
        errors = []
        ratios = []
                
        MIN_VAL = 1e-3
        MAX_VAL = 80
        
        mono_flag = not self.freeze_tp # and self.opt.dataset == "cityscapes_preprocessed"
        if hard_test_mono:
            mono_flag = True
        if self.opt.debug:
            print(self.opt.dataset)
            print(f"mono_flag = {mono_flag} ")
        if mono_flag:
            pred_disps_mono = []
            errors_mono = []
            ratios_mono = []

        with torch.no_grad():
            iterator = self.val_loader
            i = 0
            for data in iterator:# self.val_loader: #tqdm(self.val_loader, desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Validation"):
                # if i == 600 and self.opt.ddad:
                #     break
                input_color = data[("color", 0, 0)].to(self.device)
                # if torch.cuda.is_available():
                #     input_color = input_color.cuda()
                if self.opt.static_camera:
                    for f_i in self.val_frames_to_load:
                        data["color", f_i, 0] = data[('color', 0, 0)]

                # predict poses
                pose_feats = {f_i: data["color", f_i, 0] for f_i in self.val_frames_to_load}
                if torch.cuda.is_available():
                    pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.val_frames_to_load[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.model.module.pose_encoder(torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.model.module.pose(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.model.module.pose_encoder(torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.model.module.pose(pose_inputs)
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
                    lookup_frames = lookup_frames.to(self.device)
                    relative_poses = relative_poses.to(self.device)
                    K = K.to(self.device)
                    invK = invK.to(self.device)

                if self.opt.zero_cost_volume:
                    relative_poses *= 0
                
                if self.opt.notadabins:
                    min_depth, max_depth = self.depth_bin_tracker.min_depth, self.depth_bin_tracker.max_depth 
                else:
                    if self.depth_bin_tracker.updated:
                        min_depth, max_depth = self.depth_bin_tracker.compute()
                    else:
                        min_depth, max_depth = torch.Tensor([self.depth_bin_tracker.min_depth]), torch.Tensor([self.depth_bin_tracker.max_depth]) 
                
                output, _, _ = self.model.module.encoder(input_color, lookup_frames, relative_poses, K, invK, min_depth, max_depth)
                output = self.model.module.depth(output)
                

                pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_VAL, MAX_VAL)
                pred_disp = pred_disp # (b, 1, h, w)

                gt_depth = data['depth'].cpu().numpy()
                gt_height, gt_width = gt_depth.shape[-2:] 
                pred_depth = 1. / pred_disp
                pred_depth = torch.nn.functional.interpolate(pred_depth, size=(gt_height, gt_width), mode='bilinear').cpu()[:, 0].numpy()
                
                if mono_flag:
                    output_mono = self.model.module.mono_depth(self.model.module.mono_encoder(input_color))
                    pred_disp_mono, _ = disp_to_depth(output_mono[("disp", 0)], MIN_VAL, MAX_VAL)
                    pred_mono = 1. / pred_disp_mono
                    pred_mono = torch.nn.functional.interpolate(pred_mono, size=(gt_height, gt_width), mode='bilinear').cpu()[:, 0].numpy()
                    
                for b_idx in range(pred_disp.shape[0]):
                    pred_ = pred_depth[b_idx]
                    gt_ = gt_depth[b_idx]
                    # vis_depth(pred_, "pred")
                    # vis_depth(gt_, "gt")
                    # img = Image.fromarray(np.uint8(255 * data[("color", 0, 0)][0].permute(1,2,0))) # no opencv required
                    # img.save("./debugout/file.png")
                    
                    mask = np.logical_and(gt_ > MIN_VAL, gt_ < 200)
                    
                    pred_ = pred_[mask]
                    gt_ = gt_[mask]
                    # print(len(gt_))
                    # print(np.median(gt_), np.median(pred_))
                    # print(gt_.min(), gt_.max(), gt_.mean())
                    # print(pred_.min(), pred_.max(), pred_.mean())
                    
                    pred_ *= self.opt.pred_depth_scale_factor
                    if not self.opt.disable_median_scaling:
                        ratio = np.median(gt_) / np.median(pred_)
                        ratios.append(ratio)
                        pred_ *= ratio
                    
                    pred_[pred_ < MIN_VAL] = MIN_VAL
                    pred_[pred_ > 200] = 200

                    errors.append(compute_errors(gt_, pred_))
                                    
                    if mono_flag:
                        pred_mono_ = pred_mono[b_idx]
                        
                        pred_mono_ = pred_mono_[mask]
                    
                        ratio_mono = np.median(gt_) / np.median(pred_mono_)
                        ratios_mono.append(ratio_mono)
                        pred_mono_ *= ratio_mono
                        pred_mono_[pred_mono_ < MIN_VAL] = MIN_VAL
                        pred_mono_[pred_mono_ > 200] = 200

                        errors_mono.append(compute_errors(gt_, pred_mono_))

                i += 1
                    
            
            mean_errors = np.array(errors).mean(0)
            if mono_flag:
                mean_errors_mono = np.array(errors_mono).mean(0)
                if self.opt.debug:
                    print(ratio, ratio_mono)
                    
        if mono_flag:
            return mean_errors, mean_errors_mono 
        else:
            return mean_errors
    
    
    def val(self, hard_test_mono=False):
        """Validate the model on a single minibatch
        """
        pred_disps = []
        MIN_VAL = 1e-3
        MAX_VAL = 80 # 200 if self.opt.ddad else 80
        
        mono_flag = not self.freeze_tp # and self.opt.dataset == "cityscapes_preprocessed"
        if hard_test_mono:
            mono_flag = True
        if self.opt.debug:
            print(self.opt.dataset)
            print(f"mono_flag = {mono_flag} ")
        if mono_flag:
            pred_disps_mono = []

        with torch.no_grad():
            if self.acc.is_main_process:
                iterator = tqdm(self.val_loader)#, desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Validation")
            else:
                iterator = self.val_loader
            i = 0
            for data in iterator:# self.val_loader: #tqdm(self.val_loader, desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Validation"):
                # if i == 600 and self.opt.ddad:
                #     break
                input_color = data[("color", 0, 0)].to(self.device)
                # if torch.cuda.is_available():
                #     input_color = input_color.cuda()
                if self.opt.static_camera:
                    for f_i in self.val_frames_to_load:
                        data["color", f_i, 0] = data[('color', 0, 0)]

                # predict poses
                pose_feats = {f_i: data["color", f_i, 0] for f_i in self.val_frames_to_load}
                if torch.cuda.is_available():
                    pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.val_frames_to_load[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.model.module.pose_encoder(torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.model.module.pose(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.model.module.pose_encoder(torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.model.module.pose(pose_inputs)
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
                    lookup_frames = lookup_frames.to(self.device)
                    relative_poses = relative_poses.to(self.device)
                    K = K.to(self.device)
                    invK = invK.to(self.device)

                if self.opt.zero_cost_volume:
                    relative_poses *= 0
                
                if self.opt.notadabins:
                    min_depth, max_depth = self.depth_bin_tracker.min_depth, self.depth_bin_tracker.max_depth 
                else:
                    if self.depth_bin_tracker.updated:
                        min_depth, max_depth = self.depth_bin_tracker.compute()
                    else:
                        min_depth, max_depth = torch.Tensor([self.depth_bin_tracker.min_depth]), torch.Tensor([self.depth_bin_tracker.max_depth]) 
                
                output, lowest_cost, costvol = self.model.module.encoder(input_color, lookup_frames, relative_poses, K, invK, min_depth, max_depth)
                output = self.model.module.depth(output)
                

                pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-3, 80)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps.append(pred_disp)
                
                if mono_flag:
                    output_mono = self.model.module.mono_depth(self.model.module.mono_encoder(input_color))
                    pred_disp_mono, _ = disp_to_depth(output_mono[("disp", 0)], MIN_VAL, self.opt.max_depth)
                    pred_disp_mono = pred_disp_mono.cpu()[:, 0].numpy()
                    pred_disps_mono.append(pred_disp_mono)
                i += 1
                    
            pred_disps = np.concatenate(pred_disps)
            

            if self.opt.eval_split == 'cityscapes':
                print('loading cityscapes gt depths individually due to their combined size!')
                gt_depths = os.path.join("./splits", self.opt.eval_split, "gt_depths")
            elif self.opt.eval_split == 'ddad':
                gt_depths = '/mnt/disk-2/alice/ddad_val_npy'
            else:
                gt_path = os.path.join('./splits', self.opt.eval_split, "gt_depths.npz")
                gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

            errors = []
            ratios = []
            if mono_flag:
                ratios_mono = []
                errors_mono = []
                pred_disps_mono = np.concatenate(pred_disps_mono)
                

            for i in range(pred_disps.shape[0]):

                if self.opt.eval_split == 'cityscapes':
                    gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
                    gt_height, gt_width = gt_depth.shape[:2]
                    # crop ground truth to remove ego car -> this has happened in the dataloader for input
                    # images
                    gt_height = int(round(gt_height * 0.75))
                    gt_depth = gt_depth[:gt_height]
                elif self.opt.eval_split == 'ddad':
                    gt_depth = np.load(os.path.join(gt_depths, f'{i}.npy'), allow_pickle=True)
                    gt_height, gt_width = gt_depth.shape[:2]
                else:
                    gt_depth = gt_depths[i]
                    gt_height, gt_width = gt_depth.shape[:2]

                pred_disp = pred_disps[i]
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1 / pred_disp
                
                if self.opt.eval_split == 'cityscapes':
                    # when evaluating cityscapes, we centre crop to the middle 50% of the image.
                    # Bottom 25% has already been removed - so crop the sides and the top here
                    gt_depth = gt_depth[256:, 192:1856]
                    pred_depth = pred_depth[256:, 192:1856]


                if self.opt.eval_split == "eigen":
                    mask = np.logical_and(gt_depth > MIN_VAL, gt_depth < MAX_VAL)

                    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                    0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)
                elif self.opt.eval_split == 'cityscapes':
                    mask = np.logical_and(gt_depth > MIN_VAL, gt_depth < MAX_VAL)
                else:
                    mask = np.logical_and(gt_depth > MIN_VAL, gt_depth < MAX_VAL)
                
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
                
                pred_depth *= self.opt.pred_depth_scale_factor
                if not self.opt.disable_median_scaling:
                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio
                
                pred_depth[pred_depth < MIN_VAL] = MIN_VAL
                pred_depth[pred_depth > MAX_VAL] = MAX_VAL

                errors.append(compute_errors(gt_depth, pred_depth))
                
                if mono_flag:
                    pred_disp_mono = pred_disps_mono[i]
                    pred_disp_mono = cv2.resize(pred_disp_mono, (gt_width, gt_height))
                    pred_depth_mono = 1 / pred_disp_mono
                    if self.opt.eval_split == 'cityscapes':
                        pred_depth_mono = pred_depth_mono[256:, 192:1856]
                    
                    pred_depth_mono = pred_depth_mono[mask]
                    ratio_mono = np.median(gt_depth) / np.median(pred_depth_mono)
                    ratios_mono.append(ratio_mono)
                    pred_depth_mono *= ratio_mono
                    pred_depth_mono[pred_depth_mono < MIN_VAL] = MIN_VAL
                    pred_depth_mono[pred_depth_mono > MAX_VAL] = MAX_VAL

                    errors_mono.append(compute_errors(gt_depth, pred_depth_mono))

            mean_errors = np.array(errors).mean(0)
            
            
            if mono_flag:
                mean_errors_mono = np.array(errors_mono).mean(0)
                if self.opt.debug:
                    print(ratio, ratio_mono)
        if mono_flag:
            return mean_errors, mean_errors_mono 
        else:
            return mean_errors
    
    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """ 
        # if mono and self.opt.grad_loss:
        #     inputs[("label", 0, 0)] = self.generate_panoptic(inputs[("color", 0, 0)])
            # ori_img_size = inputs[("color", 0, 0)].shape[-2:]
            # for frame_id in self.opt.frame_ids:
            #     self.generate_sem_seg(inputs, frame_id, 0, ori_img_size)

        for scale in range(self.opt.sclm+1):
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            
            
            outputs[("depth", 0, scale)] = depth
            
            # if mono and (self.opt.sem_loss or self.opt.grad_loss or self.opt.sem_mask):
            #     self.generate_sem_seg(inputs, 0, scale, ori_img_size)

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()
                
                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
            
            # generate I_synthesis from It+1->t and It-1->t
            # if is_multi==False and self.opt.temporal:
            #     # if not self.opt.scale_acc:
            #     #     self.generate_synthesised_image(inputs, outputs, scale)
            #     # else:
            #     if scale <= self.opt.sclm:
            #         self.generate_synthesised_image(inputs, outputs, scale)

    def generate_images_pred_dc(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """ 
        for scale in range(self.opt.sclm+1):
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            
            # depth = torch.einsum('bchw, b->bchw', disp, outputs[("scale", scale)] * self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth[:self.opt.batch_size]
            outputs[("depth", -1, scale)] = depth[self.opt.batch_size:-self.opt.batch_size]
            outputs[("depth", 1, scale)] = depth[-self.opt.batch_size:]
            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])
                project_out = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                
                pix_coords = project_out

                outputs[("sample", frame_id, scale)] = pix_coords
                
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
            
            
            # It->t-1 stored in color, -2, scale
            # T = outputs[("cam_T_cam", -1, 0)]
            # if is_multi:
            #     # don't update posenet based on multi frame prediction
            #     T = T.detach()
            
            # cam_points = self.backproject_depth[0](
            #     outputs[("depth", -1, scale)], inputs[("inv_K", 0)])
            # pix_coords = self.project_3d[0](
            #     cam_points, inputs[("K", source_scale)], T)

            # outputs[("sample", -2, scale)] = pix_coords

            # outputs[("color", -2, scale)] = F.grid_sample(
            #     inputs[("color", 0, source_scale)],
            #     outputs[("sample", -2, scale)],
            #     padding_mode="border", align_corners=True)
            
            
                    
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            # print(all_losses.shape) # B, 2, H, W
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            # print(idxs.shape) # B, 1, H, W
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_depth_diff(self, pred, target):
        return abs(pred-target) / (target)
    
    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_grad_loss = 0
        reproj_losses = []
        
        # original loss
        for scale in range(self.opt.sclm+1):
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # differently to Monodepth2, compute mins as we go
            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)


            # if self.opt.avg_reprojection:
            #     reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            # else:
            # differently to Monodepth2, compute mins as we go
            reprojection_loss, frame_idxs = torch.min(reprojection_losses, dim=1, keepdim=True)
            if self.opt.selec_reproj:
                maskm1 = (outputs[("color", -1, scale)].sum(1) < 0.1).detach()
                maskp1 = (outputs[("color", 1, scale)].sum(1) < 0.1).detach()
                maskand = (maskm1 * maskp1).detach()
                reprojection_loss[maskm1.unsqueeze(1)] = (reprojection_losses[:,1,:,:])[maskm1]
                reprojection_loss[maskp1.unsqueeze(1)] = (reprojection_losses[:,0,:,:])[maskp1]
                reprojection_loss[maskand.unsqueeze(1)] = 0
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)
            # if self.step % 50 == 0:
            #     # if self.step < 10 and batch_idx == 0:
            #     #     self.log_time(batch_idx, duration, losses["loss"].cpu().data)
            #     wandb.log({f"Train/loss": losses["loss"]}, step=self.step)
            #     for scale in range(self.opt.sclm+1):
            #         wandb.log({f"Train/loss_{scale}": losses["loss/{}".format(scale)]}, step=self.step)
            
            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                            outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                            (1 - outputs['augmentation_mask'][:self.opt.batch_size]))
                consistency_mask = (1 - reprojection_loss_mask).float()
            
            
            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
            
            if self.opt.loss_pct:
                reproj_pixel_cnt = reprojection_loss_mask.sum()
                percent = reproj_pixel_cnt / (self.opt.batch_size * self.opt.height * self.opt.width)
                if self.opt.debug :
                    print(percent)
                if self.step % 50 == 0 and self.is_main:
                    mode = "m" if is_multi else "t" 
                    wandb.log({f"Train/pp_{mode}_{scale}": percent}, step=self.step)
                
            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        
        total_loss /= (self.opt.sclm+1)
        losses["loss"] = total_loss
        
        return losses, reproj_losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
    
    def compute_median_diff(self, pred1, pred2):
        '''
            pred1 & pred2 .shape = B, 1, H, W
        '''
        median1 = torch.mean(pred1.flatten(1), dim=1)[0]
        median2 = torch.mean(pred2.flatten(1), dim=1)[0]
        
        # l1 diff
        diff = abs(median1 - median2) / (median1 + median2)
        return diff.mean() 
                
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        #     s = 0  # log only max scale
        #     for frame_id in self.opt.frame_ids:
        #         writer.add_image(
        #             "color_{}_{}/{}".format(frame_id, s, j),
        #             inputs[("color", frame_id, s)][j].data, self.step)
        #         if s == 0 and frame_id != 0:
        #             writer.add_image(
        #                 "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                 outputs[("color", frame_id, s)][j].data, self.step)

        #     disp = colormap(outputs[("disp", s)][j, 0])
        #     writer.add_image(
        #         "disp_multi_{}/{}".format(s, j),
        #         disp, self.step)

        #     disp = colormap(outputs[('mono_disp', s)][j, 0])
        #     writer.add_image(
        #         "disp_mono/{}".format(j),
        #         disp, self.step)

        #     if outputs.get("lowest_cost") is not None:
        #         lowest_cost = outputs["lowest_cost"][j]

        #         consistency_mask = \
        #             outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

        #         min_val = np.percentile(lowest_cost.numpy(), 10)
        #         max_val = np.percentile(lowest_cost.numpy(), 90)
        #         lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
        #         lowest_cost = colormap(lowest_cost)

        #         writer.add_image(
        #             "lowest_cost/{}".format(j),
        #             lowest_cost, self.step)
        #         writer.add_image(
        #             "lowest_cost_masked/{}".format(j),
        #             lowest_cost * consistency_mask, self.step)
        #         writer.add_image(
        #             "consistency_mask/{}".format(j),
        #             consistency_mask, self.step)

        #         consistency_target = colormap(outputs["consistency_target/0"][j])
        #         writer.add_image(
        #             "consistency_target/{}".format(j),
        #             consistency_target.squeeze(), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    
    def save_drop_path(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        torch.save(self.model.module.encoder, os.path.join(save_folder, "encoder.pth"))
        torch.save(self.model.module.mono_encoder, os.path.join(save_folder, "mono_encoder.pth"))

    def save_model_debug(self, save_folder):
        """Save model weights to disk
        """
        if self.opt.debug:
            print(f"save model at {save_folder}")
        # if save_step:
        #     save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
        #                                                                                self.step))
        # else:
        #     save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        save_path = os.path.join(save_folder, "model.pth")
        unwrapped_model = self.acc.unwrap_model(self.model)
        self.acc.save(unwrapped_model.state_dict(), save_path)
        # self.acc.save(unwrapped_model.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "track.pth")
        to_save = {}
        to_save['height'] = self.opt.height
        to_save['width'] = self.opt.width
        if not self.opt.notadabins and not self.freeze_tp:
            # save estimates of depth bins
            min_depth, max_depth = self.depth_bin_tracker.compute()
            to_save['min_depth_bin'] = min_depth
            to_save['max_depth_bin'] = max_depth
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


    
    def load_mono_model(self):
        if self.opt.mono_replk:
            path = os.path.join(self.opt.mono_weights_folder, "mono_encoder.pth")
            self.models["mono_encoder"] = torch.load(path)            

            model_list = ['pose_encoder', 'pose', 'mono_depth']
        else: 
            model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
            
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
    
    

    def load_model(self, transfer=False):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        
        model_path = os.path.join(self.opt.load_weights_folder, "model.pth")
        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False) 
        self.model.dc = self.opt.dc  
        if transfer:
            return
        track_path = os.path.join(self.opt.load_weights_folder, "track.pth")
        track_dict = torch.load(track_path, map_location='cpu')
        min_depth_bin = track_dict.get('min_depth_bin')
        max_depth_bin = track_dict.get('max_depth_bin')
        print(min_depth_bin, max_depth_bin)
        self.depth_bin_tracker.load(min_depth_bin, max_depth_bin)
        self.depth_bin_tracker.to(self.device)
        print("after ", self.depth_bin_tracker.min_depth, self.depth_bin_tracker.max_depth)

        # loading adam state
        if self.opt.load_weights_folder is not None:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                try:
                    print("Loading Adam weights")
                    optimizer_dict = torch.load(optimizer_load_path, map_location='cpu')
                    self.model_optimizer.load_state_dict(optimizer_dict)
                except ValueError:
                    print("Can't load Adam - using random")
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
