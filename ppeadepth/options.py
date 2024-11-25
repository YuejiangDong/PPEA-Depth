# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This softsemare is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ManyDepth options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="../input")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark",
                                          "cityscapes_preprocessed", "part", "cityscapes"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--depth_binning",
                                 help="defines how the depth bins are constructed for the cost"
                                      "volume. 'linear' is uniformly sampled in depth space,"
                                      "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse', 'log'],
                                 default='log'),
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=96)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--freeze_teacher_and_pose",
                                 action="store_true",
                                 help="If set, freeze the weights of the single frame teacher"
                                      " network and pose network.")
        self.parser.add_argument("--freeze_teacher_epoch",
                                 type=int,
                                 default=150,
                                 help="Sets the epoch number at which to freeze the teacher"
                                      "network and the pose network.")
        self.parser.add_argument("--freeze_teacher_step",
                                 type=int,
                                 default=-1,
                                 help="Sets the step number at which to freeze the teacher"
                                      "network and the pose network. By default is -1 and so"
                                      "will not be used.")
        self.parser.add_argument("--pytorch_random_seed",
                                 default=None,
                                 type=int)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
        self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--mono_weights_folder",
                                 type=str)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=["eigen", "eigen_zhou", "eigen_benchmark", "benchmark", "odom_9",
                                          "odom_10", "cityscapes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--zero_cost_volume",
                                 action="store_true",
                                 help="If set, during evaluation all poses will be set to 0, and "
                                      "so we will evaluate the model in single frame mode")
        self.parser.add_argument('--static_camera',
                                 action='store_true',
                                 help='If set, during evaluation the current frame will also be'
                                      'used as the lookup frame, to simulate a static camera')
        self.parser.add_argument('--eval_teacher',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')

        self.parser.add_argument('--sem_config_path',
                              #    default='./semantic_config_weight/maskformer2_R50_bs16_90k.yaml',
                              #    default='./semantic_config_weight/maskformer2_swin_tiny_bs16_90k.yaml',
                                 default='./semantic_config_weight/maskformer2_swin_large_IN21k_384_bs16_90k.yaml',
                                 #default='./semantic_config_weight/maskformer2_swin_small_bs16_90k.yaml', 
                                 type=str,
                                 help='Configuration file path of the pretrained semantic segmentation model')
        self.parser.add_argument('--sem_weight_path',
                              #    default='./semantic_config_weight/model_final_cc1b1f.pkl',
                              #    default='./semantic_config_weight/model_final_2d58d4.pkl',
                                 default='./semantic_config_weight/model_final_17c1ee.pkl', 
                                 #default='./semantic_config_weight/model_final_fa26ae.pkl', # 
                                 type=str,
                                 help='Path of model weight of the pretrained semantic segmentation model')
        self.parser.add_argument('--sem_loss',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--sem_mask',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--grad_loss',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--grad_loss_weight',
                                 default=0.5,
                                 type=float,
                                 help='Path of model weight of the pretrained semantic segmentation model')
        
        self.parser.add_argument('--num_classes',
                                 type=int,
                                 default=19,
                                 help='If set, the teacher network will be evaluated')
        
        
        self.parser.add_argument('--train_sem',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')

        self.parser.add_argument('--ins_config_path',
                                 default='./instance_config_weight/maskformer2_swin_large_IN21k_384_bs16_90k.yaml',
                                 type=str,
                                 help='Configuration file path of the pretrained instance segmentation model')
        self.parser.add_argument('--ins_weight_path',
                                 default='./instance_config_weight/model_final_dfa862.pkl', 
                                 type=str,
                                 help='Path of model weight of the pretrained semantic segmentation model')
        self.parser.add_argument('--temporal',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--ins_threshold',
                                 default=0.9,
                                 type=float,
                                 help='Path of model weight of the pretrained semantic segmentation model')
        self.parser.add_argument('--main_temporal',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        
        
        self.parser.add_argument('--pan_config_path',
                                 default='./panoptic_config_weight/maskformer2_swin_large_IN21k_384_bs16_90k.yaml',
                                 type=str,
                                 help='Configuration file path of the pretrained instance segmentation model')
        self.parser.add_argument('--pan_weight_path',
                                 default='./panoptic_config_weight/model_final_064788.pkl', 
                                 type=str,
                                 help='Path of model weight of the pretrained semantic segmentation model')
        self.parser.add_argument('--pan',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--new_mask',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument("--update_once",
                                 help='if Multi Loss rebalancing just update the weights once',
                                 action='store_true')         

        self.parser.add_argument('--scale_acc',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--replace',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
     
        # options for wandb & new validate function
        self.parser.add_argument("--name", type=str, default="test")
        self.parser.add_argument("--tags", type=str, default="multi")
        self.parser.add_argument("--validate_every", "--validate-every", type=int, default=1000)
        self.parser.add_argument("--debug", action="store_true")
    
        # options for architecture
        self.parser.add_argument("--replk", action="store_true")
        self.parser.add_argument("--mono_replk", action="store_true")
        self.parser.add_argument("--rep_size", type=str, default='b', choices=['b', 'l', 'xl'])
        self.parser.add_argument("--use_checkpoint", action="store_true")
        self.parser.add_argument("--st", action="store_true")
        self.parser.add_argument("--mono_st", action="store_true")
        self.parser.add_argument("--st2", action="store_true")
        self.parser.add_argument("--mono_st2", action="store_true")
        
        # options for self distill
        self.parser.add_argument("--self_distill", action="store_true")
        
        # options for ensemble
        self.parser.add_argument("--ensemble_eval", action="store_true")
        
        self.parser.add_argument("--cs_eval_path", type=str, default="../cityscapes")
        
        # options for evaluation
        self.parser.add_argument("--eval_cs", action="store_true")
        self.parser.add_argument("--oldver", action="store_true")
        
        self.parser.add_argument("--train_cs", action="store_true")
        self.parser.add_argument("--w1", type=float, default=0.9)
        self.parser.add_argument("--w2", type=float, default=0.85)
        
        self.parser.add_argument("--k", type=int, default=1)
        self.parser.add_argument("--teacher", type=int) # 0: resnet18, old, 1: replk new
        self.parser.add_argument("--main", type=int) # 0: resnet18, 1: replk
        
        self.parser.add_argument("--m1", type=int) # 0: replk, 1:replkmatching, 2: resnet, 3: resnetmatching
        self.parser.add_argument("--m2", type=int) 
        
        self.parser.add_argument("--es_dyn", action="store_true")
        self.parser.add_argument("--sclm", type=int, default=0) 
        
        self.parser.add_argument("--opt_flow", action="store_true")
        self.parser.add_argument("--opt_path", type=str, default="/home/share/dyj/kittiflow")
        
        self.parser.add_argument("--loss_pct", action="store_true")
        
        self.parser.add_argument("--adapter", action="store_true")
        self.parser.add_argument("--g_blk", type=float, default=1.0)
        self.parser.add_argument("--g_ffn", type=float, default=1.0)
        self.parser.add_argument("--mono_trans", action="store_true")
        self.parser.add_argument("--trans", action="store_true")
        self.parser.add_argument("--mono_input", action="store_true")
        self.parser.add_argument("--input", action="store_true")
        
        self.parser.add_argument("--adpt_test", type=int, default=4) # test id for adapter structure
        
        # OPTIONS for SCHEDULER
        self.parser.add_argument("--cos", action="store_true")
        self.parser.add_argument("--step_lr", action="store_true")
        
        
        self.parser.add_argument("--dc", action="store_true")
        self.parser.add_argument("--notadabins", action="store_true")
        
        self.parser.add_argument("--test_scale", action="store_true")
        self.parser.add_argument('--freeze_pose', action='store_true')
        
        self.parser.add_argument('--s2_fullft', action='store_true')
        self.parser.add_argument('--special_fz', action='store_true')
        self.parser.add_argument('--initdc', action='store_true')
        
        self.parser.add_argument('--load_clcb', action='store_true')
        self.parser.add_argument('--saveoff', action='store_true')
        self.parser.add_argument("--validate_from", type=int, default=0) # test id for adapter structure
        self.parser.add_argument("--pose_cnn", action="store_true") 
        self.parser.add_argument("--save_until", type=int, default=0) # test id for adapter structure
        self.parser.add_argument('--separate_load', action='store_true')
        self.parser.add_argument("--ratio", type=float, default=0.25) # test id for adapter structure
        
        self.parser.add_argument("--dec_id", type=int, default=1)
        
        self.parser.add_argument("--dc_r", type=float, default=0.25) # test id for adapter structure
        
        self.parser.add_argument("--ddad", action="store_true")

        # for dynamicdepth
        # Multi(Future) frame cost volume selection
        self.parser.add_argument("--cv_min",
                                 help="If set, cost volume will choose min value from multiple ref frames",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"], )

        # Select Reproj Loss
        self.parser.add_argument('--selec_reproj',
                                 action='store_false',
                                 help='If set, will select reprojection loss by warping holes')
        self.parser.add_argument('--zero_img',
                                 action='store_false',
                                 help='If set, will set the imgs warping hole area as 0 before reprojection loss')
     
        # mask occlusion hole in cost volume
        self.parser.add_argument('--cv_set_1',
                                 action='store_true',
                                 help='If set, will set the occlusion region of cost volume to 1')
        self.parser.add_argument('--cv_pool',
                                 action='store_false',
                                 help='If set, will max pool the occlusion region')
        self.parser.add_argument("--cv_pool_radius",
                                 type=int,
                                 help="the max_pooling (kernel size-1)/2 for occlusion region in cost volume",
                                 default=1)
        self.parser.add_argument("--cv_pool_th",
                                 type=float,
                                 help="the threshold for determing occlusion",
                                 default=0.7)

        # Train teacher only
        self.parser.add_argument('--train_teacher_only',
                                 action='store_true',
                                 help='If set, will only train the teacher net')
        self.parser.add_argument('--no_multi_loss',
                                 action='store_true',
                                 help='If set, will not compute the losses for multi frame branch')
        self.parser.add_argument('--no_warp',
                                 action='store_true',
                                 help='If set, will not warp ref image based on mono teacher depth and pose')
        self.parser.add_argument('--no_teacher_warp',
                                 action='store_false',
                                 help='If set, the warp of dynamic obj in ref image will not be imposed to the teacher itself')
        self.parser.add_argument('--no_reproj_doj',
                                 action='store_true',
                                 help='If set, will not include the dynamic objs area in reprojection loss')

        self.parser.add_argument('--ktf', action='store_true')
        self.parser.add_argument('--dec_only', action='store_true')
        self.parser.add_argument('--dec_ratio', type=float, default=0.25)

        self.parser.add_argument('--fullft_reb', action='store_true')
        self.parser.add_argument('--lps2', action='store_true')
        self.parser.add_argument('--eval', action='store_true')
        
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
