# PPEA-Depth
This repository is the official implementation of PPEA-Depth: Progressive Parameter-Efficient Adaptation for Self-Supervised Monocular Depth Estimation (AAAI 2024).

In this work, we propose a new training scheme for better self-supervised depth estimation on challenging datasets including dynamic scenes. 

Stage 1: 
+ Train on a dataset primarily composed of static scenes;
+ Keep Frozen: encoder;
+ Tuning: encoder adapter + decoder.

Stage 2:
+ Train on more intricate datasets involving dynamic scenes;
+ Keep Frozen: encoder + decoder;
+ Tuning: encoder adapter + decoder adapter.
  

## :zap: Quick start
### Data Preparation
Please refer to the repositories of [ManyDepth](https://github.com/nianticlabs/manydepth) for dataset preparing. Download [gt depths of CityScapes](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), and unzip into folder `./split/cityscapes`.


### Environment
+ python 3.8
+ accelerate == 0.18.0
+ torch == 1.10.0+cu113 torchaudio == 0.10.0+cu113 torchvision == 0.11.1+cu113
+ torch-scatter == 2.0.9 torch-sparse == 0.6.13
+ torchmetrics == 0.11.1


### Training
```
# For Stage 1 on KITTI
accelerate launch --multi_gpu -m ppeadepth.train --adapter --use_checkpoint --validate_every 3000 --num_epochs 30

# For Stage 2 on CityScapes
accelerate launch --multi_gpu -m ppeadepth.train --train_cs --dc --adapter --use_checkpoint --validate_every 1000 --num_workers 5 --learning_rate 1e-5
```


### Evaluation
```
# on KITTI
accelerate launch --multi_gpu -m ppeadepth.train --adapter --use_checkpoint --eval --rep_size l --load_weights_folder <model_path>

# on CityScapes
accelerate launch --multi_gpu -m ppeadepth.train --train_cs --dc --adapter --use_checkpoint --validate_every 1000 --eval --load_weights_folder <model_path>
```


## :file_folder: Pre-Trained Models
|                            Models                            | AbsRel | SqRel| RMSE | RMSElog | a1 | a2| a3|
| :----------------------------------------------------------: | :----: | :----: | :----: |:----: |:----: |:----: |:----: |
| [KITTI](https://cloud.tsinghua.edu.cn/d/9567c5e132e14c239e7a/) |0.088 |0.649 | 4.105 | 0.167| 0.917 | 0.968 | 0.984 |
| [CityScapes](https://cloud.tsinghua.edu.cn/d/e4485e642c6848a4ae3a/) | 0.100 | 0.976 | 5.673 | 0.152 | 0.904 | 0.977 | 0.992 |

