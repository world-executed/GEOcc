<div align="center">
<h2>GEOcc: Geometrically Enhanced 3D Occupancy Network with Implicit-Explicit Depth Fusion and Contextual Self-Supervision</h2>
Xin Tan<sup>*</sup>, Wenbin Wu<sup>*</sup>, Zhiwei Zhang, Chaojie Fan, Yong Peng, Zhizhong Zhang<sup>+</sup>, Yuan Xie, Lizhuang Ma

<sup>*</sup>equal contribution, <sup>+</sup>corresponding authors

<a href="https://arxiv.org/abs/2405.10591"><img src='https://img.shields.io/badge/arXiv-GEOcc-red' alt='Paper PDF'></a>
</div>


This is a PyTorch/GPU implementation of the paper [GEOcc](https://arxiv.org/abs/2405.10591)

## Introduction
3D occupancy perception holds a pivotal role in recent vision-centric autonomous driving systems by converting surround-view images into integrated geometric and semantic representations within dense 3D grids. Nevertheless, current models still encounter two main challenges: modeling depth accurately in the 2D-3D view transformation stage, and overcoming the lack of generalizability issues due to sparse LiDAR supervision. To address these issues, this paper presents GEOcc, a Geometric-Enhanced Occupancy network tailored for vision-only surround-view perception. Our approach is three-fold: 1) Integration of explicit lift-based depth prediction and implicit projection-based transformers for depth modeling, enhancing the density and robustness of view transformation. 2) Utilization of mask-based encoder-decoder architecture for fine-grained semantic predictions; 3) Adoption of context-aware self-training loss functions in the pertaining stage to complement LiDAR supervision, involving the re-rendering of 2D depth maps from 3D occupancy features and leveraging image reconstruction loss to obtain denser depth supervision besides sparse LiDAR ground-truths. Our approach achieves State-Of-The-Art performance on the Occ3D-nuScenes dataset with the least image resolution needed and the most weightless image backbone compared with current models, marking an improvement of 3.3% due to our proposed contributions. Comprehensive experimentation also demonstrates the consistent superiority of our method over baselines and alternative approaches.

## Get Started

#### Installation and Data preparation

step 1. Follow [install.md](install.md) to install environment.

step 2. Prepare nuScenes dataset as introduced in [nuscenes_det.md](nuscenes_det.md) and occupancy ground truth from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction). The folder should be arranged as :
```
└── nuscenes
    ├── v1.0-trainval
    ├── sweeps
    ├── samples
    └── gts
```

step 3. Create the pkl file for dataset by running:
```shell
python tools/create_data_bevdet.py
```

step 4. Download the CBGS model weights for [BEVDet-R50-4DLongterm-Stereo-CBGS](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) and [BEVDet-STBase-4D-Stereo-512x1408-CBGS](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1). Place the two pth files in the project root directory.


#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```
#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```
