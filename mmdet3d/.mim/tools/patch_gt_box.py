# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from tqdm import tqdm
config = '/data4/wwb/BEVDet/configs/bevdet_occ_ie/bevdet-occ-ie-prompt.py'

cfg = Config.fromfile(config)

cfg = compat_cfg(cfg)

# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

cfg.model.pretrained = None

# if args.gpu_ids is not None:
#     cfg.gpu_ids = args.gpu_ids[0:1]
#     warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
#                     'Because we only support single GPU mode in '
#                     'non-distributed testing. Use the first GPU '
#                     'in `gpu_ids` now.')
# else:
#     cfg.gpu_ids = [args.gpu_id]

# init distributed env first, since logger depends on the dist info.
# if args.launcher == 'none':
#     distributed = False
# else:
#     distributed = True
#     init_dist(args.launcher, **cfg.dist_params)

test_dataloader_default_args = dict(
    samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}

grid_lower_bound = torch.Tensor([-40, -40, -1.0])
grid_interval= torch.Tensor([0.4, 0.4, 0.4])
cls_map = {5: 1, 7: 2, 3: 3, 0: 4, 2: 5, 6: 6, 8: 7, 9: 8, 4: 9, 1: 10}
# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, **test_loader_cfg)
result = mmcv.load('/data4/wwb/BEVDet/work_dirs/out/occ-ie-e12.pkl')
for batch_id, data in tqdm(enumerate(data_loader)):
    # import pdb;pdb.set_trace()
    img_meta = data['img_metas'][0].data[0][0]
    bboxes = img_meta.get('gt_bboxes_3d').data
    labels = img_meta.get('gt_labels_3d').data.cuda()
    coord = bboxes.corners.cuda()
    # only one box
    if coord.dim()==2:
        coord = coord.unsqueeze(0)
    # coord shift
    assert coord.dim()==3
    N, P, _ = coord.shape
    coord = torch.cat([coord,torch.ones(N,P,1).cuda()],dim=2)
    coord = coord.unsqueeze(3)
    lidar2ego = img_meta['lidar2ego']
    lidar2ego = torch.Tensor(lidar2ego).cuda()
    lidar2ego = lidar2ego.repeat(N, P, 1, 1)
    coord = lidar2ego @ coord
    coord = coord.squeeze(3)[...,:3]
    coord = ((coord - grid_lower_bound.to(coord)) /
        grid_interval.to(coord)).int()
    # filter outer bound
    coord[coord<0]=0
    coord[coord>200]=200
    coord[...,2][coord[...,2]>16]=16
    for vertex, label in zip(coord, labels):
        min_coord = torch.min(vertex, dim=0).values.cpu().numpy()
        max_coord = torch.max(vertex, dim=0).values.cpu().numpy()
        x_range, y_range, z_range = zip(min_coord, max_coord)
        result[batch_id][slice(*x_range), slice(*y_range), slice(*z_range)] = cls_map[label.item()]

mmcv.dump(result, '/data4/wwb/BEVDet/work_dirs/out/occ-ie-e12-new.pkl')