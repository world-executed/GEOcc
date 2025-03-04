# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore,Metric_lidarseg
import torch
import torch.nn.functional as F
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            if index%100==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir + "%d.jpg"%index))

        return self.occ_eval_metrics.count_miou()
    
    def evaluate_lidarseg(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_lidarseg(
            num_classes=17,
            use_lidar_mask=False,
            use_image_mask=False)
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
        

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            lidar2ego_rotation = info['lidar2ego_rotation']
            lidar2ego_translation = info['lidar2ego_translation']
            lidar2ego = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                        inverse=False)
                
            points = np.fromfile(info['lidar_path'], dtype=np.float32)
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                        nusc.get('lidarseg', 
                                            nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                                            )['filename'])
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
            points_label_merge = np.zeros_like(points_label)
            label_merged_map = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26: 
            13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0}
            for key in label_merged_map:
                points_label_merge[points_label==key] = label_merged_map[key]
            points_gt = points_label_merge
            points = points.reshape(-1,5)
            # points_class = get_points_type('LIDAR')
            # points = points_class(
            # points, points_dim=points.shape[-1], attribute_dims=None)
            occ_pred = torch.tensor(occ_pred)
            points = torch.tensor(points)
            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            occ_pred = occ_pred
            points_pred, free_mask = self.get_points(torch.Tensor(gt_semantics),points,lidar2ego)
            noise_mask = points_gt==0
            mask = np.logical_or(free_mask,noise_mask)
            points_pred = points_pred[~mask]
            points_gt = points_gt[~mask]
            self.occ_eval_metrics.add_batch(points_pred, points_gt,None,None)

            # if index%100==0 and show_dir is not None:
            #     gt_vis = self.vis_occ(gt_semantics)
            #     pred_vis = self.vis_occ(occ_pred)
            #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
            #                  os.path.join(show_dir + "%d.jpg"%index))

        return self.occ_eval_metrics.count_miou()
    
    def get_points(self, voxel_preds, points_i,lidar2ego):
        point_cloud_range=np.array([-40, -40, -1.0, 40, 40, 5.4])

        n=points_i.shape[0]
        xyz = points_i[:, :3].float()
        xyz1 = torch.cat([xyz,torch.ones([n,1])],dim=1)
        xyz1 = torch.Tensor(lidar2ego) @ xyz1.T
        xyz = xyz1[:3,:].T
        pc_range_min = point_cloud_range[:3]
        pc_range = point_cloud_range[3:]-point_cloud_range[:3]
        points_i = (xyz - pc_range_min) / pc_range
        # import pdb;pdb.set_trace()
        points_i = (points_i * 2) - 1
        points_i = points_i[..., [2, 1, 0]]
        
        out_of_range_mask = (points_i < -1) | (points_i > 1)
        out_of_range_mask = out_of_range_mask.any(dim=1)
        points_i = points_i.view(1, 1, 1, -1, 3).float()
        
        point_logits_i = F.grid_sample(voxel_preds.float().unsqueeze(0).unsqueeze(1), points_i, mode='nearest',padding_mode='border', align_corners=True)
        point_logits_i = point_logits_i.squeeze().t().contiguous() # [b, n, c]
        point_logits_i = point_logits_i.long().numpy()
        # import pdb;pdb.set_trace()
        free_mask = point_logits_i==17
        return point_logits_i, free_mask

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis