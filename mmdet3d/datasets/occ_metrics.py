import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]

class Metric_lidarseg():
    def __init__(self,
                 save_dir='.',
                 num_classes=17,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['other','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.schist = np.zeros((2,2))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        miou = str(round(np.nanmean(mIoU[1:self.num_classes]) * 100, 2))
        print(f'===> mIoU of {self.cnt} samples: ' + miou)
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        ret = dict(zip(self.class_names, mIoU))
        ret['miou'] = miou

        return ret

class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 exclude_zero=False,
                 ):
        if num_classes == 18:
            self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        elif num_classes == 20:
            self.class_names = [
                'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
                'pole', 'traffic-sign'
            ]
        else:
            raise NotImplementedError
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.exclude_zero = exclude_zero

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.schist = np.zeros((2,2))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        if not self.exclude_zero:
            k = (gt >= 0) & (gt < n_cl)  # exclude 255
        else:
            k= (gt > 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        b_pred = np.zeros_like(masked_semantics_pred)
        b_gt = np.zeros_like(masked_semantics_gt)
        b_pred[masked_semantics_pred!=17] = 1
        b_gt[masked_semantics_gt!=17] = 1
        _, _schist = self.compute_mIoU(b_pred, b_gt, 2)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist
        self.schist += _schist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        miou = str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2))
        print(f'===> mIoU of {self.cnt} samples: ' + miou)
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        IoU = self.per_class_iu(self.schist)
        iou = str(round(np.nanmean(IoU[1]) * 100, 2))

        ret = dict(zip(self.class_names, mIoU))
        ret['miou'] = miou
        ret['iou'] = iou
        return ret

import torch
import torch.nn.functional as F
from collections import defaultdict
class Metric_depth():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.min_depth = 1
        self.max_depth = 45
        self._metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, depth_pred, depth_gt):
        # depth_pred (B, 1, h, w)
        error_metric_dict = defaultdict(float)
        _, _, h, w = depth_gt.shape
        depth_pred = F.interpolate(depth_pred, [h, w], mode='bilinear', align_corners=False)
        depth_pred = torch.clamp(depth_pred, self.min_depth, self.max_depth)
        mask = (depth_gt > self.min_depth) * (depth_gt < self.max_depth)
        mask = mask.bool()
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_pred_metric = torch.clamp(depth_pred, self.min_depth, self.max_depth)
        depth_errors_metric = self.cal_depth_error(depth_pred_metric, depth_gt)
        for i in range(len(depth_errors_metric)):
            key = self._metric_names[i]
            error_metric_dict[key] += depth_errors_metric[i]
        
        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy()/6

        return error_metric_dict

    def cal_depth_error(self, pred, target):
        abs_rel = torch.mean(torch.abs(pred-target) / target)
        sq_rel = torch.mean((pred-target).pow(2) / target)
        rmse = torch.sqrt(torch.mean(pred-target).pow(2))
        rmse_log = torch.sqrt(torch.mean((torch.log(target)-torch.log(pred)).pow(2)))
        thresh = torch.max((target/pred),(pred/target))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    def get_downsampled_gt_depth(self, gt_depths, downsample):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)

        gt_depths = (gt_depths - (self.depth_cfg[0] -
                                    self.depth_cfg[2])) / \
                    self.depth_cfg[2]

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        miou = str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2))
        print(f'===> mIoU of {self.cnt} samples: ' + miou)
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        ret = dict(zip(self.class_names, mIoU))
        ret['miou'] = miou
        return ret


class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))


