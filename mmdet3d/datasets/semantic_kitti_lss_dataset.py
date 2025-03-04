import numpy as np
import glob
import os

from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset
from .occ_metrics import Metric_mIoU
from tqdm import tqdm

@DATASETS.register_module()
class CustomSemanticKITTILssDataset(SemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, split, camera_used, occ_size, pc_range, 
                 load_continuous=False, multi_adj_frame_id_cfg=(1,1+1,1), 
                 *args, **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        
        self.load_continuous = load_continuous
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
            "test": ["08"],
            "test-submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        
        self.sequences = self.splits[split]
        self.n_classes = 20
        super().__init__(*args, **kwargs)
        self._set_group_flag()
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
    
    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out

    @staticmethod
    def read_pose(pose_path):
        poses = []
        with open(pose_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                pose = np.identity(4)
                pose[:3,:4] = np.array([float(x) for x in line.split()]).reshape(3,4)
                poses.append(pose)
        return poses

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            poses = self.read_pose(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence)
                        
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        "pose": poses[int(img_id)]
                    })
                
        scans = sorted(scans, key = lambda x:(x["sequence"],x["frame_id"])) # sort by time
        return scans  # return to self.data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        # init for pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']
        return None if info is None else np.load(info)

    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
            "pose": pose,
        '''
        
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )
        
        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        poses = []
        
        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            ints_mat = info['P{}'.format(int(cam_type))].copy()
            exts_mat = info['T_velo_2_cam'].copy()
            shift = ints_mat[:3, 3]
            ints = ints_mat[:3, :3]
            rots = exts_mat[:3, :3]
            trans = exts_mat[:3, 3]
            # adding shift to trans
            exts_mat[:3, 3] -= rots@np.linalg.inv(ints)@shift
            cam_intrinsics.append(ints)
            lidar2cam_rts.append(exts_mat)
            poses.append(info['pose'])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                poses=poses,
            ))
        
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index)

        # info_adj
        info_adj_list = self.get_adj_info(info, index)
        input_dict.update(dict(adjacent=info_adj_list))

        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        assert self.multi_adj_frame_id_cfg[0] == 1
        assert self.multi_adj_frame_id_cfg[2] == 1
        adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['sequence'] == info[
                    'sequence']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])

        return info_adj_list
        
    # def evaluate(self, results, logger=None, **kwargs):
    #     if results is None:
    #         logger.info('Skip Evaluation')
        
    #     if 'ssc_scores' in results:
    #         # for single-GPU inference
    #         ssc_scores = results['ssc_scores']
    #         class_ssc_iou = ssc_scores['iou_ssc'].tolist()
    #         res_dic = {
    #             "SC_Precision": ssc_scores['precision'].item(),
    #             "SC_Recall": ssc_scores['recall'].item(),
    #             "SC_IoU": ssc_scores['iou'],
    #             "SSC_mIoU": ssc_scores['iou_ssc_mean'],
    #         }
    #     else:
    #         # for multi-GPU inference
    #         assert 'ssc_results' in results
    #         ssc_results = results['ssc_results']
    #         completion_tp = sum([x[0] for x in ssc_results])
    #         completion_fp = sum([x[1] for x in ssc_results])
    #         completion_fn = sum([x[2] for x in ssc_results])
            
    #         tps = sum([x[3] for x in ssc_results])
    #         fps = sum([x[4] for x in ssc_results])
    #         fns = sum([x[5] for x in ssc_results])
            
    #         precision = completion_tp / (completion_tp + completion_fp)
    #         recall = completion_tp / (completion_tp + completion_fn)
    #         iou = completion_tp / \
    #                 (completion_tp + completion_fp + completion_fn)
    #         iou_ssc = tps / (tps + fps + fns + 1e-5)
            
    #         class_ssc_iou = iou_ssc.tolist()
    #         res_dic = {
    #             "SC_Precision": precision,
    #             "SC_Recall": recall,
    #             "SC_IoU": iou,
    #             "SSC_mIoU": iou_ssc[1:].mean(),
    #         }
        
    #     class_names = [
    #         'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    #         'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    #         'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    #         'pole', 'traffic-sign'
    #     ]
    #     for name, iou in zip(class_names, class_ssc_iou):
    #         res_dic["SSC_{}_IoU".format(name)] = iou
        
    #     eval_results = {}
    #     for key, val in res_dic.items():
    #         eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
    #     eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
    #     if logger is not None:
    #         logger.info('SemanticKITTI SSC Evaluation')
    #         logger.info(eval_results)
        
    #     return eval_results
        
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=20,
            use_lidar_mask=False,
            use_image_mask=False,
            exclude_zero=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt = np.load(info['voxel_path'])
            
            gt_semantics = occ_gt
            mask_lidar = None
            mask_camera = None
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, None, None)

            if index%100==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir + "%d.jpg"%index))

        return self.occ_eval_metrics.count_miou()