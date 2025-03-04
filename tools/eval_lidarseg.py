import argparse
import mmcv
import numpy as np
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataloader, build_dataset
import torch.nn.functional as F
from mmdet3d.datasets.occ_metrics import Metric_lidarseg
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl', help='test pkl file path') 
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl)
    print(dataset.evaluate_lidarseg(outputs))

# label_name = {
#     0: 'noise', 
#     1: 'animal', 
#     2: 'human.pedestrian.adult', 
#     3: 'human.pedestrian.child', 
#     4: 'human.pedestrian.construction_worker', 
#     5: 'human.pedestrian.personal_mobility', 
#     6: 'human.pedestrian.police_officer', 
#     7: 'human.pedestrian.stroller', 
#     8: 'human.pedestrian.wheelchair', 
#     9: 'movable_object.barrier', 
#     10: 'movable_object.debris', 
#     11: 'movable_object.pushable_pullable', 
#     12: 'movable_object.trafficcone', 
#     13: 'static_object.bicycle_rack', 
#     14: 'vehicle.bicycle', 
#     15: 'vehicle.bus.bendy', 
#     16: 'vehicle.bus.rigid', 
#     17: 'vehicle.car', 
#     18: 'vehicle.construction', 
#     19: 'vehicle.emergency.ambulance', 
#     20: 'vehicle.emergency.police', 
#     21: 'vehicle.motorcycle', 
#     22: 'vehicle.trailer', 
#     23: 'vehicle.truck', 
#     24: 'flat.driveable_surface', 
#     25: 'flat.other', 
#     26: 'flat.sidewalk', 
#     27: 'flat.terrain', 
#     28: 'static.manmade', 
#     29: 'static.other', 
#     30: 'static.vegetation', 
#     31: 'vehicle.ego'}

# label_map = {
#     'animal':0, 
#     'human.pedestrian.personal_mobility':0, 
#     'human.pedestrian.stroller':0, 
#     'human.pedestrian.wheelchair':0, 
#     'movable_object.debris':0,
#     'movable_object.pushable_pullable':0, 
#     'static_object.bicycle_rack':0, 
#     'vehicle.emergency.ambulance':0, 
#     'vehicle.emergency.police':0, 
#     'noise':0, 
#     'static.other':0, 
#     'vehicle.ego':0,
#     'movable_object.barrier':1, 
#     'vehicle.bicycle':2,
#     'vehicle.bus.bendy':3,
#     'vehicle.bus.rigid':3,
#     'vehicle.car':4,
#     'vehicle.construction':5,
#     'vehicle.motorcycle':6,
#     'human.pedestrian.adult':7,
#     'human.pedestrian.child':7,
#     'human.pedestrian.construction_worker':7,
#     'human.pedestrian.police_officer':7,
#     'movable_object.trafficcone':8,
#     'vehicle.trailer':9,
#     'vehicle.truck':10,
#     'flat.driveable_surface':11,
#     'flat.other':12,
#     'flat.sidewalk':13,
#     'flat.terrain':14,
#     'static.manmade':15,
#     'static.vegetation':16
# }
label_merged_map = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26:
13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0}


def get_points(voxel_preds, points_i, point_cloud_range):
    n=points_i.shape[0]
    points_label = np.zeros(n)
    for key in label_merged_map:
        points_label[points_i[:,-1]==key] = label_merged_map[key]
    
    pc_range_min = point_cloud_range[:3]
    pc_range = point_cloud_range[3:]-point_cloud_range[:3]
    points_i = (points_i[:, :3].float() - pc_range_min) / pc_range
    points_i = (points_i * 2) - 1
    points_i = points_i[..., [2, 1, 0]]
    
    out_of_range_mask = (points_i < -1) | (points_i > 1)
    out_of_range_mask = out_of_range_mask.any(dim=1)
    points_i = points_i.view(1, 1, 1, -1, 3)
    point_logits_i = F.grid_sample(voxel_preds, points_i, mode='bilinear', 
                            padding_mode='border', align_corners=True)
    point_logits_i = point_logits_i.squeeze().t().contiguous() # [b, n, c]
    # point_logits.append(point_logits_i)
def eval_lidarseg(cfg,occ_results):
    point_cloud_range = np.array(cfg.point_cloud_range)
    occ_eval_metrics = Metric_lidarseg(
    num_classes=16,
    use_lidar_mask=False,
    use_image_mask=False)

    print('\nStarting Evaluation...')
    for index, occ_pred in enumerate(tqdm(occ_results)):
        point_pred, point_gt = get_points(occ_pred,None ,point_cloud_range)
        occ_eval_metrics.add_batch(point_pred, point_gt, None, None)

        # if index%100==0 and show_dir is not None:
        #     gt_vis = self.vis_occ(gt_semantics)
        #     pred_vis = self.vis_occ(occ_pred)
        #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
        #                     os.path.join(show_dir + "%d.jpg"%index))

    return occ_eval_metrics.count_miou()
    

if __name__ == '__main__':
    main()