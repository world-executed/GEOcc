
from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config, DictAction
from collections import OrderedDict
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.datasets import build_dataset
import torch.nn.functional as F
try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()

def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,      #
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    cam_names=None,
    timestamp=None,
    mode=0,          # mode:0 pred, 1 gt
):
    h, w, z = voxels.shape
    # print(f"voxels.shape:{voxels.shape}")
    # exit()
    if grid is not None:
        grid = grid.astype(np.int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    # if mode == 0: # occupancy pred
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    grid_coords[grid_coords[:, 3] == 17, 3] = 20

    # draw a simple car at the middle
    # car_vox_range = np.array([
    #     [w//2 - 2 - 4, w//2 - 2 + 4],
    #     [h//2 - 2 - 4, h//2 - 2 + 4],
    #     [z//2 - 2 - 3, z//2 - 2 + 3]
    # ], dtype=np.int)
    # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    # car_label = np.zeros([8, 8, 6], dtype=np.int)
    # car_label[:3, :, :2] = 17
    # car_label[3:6, :, :2] = 18
    # car_label[6:, :, :2] = 19
    # car_label[:3, :, 2:4] = 18
    # car_label[3:6, :, 2:4] = 19
    # car_label[6:, :, 2:4] = 17
    # car_label[:3, :, 4:] = 19
    # car_label[3:6, :, 4:] = 17
    # car_label[6:, :, 4:] = 18
    # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    # grid_coords[car_indexes, 3] = car_label.flatten()
    
    # elif mode == 1: # occupancy gt
    #     indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
    #     indexes, pt_index = np.unique(indexes, return_index=True)
    #     pred_pts = pred_pts[pt_index]
    #     grid_coords = grid_coords[indexes]
    #     grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    # elif mode == 2: # lidar gt
    #     indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
    #     indexes, pt_index = np.unique(indexes, return_index=True)
    #     gt_label = pt_label[pt_index]
    #     grid_coords = grid_coords[indexes]
    #     grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    # else:
    #     raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19, # 19
    )

    # colors = np.array(
    #     [
    #         [255, 120,  50, 255],       # barrier              orange
    #         [255, 192, 203, 255],       # bicycle              pink
    #         [255, 255,   0, 255],       # bus                  yellow
    #         [  0, 150, 245, 255],       # car                  blue
    #         [  0, 255, 255, 255],       # construction_vehicle cyan
    #         [255, 127,   0, 255],       # motorcycle           dark orange
    #         [255,   0,   0, 255],       # pedestrian           red
    #         [255, 240, 150, 255],       # traffic_cone         light yellow
    #         [135,  60,   0, 255],       # trailer              brown
    #         [160,  32, 240, 255],       # truck                purple                
    #         [255,   0, 255, 255],       # driveable_surface    dark pink
    #         # [175,   0,  75, 255],       # other_flat           dark red
    #         [139, 137, 137, 255],
    #         [ 75,   0,  75, 255],       # sidewalk             dard purple
    #         [150, 240,  80, 255],       # terrain              light green          
    #         [230, 230, 250, 255],       # manmade              white
    #         [  0, 175,   0, 255],       # vegetation           green
    #         [  0, 255, 127, 255],       # ego car              dark cyan
    #         [255,  99,  71, 255],       # ego car
    #         [  0, 191, 255, 255]        # ego car
    #     ]
    # ).astype(np.uint8)
    colors = np.array(
        [
            [  0,   0,   0, 255],       # other                Black
            [112, 128, 144, 255],       # barrier              Slategrey
            [220,  20,  60, 255],       # bicycle              Crimson
            [255, 127,  80, 255],       # bus                  Coral
            [255, 158,   0, 255],       # car                  Orange
            [233, 150,  70, 255],       # construction_vehicle Darksalmon
            [255,  61,  99, 255],       # motorcycle           Red
            [  0,   0, 230, 255],       # pedestrian           Blue
            [ 47,  79,  79, 255],       # traffic_cone         Darkslategrey
            [255, 140,   0, 255],       # trailer              Darkorange
            [255,  99,  71, 255],       # truck                Tomato                
            [  0, 207, 191, 255],       # driveable_surface    nuTonomy green
            # [175,   0,  75, 255],       # other_flat           dark red
            [175,   0,  75, 255],       # other_flat           
            [ 75,   0,  75, 255],       # sidewalk             
            [112, 180,  60, 255],       # terrain                        
            [222, 184, 135, 255],       # manmade              Burlywood
            [  0, 175,   0, 255],       # vegetation           
            [  0, 255, 127, 255],       # ego car              dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene
    scene.camera.position = [0.75131739, -35.08337438,  16.71378558]
    scene.camera.focal_point = [0.75131739, -34.21734897,  16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    
    if mode==0:
        save_path = os.path.join(save_dirs, 'pred_normal.png')
    elif mode==1:
        save_path = os.path.join(save_dirs, 'gt_normal.png')
    print(f"save_path:{save_path}")
    mlab.savefig(save_path)
    for i, cam_name in enumerate(cam_names):
        # if cam_name != 'CAM_FRONT_LEFT':
        #     continue
        scene.camera.position = cam_positions[i] - np.array([0.0, -2.5, -3.0])
        scene.camera.focal_point = focal_positions[i] - np.array([0.0, -2.5, -3.0])
        scene.camera.view_angle = 60 if i != 3 else 60
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        scene.camera.compute_view_plane_normal()
        scene.render()
        if mode == 0:
            save_path = os.path.join(save_dirs, f'pred_{cam_name}.png')
        elif mode == 1:
            save_path = os.path.join(save_dirs, f'gt_{cam_name}.png')
        print(f"save_path:{save_path}")
        mlab.savefig(save_path)
    scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
    scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0., 1., 0.]
    scene.camera.clipping_range = [0.01, 400.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    if mode==0:
        save_path = os.path.join(save_dirs, 'pred_bev.png')
    elif mode==1:
        save_path = os.path.join(save_dirs, 'gt_bev.png')
    print(f"save_path:{save_path}")
    mlab.savefig(save_path)

    #mlab.show()


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('load_path', default='pred output pkl file')
    parser.add_argument('config', default='test config file path')
    parser.add_argument('--save-path', type=str, default='out/frames')
    parser.add_argument('--frame-idx', type=int, default=0, nargs='+', 
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--vis-gt', action='store_true', help='vis gt or not')
    parser.add_argument('--save-same-path', action='store_true', help='fix save path in same frames')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.config)

    cfg = compat_cfg(cfg)
    dataset = build_dataset(cfg.data.test)

    res = mmcv.load(args.load_path)

    for index in args.frame_idx:
        info = dataset.data_infos[index]
        ego2cam_rts = []
        cam_positions = []
        focal_positions = []
        cam_names = []
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        
        for cam_type, cam_info in info['cams'].items():
            cam_names.append(cam_type)
            # obtain ego to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            ego2cam_rt = np.matmul(lidar2cam_rt, ego2lidar)

            f = 0.0055
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))

        gt_vox = occ_gt['semantics']
        visible_mask = occ_gt['mask_camera'].astype(bool)
        for cls in range(11): # foreground do not use visible mask
            mask = gt_vox == cls
            visible_mask[mask] = True

        pred_vox = res[index]
        if isinstance(pred_vox, dict):
            pred_vox = pred_vox['occ']
        pred_vox[~visible_mask] = 17

        voxel_origin = [-40, -40, -1.0]
        voxel_max = [40.0, 40.0, 5.4]
        grid_size = [200, 200, 16]
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        if args.save_same_path:
            frame_dir = args.save_path
        else:
            frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)

        if args.vis_gt:
            for cam_type, cam_info in info['cams'].items():
                cam_path = cam_info['data_path']
                shutil.copy(cam_path, os.path.join(frame_dir, str(cam_type)+'.jpg'))

        # trans to lidar cord
        gt_vox = gt_vox.transpose(2, 0, 1)  # h w z -> z h w
        gt_vox = np.rot90(gt_vox, 1, [1, 2])
        gt_vox = gt_vox.transpose(1, 2, 0)
        pred_vox = pred_vox.transpose(2, 0, 1)  # h w z -> z h w
        pred_vox = np.rot90(pred_vox, 1, [1, 2])
        pred_vox = pred_vox.transpose(1, 2, 0)
        draw(pred_vox,
             voxel_origin,
             resolution,
            #  grid.squeeze(0).cpu().numpy(), 
            #  pt_label.squeeze(-1),
             save_dirs=frame_dir,
             cam_positions=cam_positions,
             focal_positions=focal_positions,
             cam_names=cam_names,
            #  timestamp=timestamp,
             mode=0)
        if args.vis_gt:
            draw(gt_vox,
                voxel_origin, 
                resolution, 
                #  grid.squeeze(0).cpu().numpy(), 
                #  pt_label.squeeze(-1),
                save_dirs=frame_dir,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                cam_names=cam_names,
                #  timestamp=timestamp,
                mode=1)

