import torch
import torch.nn as nn
from collections import OrderedDict

from mmcv.cnn.bricks.conv_module import ConvModule
from .geometry_util import vec_to_matrix


class Pose(nn.Module):
    """
    Class for multi-camera pose calculation 
    """
    def __init__(self, 
                 num_cams=6,
                 spatio=True,
                 spatio_temporal=True,
                 use_gt=False,
                 ):
        super(Pose, self).__init__()
        self.num_cams = num_cams
        self.spatio = spatio
        self.spatio_temporal = spatio_temporal
        self.use_gt = use_gt
        if not use_gt:
            self.pose_decoder = PoseDecoder(num_ch_enc = [192],
                                            num_input_features=1, 
                                            num_frames_to_predict_for=1, 
                                            stride=2)
            self.bev_shape = (1, 192, 100, 100, 8)
                    # self.reduce_dim = nn.Sequential(*conv2d(encoder_dims, 256, kernel_size=3, stride = stride).children(),
                    #                         *conv2d(256, feat_out_dim, kernel_size=3, stride = stride).children())          
            self.bev_reduce_dim = ConvModule(self.bev_shape[1]*self.bev_shape[4],
                                            self.bev_shape[1],
                                            kernel_size=1,
                                            padding=0,
                                            conv_cfg=dict(type='Conv2d'),)
        self.rel_cam_list = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]}


    def compute_pose(self, inputs):
        """
        This function computes multi-camera posse in accordance with the network structure.
        """
        if self.use_gt:
            pose = self.get_single_pose_gt(inputs)
        else:
            pose = self.get_single_pose(inputs)
        pose = self.distribute_pose(pose, inputs['extrinsics'], inputs['extrinsics_inv'])
        return pose
    
    def get_single_pose(self, inputs):
        """
        This function computes pose for a single camera.
        """

        bev_feat = inputs['bev_feat']
        # bda
        bda = inputs['bda']
        for bs in range(bda.shape[0]):
            flip_x = bda[bs][0][0]==-1
            flip_y = bda[bs][1][1]==-1 # bug
            flip_dims = []
            if flip_x:
                flip_dims.append(2)
            if flip_y:
                flip_dims.append(3)
            bev_feat[bs] = bev_feat[bs].flip(dims=flip_dims)
        # b,c,x,y,z -> b,cz,x,y
        bev_feat = torch.cat(bev_feat.unbind(4),dim=1)
        bev_feat = self.bev_reduce_dim(bev_feat)
        # frame_ids [-1, 0]
        output = {}
        axis_angle, translation = self.pose_decoder([[bev_feat]])
        # translation = torch.clamp(translation, -4.0, 4.0)
        output[('cam_T_cam', 0, -1)] = vec_to_matrix(axis_angle[:, 0], translation[:, 0], invert=True)
        
        # for f_i in self.frame_ids[1:]:
        #     # To maintain ordering we always pass frames in temporal order
        #     frame_ids = [-1, 0] if f_i < 0 else [0, 1]
        #     axisangle, translation = net(inputs, frame_ids, cam)
        #     output[('cam_T_cam', 0, f_i)] = vec_to_matrix(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))            
        return output
    
    def get_single_pose_gt(self, inputs):
        sensor2egos = inputs['sensor2egos']
        ego2globals = inputs['ego2globals']
        B = sensor2egos.shape[0]
        # sensor2egos = sensor2egos.view(B, 6, -1, 4, 4)
        # ego2globals = ego2globals.view(B, 6, -1, 4, 4)
        output = {}
        # output['cam_T_cam', 0, -1] = torch.inverse(T) # t->(t-1)
        # T 应该是sensor坐标系下的t->(t-1) 而不是ego坐标系下
        # c[0][t-1] = T @ c[0][t]
        # sensor2ego(extrinsic) not change with time
        # T =    ego2sensor[0][1] @ global2ego[0][1] @ ego2global[0][0] @ sensor2ego[0][0]
        ref_ext = sensor2egos[:,0,...].double()
        ref_ext_inv = torch.inverse(ref_ext)
        T = ref_ext_inv @ torch.inverse(ego2globals[:,6,...].double()) @ ego2globals[:,0,...].double() @ ref_ext
        output['cam_T_cam', 0, -1] = T 
        return output
        
    def distribute_pose(self, poses, exts, exts_inv):
        """
        This function distrubutes pose to each camera by using the canonical pose and camera extrinsics.
        (default: reference camera 0)
        """
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam',cam)] = {}
        # Refernce camera(canonical)
        ref_ext = exts[:, 0, ...]
        ref_ext_inv = exts_inv[:, 0, ...]

        ref_T = poses['cam_T_cam', 0, -1].float() # canonical pose      
        # Relative cameras(canonical)            
        for cam in range(self.num_cams):
            cur_ext = exts[:,cam,...]
            cur_ext_inv = exts_inv[:,cam,...]                
            cur_T = cur_ext_inv @ ref_ext @ ref_T @ ref_ext_inv @ cur_ext

            outputs[('cam',cam)][('cam_T_cam', 0, -1)] = cur_T            
        return outputs
    
    def compute_relative_cam_poses(self, inputs, outputs, cam):
        """
        This function computes spatio & spatio-temporal transformation for images from different viewpoints.
        """
        ref_ext = inputs['extrinsics'][:, cam, ...]
        target_view = outputs[('cam', cam)]
        
        rel_pose_dict = {}
        # precompute the relative pose
        if self.spatio:
            # current time step (spatio)
            for cur_index in self.rel_cam_list[cam]:
                # for partial surround view training
                if cur_index >= self.num_cams:
                    continue

                cur_ext_inv = inputs['extrinsics_inv'][:, cur_index, ...]
                rel_pose_dict[(0, cur_index)] = torch.matmul(cur_ext_inv, ref_ext)

        if self.spatio_temporal:
            # different time step (spatio-temporal)
            for cur_index in self.rel_cam_list[cam]:
                # for partial surround view training
                if cur_index >= self.num_cams:
                    continue

                T = target_view[('cam_T_cam', 0, -1)]
                # assuming that extrinsic doesn't change
                rel_ext = rel_pose_dict[(0, cur_index)]
                rel_pose_dict[(-1, cur_index)] = torch.matmul(rel_ext, T) # using matmul speed up
        return rel_pose_dict
    
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation