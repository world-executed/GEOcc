from collections import defaultdict
import torch
import torch.nn as nn
from .pose import Pose
from .view_rendering import ViewRendering
from .reconstruction_loss import ReconstructionLoss
import einops
import torch.nn.functional as F

from mmdet.models import BACKBONES
@BACKBONES.register_module()
class ReconstructionProxy(nn.Module):
    def __init__(self,
                 num_cams=6):
        super(ReconstructionProxy, self).__init__()
        self.num_cams = num_cams
        self.pose = Pose()
        self.view_rendering = ViewRendering()
        self.losses = ReconstructionLoss()

    def forward(self, bev_feats, img_inputs, pred_disp, pred_depth, **kwargs):
        (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda) = img_inputs[:7]
        inputs = {}
        inputs['bev_feat'] = bev_feats[0].clone()
        inputs['extrinsics'] = sensor2egos[:,:6,...]
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        inputs[('color', 0, 0)] = imgs[:,:6,...]
        inputs[('color', -1, 0)] = imgs[:,6:12,...]
        mask_shape = list(inputs[('color', 0, 0)].shape)
        mask_shape[2] = 1
        inputs['mask'] = torch.ones(mask_shape).cuda()
        inputs[('K', 0)] = intrins[:,:6,...]
        inputs[('inv_K', 0)] = torch.inverse(inputs[('K', 0)])
        inputs['post_rots'] = post_rots
        inputs['post_trans'] = post_trans
        inputs['bda'] = bda 
        outputs = self.estimate_pose(inputs) 
        
        pred_disp = einops.rearrange(pred_disp, '(b n) h w -> b n 1 h w',n=self.num_cams)
        pred_depth = einops.rearrange(pred_depth, '(b n) h w -> b n h w',n=self.num_cams)
        for cam in range(self.num_cams):
            outputs[('cam', cam)][('disp', 0)] = pred_disp[:,cam,...]
            outputs[('cam', cam)][('depth', 0)] = pred_depth[:,cam,...]
        losses = self.compute_losses(inputs, outputs)
        return losses 
    
    def estimate_pose(self, inputs):
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        pose_pred = self.pose.compute_pose(inputs)                

        for cam in range(self.num_cams):       
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])              

        return outputs

    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """          
        cam_loss_dict = defaultdict(float)
        # generate image and compute loss per cameara
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            loss_dict = self.losses(inputs, outputs, cam)
            for k, v in loss_dict.items():
                cam_loss_dict[k] += v /self.num_cams
                # k = '{}_{}'.format(k, cam)
                # cam_loss_dict[k] = v / self.num_cams
      
        return cam_loss_dict

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)  

from mmdet.models import BACKBONES
@BACKBONES.register_module()
class ReconstructionProxyNerf(ReconstructionProxy):
    def __init__(self,
                 num_cams=6,
                 render_downsample=10):
        super(ReconstructionProxyNerf, self).__init__()
        self.num_cams = num_cams
        self.pose = Pose(use_gt=True)
        self.view_rendering = ViewRendering(height=900//render_downsample,width=1600//render_downsample)
        self.losses = ReconstructionLoss()
        self.render_downsample = render_downsample

    def forward(self, bev_feats, img_inputs, pred_depth, **kwargs):
        (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda) = img_inputs[:7]
        img_ori = kwargs.get('img_ori')
        img_ori = img_ori.permute(0,1,4,2,3)
        intrins[...,:2,:]/=self.render_downsample


        inputs = {}
        inputs['bev_feat'] = bev_feats[0].clone()
        inputs['extrinsics'] = sensor2egos[:,:6,...]
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        inputs[('color', 0, 0)] = img_ori[:,:6,...]
        inputs[('color', -1, 0)] = img_ori[:,6:12,...]
        mask_shape = list(inputs[('color', 0, 0)].shape)
        mask_shape[2] = 1
        inputs['mask'] = torch.ones(mask_shape).cuda()
        # mask no depth
        # inputs['mask'] = torch.where(pred_depth.unsqueeze(2)==0,torch.zeros_like(inputs['mask']), inputs['mask'])
        # pred_depth = torch.where(pred_depth==0,torch.ones_like(pred_depth)*1e5,pred_depth)
        inputs[('K', 0)] = intrins[:,:6,...]
        inputs[('inv_K', 0)] = torch.inverse(inputs[('K', 0)])
        eye_rots = torch.eye(3).reshape(1,1,3,3).expand_as(post_rots).cuda()
        zero_trans = torch.zeros(3).reshape(1,1,3).expand_as(post_trans).cuda()
        inputs['post_rots'] = eye_rots
        inputs['post_trans'] = zero_trans
        inputs['bda'] = bda 
        inputs['sensor2egos'] = sensor2egos
        inputs['ego2globals'] = ego2globals
        outputs = self.estimate_pose(inputs) # use gt pose
        
        # pred_depth = einops.rearrange(pred_depth, '(b n) h w -> b n h w',n=self.num_cams)
        for cam in range(self.num_cams):
            outputs[('cam', cam)][('depth', 0)] = pred_depth[:,cam,...]
        losses = self.compute_losses(inputs, outputs)
        return losses 
    
    # def estimate_pose(self, inputs): move to pose