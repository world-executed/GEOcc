import torch
import torch.nn as nn
from .geometry_util import matrix_to_euler_angles 
from .loss_util import compute_photometric_loss, compute_edg_smooth_loss, compute_masked_loss, compute_auto_masks


class ReconstructionLoss(nn.Module):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self,):
        self.frame_ids = [0,-1]
        self.tempo_coeff = 0.1
        self.spatio_coeff = 0.03
        self.spatio_tempo_coeff = 0.1
        self.disparity_smoothness = 0.001
        super(ReconstructionLoss, self).__init__()
    
    def compute_reproj_loss(self, inputs, target_view, cam=0, scale=0, ref_mask=None):
        """
        This function computes reprojection loss using auto mask. 
        """
        reprojection_losses = []
        for frame_id in self.frame_ids[1:]:
            reproj_loss_args = {
                'pred': target_view[('color', frame_id, scale)],
                'target': inputs['color',0, 0][:, cam, ...]
            }
            reprojection_losses.append(
                compute_photometric_loss(**reproj_loss_args)
            )                
            
        reprojection_losses = torch.cat(reprojection_losses, 1)
        reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
        
        identity_reprojection_losses = []
        for frame_id in self.frame_ids[1:]:
            identity_reproj_loss_args = {
                'pred': inputs[('color', frame_id, 0)][:, cam, ...],
                'target': inputs['color',0, 0][:, cam, ...]              
            }
            identity_reprojection_losses.append(
                compute_photometric_loss(**identity_reproj_loss_args)
            )

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        identity_reprojection_losses = identity_reprojection_losses + \
                                        1e-5 * torch.randn(identity_reprojection_losses.shape).cuda()
        identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)             
           
        # find minimum losses
        reprojection_auto_mask = compute_auto_masks(reprojection_loss, identity_reprojection_loss)
        reprojection_auto_mask *= ref_mask

        target_view[('reproj_loss', scale)] = reprojection_auto_mask * reprojection_loss
        target_view[('reproj_mask', scale)] = reprojection_auto_mask
        return compute_masked_loss(reprojection_loss, reprojection_auto_mask)
   
    def compute_smooth_loss(self, inputs, target_view, cam = 0, scale = 0, ref_mask=None):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        color = inputs['color', 0, scale][:, cam, ...]
        disp = target_view[('disp', scale)]
        # dim = 2,3 -> 1,2
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-8)
        return compute_edg_smooth_loss(color, norm_disp)
    
    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        """        
        # self occlusion mask * overlap region mask
        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]         
        }        
        spatio_loss = compute_photometric_loss(**loss_args)
        
        target_view[('overlap_mask', 0, scale)] = spatio_mask         
        return compute_masked_loss(spatio_loss, spatio_mask) 

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
            pred_mask = pred_mask * reproj_loss_mask 
            
            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color',0, 0][:,cam, ...]
            } 
            
            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)
        
        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)    

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)
     
        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask) 
    
    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """        
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]
   
        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...] 
        
        trans_loss = 0.
        angle_loss = 0.
     
        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]    

            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff
        
        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss
    
    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        target_view = outputs[('cam', cam)]

        kargs = {
            'cam': cam,
            'scale': 0,
            'ref_mask': inputs['mask'][:,cam,...]
        }
        use_disp_loss = True if target_view.get(('disp', 0)) else False
        reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
        if use_disp_loss:
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
        spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
        
        kargs['reproj_loss_mask'] = target_view[('reproj_mask', 0)]
        spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)   

        loss_dict['reproj_loss'] = self.tempo_coeff * reprojection_loss
        loss_dict['spatio_loss'] = self.spatio_coeff * spatio_loss
        loss_dict['spatio_tempo_loss'] = self.spatio_tempo_coeff * spatio_tempo_loss
        if use_disp_loss:
            loss_dict['smooth'] = self.disparity_smoothness * smooth_loss
        # pose consistency loss
        # if self.pose_model == 'fsm' and cam != 0:
        #     pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs)
        # else:
        #     pose_loss = 0
            
        # cam_loss += reprojection_loss
        # cam_loss += self.disparity_smoothness * smooth_loss
        # cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss                            
        # cam_loss += self.pose_loss_coeff* pose_loss
        
        ##########################
        # for logger
        ##########################
        # if scale == 0:
        #     loss_dict['reproj_loss'] = reprojection_loss.item()
        #     loss_dict['spatio_loss'] = spatio_loss.item()
        #     loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
        #     loss_dict['smooth'] = smooth_loss.item()
        #     # if self.pose_model == 'fsm' and cam != 0:
            #     loss_dict['pose'] = pose_loss.item()
            
            # log statistics
            # self.get_logs(loss_dict, target_view, cam)                        
    
        # cam_loss /= len(self.scales)
        return loss_dict