# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks import MaxPool2d
from torch import nn
import numpy as np
from mmcv.runner import force_fp32
import torch.nn.functional as F
from .. import builder
from mmdet.models.builder import build_backbone
import einops

@DETECTORS.register_module()
class GEOccPre(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=False,
                 class_wise=False,
                 occ_head=None,
                 m2f_head=None,
                 bev_inchannels = 128,
                 bev_outchannels = 512, # no neck
                 use_depth_back_proj = False,
                 use_img_recons_proxy = False,
                 depth_back_projector = None,
                 depth_decoder = None,
                 img_recons_proxy = None,
                 use_depth_sup= True,
                 use_sem_sup = True,
                 nerf_head = None,
                 **kwargs):
        super(GEOccPre, self).__init__(**kwargs)
        self.img_bev_encoder_neck = None
        self.use_mask = use_mask
        self.align_after_view_transfromation = False
        self.bev_inchannels = bev_inchannels
        self.bev_down = ConvModule(
                    self.bev_inchannels,
                    self.bev_inchannels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', ),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
        self.occ_head = builder.build_head(occ_head)
        # self.m2f_head = builder.build_head(m2f_head)
        
        self.use_depth_sup = use_depth_sup
        self.use_sem_sup = use_sem_sup
        self.density_mlp = nn.Sequential(
            nn.Linear(bev_outchannels,bev_outchannels),
            nn.Softplus(),
            nn.Linear(bev_outchannels, 2),
            nn.Softplus(),
        )

        nerf_head.update(
            use_depth_sup=self.use_depth_sup,
            use_sem_sup=self.use_sem_sup
        )
        self.nerf_head = builder.build_backbone(nerf_head)
        self.use_depth_back_proj = use_depth_back_proj
        self.use_img_recons_proxy = use_img_recons_proxy
        if self.use_depth_back_proj:
            self.depth_back_projector = build_backbone(depth_back_projector)
            self.depth_decoder = build_backbone(depth_decoder)
        if self.use_img_recons_proxy:
            self.proxy = build_backbone(img_recons_proxy)
        


    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        img_feats_key_frame = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv, img_feat = \
                        self.prepare_bev_feat(*inputs_curr)
                    ### ADD key frame use semantic
                    depth_key_frame = depth
                    img_feats_key_frame = (feat_curr_iv, img_feat.flatten(0,1))
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv, _ = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # x = self.bev_encoder(bev_feat)

        return bev_feat, depth_key_frame, img_feats_key_frame

    
    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat, None
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        return bev_feat, depth, stereo_feat, x
    
    @force_fp32()
    def bev_encoder(self, x):
        """
        Return: multiscale bev_feat 
        feat_level=3 downsample = 1x,2x,4x
        """
        x = self.img_bev_encoder_backbone(x)
        # x = self.img_bev_encoder_neck(x)
        # if type(x) in [list, tuple]:
        #     x = x[0]
        return x
    
    def extract_bev_feat(self, img_inputs, img_metas, **kwargs):
        bev_feat, depth, img_feats = self.extract_img_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        # up sample img_feats to origin
        # up_feats = F.interpolate(img_feats[0],img_metas[0]['img_shape'],mode='bilinear',align_corners=True)
        img_feat = img_feats[0]
        img_feat = einops.rearrange(img_feat, '(b n) c h w -> b n c h w',b = bev_feat.size(0))
        post_rots, post_trans, bda = img_inputs[4:7]
        post_rots = post_rots[:,:6,...]
        post_trans = post_trans[:,:6,...]
        outs = self.occ_head([img_feat], img_metas, post_rots, post_trans, bda)
        bev_feat = torch.cat([bev_feat, outs],dim=1) #32*2 + 64
        bev_feat = self.bev_down(bev_feat)
        multi_scale_bev_feats = self.bev_encoder(bev_feat)
        multi_scale_bev_feats_reshape = [] # b,c,x,y,z
        for feat in multi_scale_bev_feats:
            feat = feat.permute(0,1,4,3,2)
            multi_scale_bev_feats_reshape.append(feat)
        return depth, multi_scale_bev_feats_reshape, img_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        depth, bev_feat, img_feats = self.extract_bev_feat(img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        # loss
        losses = dict()
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_feat = F.interpolate(bev_feat[-1], scale_factor=8, mode='trilinear', align_corners=True).permute(0,2,3,4,1)
        density_prob = self.density_mlp(voxel_feat)
        density = density_prob[...,0]

        loss_nerf, depth_nerf = self.nerf_head(density, rays=kwargs['rays'], label_depth=kwargs['label_depth'],bda=img_inputs[6])

        losses.update(loss_nerf)
        if self.use_img_recons_proxy:
            loss_rc = self.proxy(bev_feat, img_inputs, depth_nerf, depth_downsample=5, **kwargs)
            losses.update(loss_rc)
        
        return losses
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        depth, bev_feat, _ = self.extract_bev_feat(img_inputs=img, img_metas=img_metas, **kwargs)
        """Test function without augmentaiton."""
        output = self.m2f_head.simple_test(bev_feat, img_metas)
        output_voxels = output['output_voxels'][0]
        score, pred_occ = torch.max(torch.softmax(output_voxels, dim=1), dim=1)
        pred_occ = pred_occ.cpu().numpy().astype(np.uint8)
        return pred_occ
