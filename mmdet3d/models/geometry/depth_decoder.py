import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
import einops

from mmdet.models import BACKBONES
@BACKBONES.register_module()
class DepthDecoder(nn.Module):
    def __init__(self, 
                 img_level=3,
                 img_in_dim=[3, 256, 256],
                 img_height=256,
                 img_width=704,
                 depth_in_dim=256,
                 depth_out_dim=1,
                 img_downsample=[1, 4, 4],
                 depth_cfg=[1.0, 45.0, 0.5],
                 loss_depth_weight=0.05,
                 back_proj_loss_weight=0.05,
                 out_scale=[0,1,2]):
        super(DepthDecoder,self).__init__()
        self.img_level = img_level
        self.scale_facter = img_downsample
        self.depth_cfg = depth_cfg
        self.D = (depth_cfg[1]-depth_cfg[0])/depth_cfg[2]
        self.D = (int)(self.D)
        self.width = img_width
        self.height = img_height
        self.min_depth = depth_cfg[0]
        self.max_depth = depth_cfg[1]
        self.focal_length_scale = 300
        self.loss_depth_weight = loss_depth_weight
        self.back_proj_loss_weight = back_proj_loss_weight
        self.out_scale = out_scale
        self.lateral_conv = ModuleList()
        self.conv = ModuleList()
        self.disp = ModuleList()
        conv_cfg = dict(type='Conv2d')
        norm_cfg = dict(type='GN', num_groups=32)
        act_cfg = None
        if depth_out_dim == 1:
            act_cfg = dict(type='Sigmoid') # disp need sigmoid
        for i in range(img_level):
            self.lateral_conv.append(ConvModule(depth_in_dim,
                                                depth_in_dim,
                                                kernel_size=3,
                                                padding=1,
                                                conv_cfg=conv_cfg,
                                                act_cfg=dict(type='ELU')))
            self.conv.append(ConvModule(depth_in_dim+img_in_dim[i],
                                        depth_in_dim,
                                        kernel_size=3,
                                        padding=1,
                                        conv_cfg=conv_cfg,
                                        act_cfg=dict(type='ELU')))
            if i in self.out_scale:
                self.disp.append(ConvModule(depth_in_dim,
                                            depth_out_dim,
                                            kernel_size=3,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            act_cfg=act_cfg))
        # F.interpolate(x, scale_factor=2, mode='nearest')
    def forward(self, inputs):
        """
        inputs: multi_lvl_img_feat, depth_encode
        in_level = 3
        img_feat:
        (bn, 3,   256, 704)
        (bn, 256, 64, 176)
        (bn, 256, 16, 44)
        depth_feat:
        (bn, 256, 16, 44)
        """
        # d=lconv(d)
        # d=conv(d+img(2))   disp(d) 16x44
        # d=up(d)
        # d=lconv(d)
        # d=conv(d+img(1))   disp(d) 64x176
        # d=up(d)
        # d=lconv(d)
        # d=conv(d+img(0))   disp(d) 256x704
        depth_disp = []
        x = inputs[-1]
        for i in range(self.img_level-1, -1, -1):
            x = self.lateral_conv[i](x)
            
            x = torch.cat([x, inputs[i]], dim = 1)
            x = self.conv[i](x)
            if i in self.out_scale:
                disp = self.disp[i](x)
                disp = disp.squeeze(1)
                depth_disp.append(disp)
            x = F.interpolate(x, scale_factor=self.scale_facter[i], mode='nearest')
        

        return depth_disp

    def to_depth(self, disp_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        # disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        # depth = einops.rearrange(depth, '(b n) c w h -> b n c w h',b=K_in.size(0))
        # depth = depth * K_in[:, :, 0:1, 0:1, None]
        # depth = depth * torch.abs(post_rots[:, :, 0:1, 0:1, None])
        # depth = depth / self.focal_length_scale
        return depth
    
    def get_multi_level_depth_loss(self, multi_level_depth, depth_labels):
        losses = dict()
        for lvl, depth_pred in enumerate(multi_level_depth):
            # depth_pred = torch.softmax(depth_pred, dim=1)
            # loss = self.get_depth_loss(depth_pred, depth_labels)
            # losses['back_proj_loss_{}'.format(lvl)] = self.back_proj_loss_weight * loss
            loss = self.get_l2_loss(depth_pred, depth_labels)
            losses['back_proj_loss_{}'.format(lvl)] = self.back_proj_loss_weight * loss
        return losses
    
    def get_l2_loss(self, depth_preds, depth_labels):
        downsample = depth_labels.shape[-1]//depth_preds.shape[-1]
        depth_labels = self.get_downsampled_gt_depth(depth_labels, downsample)
        fg_mask = depth_labels > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss = F.l1_loss(depth_preds, depth_labels, reduction='none').sum()/ max(1.0, fg_mask.sum())
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_preds, depth_labels):
        downsample = depth_labels.shape[-1]//depth_preds.shape[-1]
        depth_labels = self.get_downsampled_gt_depth(depth_labels, downsample)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss
    
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

        # gt_depths = (gt_depths - (self.depth_cfg[0] -
        #                             self.depth_cfg[2])) / \
        #             self.depth_cfg[2]

        gt_depths = torch.where((gt_depths <= self.max_depth) & (gt_depths >= 0),
                                gt_depths, torch.zeros_like(gt_depths))
        # gt_depths = F.one_hot(
        #     gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
        #                                                                    1:]
        return gt_depths.float()


# class XXX(nn.Module):
#     """
#     This class decodes encoded 2D features to estimate depth map.
#     Unlike monodepth depth decoder, we decode features with corresponding level we used to project features in 3D (default: level 2(H/4, W/4))
#     """    
#     def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
#         super(DepthDecoder, self).__init__()

#         self.num_output_channels = 1
#         self.scales = scales
#         self.use_skips = use_skips
        
#         self.level_in = level_in
#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = num_ch_dec

#         self.convs = OrderedDict()
#         for i in range(self.level_in, -1, -1):
#             num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i > 0:
#                 num_ch_in += self.num_ch_enc[i - 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

#         for s in self.scales:
#             self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_features):
#         outputs = {}
        
#         # decode
#         x = input_features[-1]
#         for i in range(self.level_in, -1, -1):
#             x = self.convs[('upconv', i, 0)](x)
#             x = [upsample(x)]
#             if self.use_skips and i > 0:
#                 x += [input_features[i - 1]]
#             x = torch.cat(x, 1)
#             x = self.convs[('upconv', i, 1)](x)
#             if i in self.scales:
#                 outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))                
#         return outputs