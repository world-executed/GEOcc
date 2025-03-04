# Copyright (c) 2023 42dot. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet

from .resnet_encoder import ResnetEncoder

from mmdet.models import BACKBONES

@BACKBONES.register_module()
class FusedDepthNet(nn.Module):
    """
    Depth fusion module
    """    
    def __init__(self, 
                 num_layers=18,
                 weights_init=True,
                 fusion_feat_in_dim=256,
                 fusion_feat_out_dim=88,
                 fusion_level=2,
                 use_skips=False,
                 scales=[0],
                 num_cams=6,
                 batch_size=1,
                 aug_depth=False):
        super(FusedDepthNet, self).__init__()
        self.num_layers=num_layers
        self.weights_init=weights_init
        self.fusion_feat_in_dim=fusion_feat_in_dim
        self.fusion_level=fusion_level
        self.use_skips=use_skips
        self.scales=scales
        self.num_cams=num_cams
        self.batch_size=batch_size
        self.aug_depth=aug_depth
        self.syn_visualize=False
        
        # feature encoder        
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)        
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 1) # number of layers, pretrained, number of input images
        del self.encoder.encoder.fc
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode = 'reflect')
        
        # fusion net
        # fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level]
        self.fusion_net = VFNet(self.fusion_feat_in_dim, fusion_feat_out_dim, model ='depth', batch_size=batch_size, fusion_level=self.fusion_level)
        
        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[:(self.fusion_level+1)]
        num_ch_dec = [16, 32, 64, 128, 256]
        # self.decoder = DepthDecoder(self.fusion_level, num_ch_enc, num_ch_dec, self.scales, use_skips = self.use_skips)
        # self.decoder = CustomDepthDecoder(target_size=(16,44),out_dim=88)
    

    def forward(self, inputs):
        outputs = {}
        
        # dictionary initialize
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}
        
        lev = self.fusion_level
        
        # packed images for surrounding view
        sf_images = inputs['img_inputs']
        # sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        packed_input = pack_cam_feat(sf_images)
        
        # feature encoder
        packed_feats = self.encoder(packed_input)            
        # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
        _, _, up_h, up_w = packed_feats[lev].size()
        
        packed_feats_list = packed_feats[lev:lev+1] \
                        + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]        
        
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))        
        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams)
        
        # fusion_net, backproject each feature into the 3D voxel space
        fusion_dict = self.fusion_net(inputs, feats_agg)        

        # feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]    
        # packed_depth_outputs = self.decoder(feat_in)            
        packed_depth_outputs = fusion_dict['proj_feat']
            
        depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams)
        return depth_outputs
        for cam in range(self.num_cams):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
        
        if self.aug_depth:
            feat_in = packed_feats[:lev] + [fusion_dict['proj_feat_aug']]              
            packed_depth_outputs = self.decoder(feat_in)
            depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams)
            for cam in range(self.num_cams):
                for k in depth_outputs.keys():
                    k_aug = k + ('aug',)
                    outputs[('cam', cam)][k_aug] = depth_outputs[k][:, cam, ...]
                    
        if self.syn_visualize:
            proj_feats = fusion_dict['syn_feat']
            outputs['disp_vis'] = []
            for feat in proj_feats:
                depth_outputs = self.decoder([feat])
                outputs['disp_vis'] += depth_outputs[('disp', 0)]
        return outputs
    
        
class DepthDecoder(nn.Module):
    """
    This class decodes encoded 2D features to estimate depth map.
    Unlike monodepth depth decoder, we decode features with corresponding level we used to project features in 3D (default: level 2(H/4, W/4))
    """    
    def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.scales = scales
        self.use_skips = use_skips
        
        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        
        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))                
        return outputs
    
class CustomDepthDecoder(nn.Module):
    def __init__(self, target_size=(16,44), out_dim=88):
        super(CustomDepthDecoder, self).__init__()
        self.target_size = target_size
        
        self.conv = conv2d(256, out_dim, kernel_size=3, nonlin = 'ELU')
    def forward(self, input_features):
        proj = input_features[-1]
        # for f in input_features:
        #     x.append(F.interpolate(f,self.target_size,mode='bilinear'))
        # x = torch.cat(x,dim=1)
        x = self.conv(proj)
        return x
