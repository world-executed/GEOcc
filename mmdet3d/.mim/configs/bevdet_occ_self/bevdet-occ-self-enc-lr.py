# Copyright (c) Phigent Robotics. All rights reserved.

# align_after_view_transfromation=False
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 8.22
# ===> barrier - IoU = 44.21
# ===> bicycle - IoU = 10.34
# ===> bus - IoU = 42.08
# ===> car - IoU = 49.63
# ===> construction_vehicle - IoU = 23.37
# ===> motorcycle - IoU = 17.41
# ===> pedestrian - IoU = 21.49
# ===> traffic_cone - IoU = 19.7
# ===> trailer - IoU = 31.33
# ===> truck - IoU = 37.09
# ===> driveable_surface - IoU = 80.13
# ===> other_flat - IoU = 37.37
# ===> sidewalk - IoU = 50.41
# ===> terrain - IoU = 54.29
# ===> manmade - IoU = 45.56
# ===> vegetation - IoU = 39.59
# ===> mIoU of 6019 samples: 36.01


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_config = {
    # 'cams': [
    #     'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
    #     'CAM_BACK', 'CAM_BACK_RIGHT'
    # ],
    'cams': [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT', 'CAM_BACK'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    'occ_size': (200, 200, 16),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
_dim_ = [128]
_ffn_dim_ = [256]
volume_h_ = [100]
volume_w_ = [100]
volume_z_ = [8]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1+1, 1)
render_downsample = 10
num_class=18
voxel_out_channels = 192
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
mask2former_num_heads = voxel_out_channels // 32
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='BEVStereo4DOCCSELFENC',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.005,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='OccupancyEncoder',
        num_stage=3,
        in_channels=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1) + 64,
        block_numbers=[2, 2, 2],
        block_inplanes=[128, 256, 512],
        block_strides=[1, 2, 2],
        out_indices=(0, 1, 2),
        with_cp=True,
        norm_cfg=norm_cfg,
    ),
    # img_bev_encoder_neck=dict(type='LSSFPN3D',
    #                           in_channels=numC_Trans*7,
    #                           out_channels=numC_Trans),
    img_bev_encoder_neck=dict(
        type='MSDeformAttnPixelDecoder3D',
        strides=[2, 4, 8],
        in_channels=[128, 256, 512],
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        norm_cfg=norm_cfg,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=1,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=voxel_out_channels,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    embed_dims=voxel_out_channels),
                feedforward_channels=voxel_out_channels * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=voxel_out_channels // 3,
            normalize=True),
    ),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=True,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
    occ_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=18,
        # conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        # conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[256, 256, 256],
        use_semantic=True,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
    ),
    m2f_head=dict(
        type='Mask2FormerOccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        num_queries=mask2former_num_queries,
        num_occupancy_classes=num_class,
        num_transformer_feat_level=2,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        # using stand-alone pixel decoder
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
        # using the original transformer decoder
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=1,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        # loss settings
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            # naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        point_cloud_range=point_cloud_range,
        train_cfg=dict(
            pts=dict(
                num_points=12544 * 4,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='ClassificationCost', weight=2.0),
                    mask_cost=dict(
                        type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dice_cost=dict(
                        type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
                sampler=dict(type='MaskPseudoSampler'),
            )),
        test_cfg=dict(
            pts=dict(
                semantic_on=True,
                panoptic_on=False,
                instance_on=False)),
    ),
    nerf_head = dict(
        type='NerfHeadDense',
        point_cloud_range = [-40, -40, -1, 40, 40, 5.4],
        voxel_size=0.4,
        scene_center=[0,0,2.2],
        radius=39,
        use_sup=True,
        weight_depth=0.1
    ),
    use_depth_sup=True,
    use_sem_sup=True,
    use_img_recons_proxy=True,
    img_recons_proxy=dict(type='ReconstructionProxyNerf',render_downsample=render_downsample)
)

# Data
dataset_type = 'NuScenesDatasetOccpancyNerfDense'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True,
        render_downsample=render_downsample),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar','mask_camera','rays','img_ori','label_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='MaskTransforms'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs','rays'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,

)

test_data_config = dict(
    use_rays=False,
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        use_rays=True,
        depth_gt_path='./data/depth_gt',
        semantic_gt_path='./data/nuscenes/seg_gt_lidarseg',
        aux_frames = [-3,-2,-1,1,2,3],
        max_ray_nums=38400,
        render_downsample=render_downsample),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-2)
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer = dict(
#     type='AdamW',
#     lr=1e-4,
#     weight_decay=1e-2,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=dict(
#         custom_keys={
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#         },
#         norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=100)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
# evaluation = dict(interval=1, pipeline=test_pipeline)
load_from="bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
# find_unused_parameters=True
# trace_config = dict(type='tb_trace', dir_name='work_dir')
# profiler_config = dict(on_trace_ready=trace_config)