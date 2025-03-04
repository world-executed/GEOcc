_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

sync_bn = True
# plugin = True
# plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']

# 20 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
num_class = len(class_names)

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
# lss_downsample = [2, 2, 2]
lss_downsample = [1, 1, 1]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = {
    'input_size': (384, 1280),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'y': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'z': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'depth': [2.0, 58.0, 0.5],
}

# settings for 3D encoder
numC_Trans = 32
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

# settings for mask2former head
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
mask2former_num_heads = voxel_out_channels // 32

multi_adj_frame_id_cfg = (1, 1+1, 1)


model = dict(
    type='BEVStereo4DOCC',
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
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          use_bn=False,# kitti only one cam
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        ignore_index=255),
    use_mask=False,
    num_classes=num_class,
)

# model = dict(
#     type='OccupancyFormer',
#     img_backbone=dict(
#         type='CustomEfficientNet',
#         arch='b7',
#         drop_path_rate=0.2,
#         frozen_stages=0,
#         norm_eval=False,
#         out_indices=(2, 3, 4, 5, 6),
#         with_cp=True,
#         init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
#     ),
#     img_neck=dict(
#         type='SECONDFPN',
#         in_channels=[48, 80, 224, 640, 2560],
#         upsample_strides=[0.25, 0.5, 1, 2, 2],
#         out_channels=[128, 128, 128, 128, 128]),
#     img_view_transformer=dict(
#         type='ViewTransformerLiftSplatShootVoxel',
#         numC_input=640,
#         cam_channels=33,
#         loss_depth_weight=1.0,
#         grid_config=grid_config,
#         data_config=data_config,
#         numC_Trans=numC_Trans,
#         vp_megvii=False),
#     img_bev_encoder_backbone=dict(
#         type='OccupancyEncoder',
#         num_stage=len(voxel_num_layer),
#         in_channels=numC_Trans,
#         block_numbers=voxel_num_layer,
#         block_inplanes=voxel_channels,
#         block_strides=voxel_strides,
#         out_indices=voxel_out_indices,
#         with_cp=True,
#         norm_cfg=norm_cfg,
#     ),
#     img_bev_encoder_neck=dict(
#         type='MSDeformAttnPixelDecoder3D',
#         strides=[2, 4, 8, 16],
#         in_channels=voxel_channels,
#         feat_channels=voxel_out_channels,
#         out_channels=voxel_out_channels,
#         norm_cfg=norm_cfg,
#         encoder=dict(
#             type='DetrTransformerEncoder',
#             num_layers=6,
#             transformerlayers=dict(
#                 type='BaseTransformerLayer',
#                 attn_cfgs=dict(
#                     type='MultiScaleDeformableAttention3D',
#                     embed_dims=voxel_out_channels,
#                     num_heads=8,
#                     num_levels=3,
#                     num_points=4,
#                     im2col_step=64,
#                     dropout=0.0,
#                     batch_first=False,
#                     norm_cfg=None,
#                     init_cfg=None),
#                 ffn_cfgs=dict(
#                     embed_dims=voxel_out_channels),
#                 feedforward_channels=voxel_out_channels * 4,
#                 ffn_dropout=0.0,
#                 operation_order=('self_attn', 'norm', 'ffn', 'norm')),
#             init_cfg=None),
#         positional_encoding=dict(
#             type='SinePositionalEncoding3D',
#             num_feats=voxel_out_channels // 3,
#             normalize=True),
#     ),
#     pts_bbox_head=dict(
#         type='Mask2FormerOccHead',
#         feat_channels=mask2former_feat_channel,
#         out_channels=mask2former_output_channel,
#         num_queries=mask2former_num_queries,
#         num_occupancy_classes=num_class,
#         pooling_attn_mask=True,
#         sample_weight_gamma=0.25,
#         # using stand-alone pixel decoder
#         positional_encoding=dict(
#             type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
#         # using the original transformer decoder
#         transformer_decoder=dict(
#             type='DetrTransformerDecoder',
#             return_intermediate=True,
#             num_layers=9,
#             transformerlayers=dict(
#                 type='DetrTransformerDecoderLayer',
#                 attn_cfgs=dict(
#                     type='MultiheadAttention',
#                     embed_dims=mask2former_feat_channel,
#                     num_heads=mask2former_num_heads,
#                     attn_drop=0.0,
#                     proj_drop=0.0,
#                     dropout_layer=None,
#                     batch_first=False),
#                 ffn_cfgs=dict(
#                     embed_dims=mask2former_feat_channel,
#                     num_fcs=2,
#                     act_cfg=dict(type='ReLU', inplace=True),
#                     ffn_drop=0.0,
#                     dropout_layer=None,
#                     add_identity=True),
#                 feedforward_channels=mask2former_feat_channel * 8,
#                 operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
#                                  'ffn', 'norm')),
#             init_cfg=None),
#         # loss settings
#         loss_cls=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=2.0,
#             reduction='mean',
#             class_weight=[1.0] * num_class + [0.1]),
#         loss_mask=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=5.0),
#         loss_dice=dict(
#             type='DiceLoss',
#             use_sigmoid=True,
#             activate=True,
#             reduction='mean',
#             naive_dice=True,
#             eps=1.0,
#             loss_weight=5.0),
#         point_cloud_range=point_cloud_range,
#     ),
#     train_cfg=dict(
#         pts=dict(
#             num_points=12544 * 4,
#             oversample_ratio=3.0,
#             importance_sample_ratio=0.75,
#             assigner=dict(
#                 type='MaskHungarianAssigner',
#                 cls_cost=dict(type='ClassificationCost', weight=2.0),
#                 mask_cost=dict(
#                     type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
#                 dice_cost=dict(
#                     type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
#             sampler=dict(type='MaskPseudoSampler'),
#         )),
#     test_cfg=dict(
#         pts=dict(
#             semantic_on=True,
#             panoptic_on=False,
#             instance_on=False)),
# )

dataset_type = 'CustomSemanticKITTILssDataset'
data_root = 'data/SemanticKITTI'
ann_file = 'data/SemanticKITTI/labels'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, 
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'gt_depth','voxel_semantics'], 
            meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False, 
         data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        split='train',
        camera_used=camera_used,
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=100)


# checkpoint_config = dict(max_keep_ckpts=1, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=30)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
evaluation = dict(interval=1, pipeline=test_pipeline)
load_from="bevdet-r50-4d-stereo-cbgs.pth"