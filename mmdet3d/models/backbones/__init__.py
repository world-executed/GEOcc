# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .dgcnn import DGCNNBackbone
from .dla import DLANet
from .mink_resnet import MinkResNet
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG
from .resnet import CustomResNet, CustomResNet3D
from .second import SECOND
from .swin import SwinTransformer
from .occnet import OccupancyEncoder
from .sam_encoder import ImageEncoderViT
from .tiny_vit_sam import TinyViT
__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'DGCNNBackbone', 'PointNet2SASSG', 'PointNet2SAMSG',
    'MultiBackbone', 'DLANet', 'MinkResNet', 'CustomResNet', 'CustomResNet3D',
    'SwinTransformer', 'OccupancyEncoder', 'ImageEncoderViT','TinyViT'
]
