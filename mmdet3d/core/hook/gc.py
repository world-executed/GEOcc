# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
import gc

__all__ = ['GCHook']


@HOOKS.register_module()
class GCHook(Hook):
    """ """

    def __init__(self):
        super().__init__()
    
    def after_epoch(self, runner):
        gc.collect()