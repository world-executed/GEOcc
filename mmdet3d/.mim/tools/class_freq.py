import os
from tqdm import tqdm
import numpy as np
gt_path = '/data4/wwb/BEVDet/data/nuscenes/gts'
cls_freq = [0]*18
for scene in tqdm(os.listdir(gt_path)):
    scene_path = os.path.join(gt_path,scene)
    for token in os.listdir(scene_path):
        info_path = os.path.join(scene_path,token,'labels.npz')
        info = np.load(info_path)
        gt = info['semantics']
        freq = np.bincount(gt.flatten())
        for i in range(18):
            cls_freq[i] += freq[i]
print(cls_freq)
np.save('cls,npy',np.array(cls_freq))