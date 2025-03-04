import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import concurrent.futures

# scene = '/data4/wwb/BEVDet/data/nuscenes/gts'
# label_path = [str(p) for p in Path(scene).rglob('*.npz')]
# counter = Counter()

# def process_file(npz):
#     label = np.load(npz)
#     gt = label['semantics']
#     return Counter(gt.flatten())

# random.shuffle(label_path)
# label_path = label_path[:1000]
# for c in tqdm(label_path):
#     counter += process_file(c)
# print(counter)

# label = {17: 608429852, 16: 9135473, 15: 8113984, 11: 7153203, 14: 2445083, 13: 2049854, 4: 1177676, 10: 505978, 9: 205358, 3: 189793, 12: 178448, 7: 113829, 1: 101017, 0: 81757, 5: 79920, 8: 20117, 6: 10505, 2: 8153}
# sorted_label = dict(sorted(label.items()))
our={'others': 0.14292518230792842, 'barrier': 0.512652765026361, 'bicycle': 0.3111149103498386, 'bus': 0.4612686765074414, 'car': 0.5509219786129582, 'cons. veh.': 0.291189709899687, 'motorcycle': 0.30456817259096447, 'pedestrian': 0.3098976109215017, 'traffic cone': 0.3547196925660402, 'trailer': 0.35204625039269927, 'truck': 0.41817020930426724, 'drive. surf.': 0.8400124495087905, 'other flat': 0.47001699397922453, 'sidewalk': 0.5551845929616837, 'terrain': 0.595048762349998, 'manmade': 0.500262930302283, 'vegetation': 0.4481935531858784}
print('\n'.join(map(str,our.keys())))
# print('\n'.join(map(str,sorted_label.keys())))
