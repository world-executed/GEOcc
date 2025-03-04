import matplotlib.pyplot as plt
import numpy as np
import os

log_file = '/data4/wwb/BEVDet/work_dirs/bevdet-occ-mfh-100x100x8/20231120_164552.log.json'

with open(log_file, 'r') as f:
    lines = f.readlines()

lines = lines[1:]
lines = [eval(line) for line in lines]

loss_cls = []
loss_depth = []
loss_mask = []
for line in lines:
    if line['mode'] == 'train':
        loss_cls.append(line['loss_cls'])
        loss_depth.append(line['loss_depth'])
        loss_mask.append(line['loss_mask'])

loss_cls = np.array(loss_cls)
loss_depth = np.array(loss_depth)
loss_mask = np.array(loss_mask)

loss_cls = np.convolve(loss_cls, np.ones(100) / 100, mode='valid')
loss_depth = np.convolve(loss_depth, np.ones(100) / 100, mode='valid')
loss_mask = np.convolve(loss_mask, np.ones(100) / 100, mode='valid')

plt.figure()
plt.plot(loss_cls, label='loss_cls')
plt.plot(loss_depth, label='loss_depth')
plt.plot(loss_mask, label='loss_mask')
plt.xlabel('Training iteration (every 50 iterations as a point)')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join('/data4/wwb/BEVDet/out', 'loss.png'))
