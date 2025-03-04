import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler




def get_rays(i, j, K, c2w, inverse_y=True):
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)   
    return rays_o, rays_d, viewdirs



def pts2ray(coor, label_depth, label_seg, c2w, cam_intrinsic):
    rays_o, rays_d, viewdirs = get_rays(coor[:,0]+0.5, coor[:,1]+0.5, K=cam_intrinsic,c2w=c2w)
    return torch.cat([
        coor, label_depth.unsqueeze(1), label_seg.unsqueeze(1),  # 0-1, 2, 3
        rays_o, rays_d, viewdirs        # 4:7,7:10,10:13
        ], dim=1
    )
    

def generate_rays(coors, label_depths, label_segs, c2w, intrins, max_ray_nums=0, time_ids=None, dynamic_class=None, balance_weight=None, weight_adj=0.3, weight_dyn=0.0, use_wrs=True):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Args:
        coors (Nx2): coordinates of valid pixels in xy
        label_depths (Nx1): depth GT of pixels
        label_segs (Nx1): semantic GT of pixels
        c2w : camera to ego
        intrins : camera intrins matrix
        width : width of RGB image
        height : height of RGB image
        max_ray_nums : 
        time_ids (Tx1) : 
        balance_weight : 
        weight_adj : 
        weight_dyn : 
        use_wrs : 
    """
    # generate rays
    rays = []
    ids = []
    for time_id in time_ids:    # multi frames
        for i in time_ids[time_id]: # multi cameras of single frame
            ray = pts2ray(coors[i], label_depths[i], label_segs[i], c2w[i], intrins[i])
            if time_id==0:
                ray = torch.cat([ray, torch.ones(ray.shape[0],1)*i], dim=1)
            else:
                ray = torch.cat([ray, torch.ones(ray.shape[0], 1)*-1], dim=1)
            rays.append(ray)
            ids.append(time_id)

    # Weighted Rays Sampling
    if not use_wrs:
        rays = torch.cat(rays, dim=0)
    else:
        weights = []
        if balance_weight is None:  # use batch data to compute balance_weight ( rather than the total dataset )
            classes = torch.cat([ray[:,3] for ray in rays])
            class_nums = torch.Tensor([0]*17)
            for class_id in range(17): 
                class_nums[class_id] += (classes==class_id).sum().item()
            balance_weight = torch.exp(0.005 * (class_nums.max() / class_nums - 1))

        key_frame_rays = rays[:6]
        key_rays_nums = sum([ray.shape[0] for ray in key_frame_rays])
        max_ray_nums = max(max_ray_nums - key_rays_nums,0)
        rays = rays[6:]
        ids = ids[6:]
        
        for i in range(len(rays)):
            # wrs-a
            ans = 1.0 if ids[i]==0 else weight_adj
            weight_t = torch.full((rays[i].shape[0],), ans)
            if ids[i]!=0:
                mask_dynamic = (dynamic_class == rays[i][:, 3, None]).any(dim=-1)
                weight_t[mask_dynamic] = weight_dyn
            # wrs-b
            weight_b = balance_weight[rays[i][..., 3].long()]

            weight = weight_b * weight_t
            weights.append(weight)

        rays = torch.cat(rays, dim=0)
        weights = torch.cat(weights, dim=0)
        if max_ray_nums!=0 and rays.shape[0]>max_ray_nums:
            sampler = WeightedRandomSampler(weights, num_samples=max_ray_nums, replacement=False)
            rays = rays[list(sampler)]

        key_frame_rays = torch.cat(key_frame_rays,0)
        rays = torch.cat([key_frame_rays, rays], dim=0)
    return rays

def generate_dense_rays(c2w, intrins, max_ray_nums=0, time_ids=None, dynamic_class=None, balance_weight=None, weight_adj=0.3, weight_dyn=0.0, use_wrs=True,render_downsample=10):
    # generate rays
    cam_num = 6
    H=900//render_downsample
    W=1600//render_downsample
    intrins[:,:2,:] /=render_downsample
    with torch.no_grad():
        coord = torch.zeros(cam_num, H, W, 2)
        rays_o_all = torch.zeros(cam_num, H, W, 3)
        rays_d_all = torch.zeros(cam_num, H, W, 3)
        viewdirs = torch.zeros(cam_num, H, W, 3)

        for i in range(cam_num):
            coor, rays_o, rays_d, viewd= get_dense_rays(H, W, intrins[i,...], c2w[i,...])
            coord[i,...] = coor
            rays_o_all[i,...] = rays_o
            rays_d_all[i,...] = rays_d
            viewdirs[i,...] = viewd
        depths = torch.zeros(cam_num, H, W, 1)
        segs = torch.zeros(cam_num, H, W, 1)
        rays = torch.cat([coord,depths,segs,rays_o_all,rays_d_all,viewdirs],dim=3)
    return rays
    # dense ray
            
def get_dense_rays(H, W, K, c2w):
    i,j = torch.meshgrid(
        torch.linspace(0, W-1,W),
        torch.linspace(0, H-1,H)
    )
    i = i.t().float()
    j = j.t().float()
    
    i,j = i+0.5, j+0.5
    rays_o, rays_d, viewdirs = get_rays(i, j, K, c2w)
    return  torch.stack((i,j),dim=2), rays_o, rays_d, viewdirs
