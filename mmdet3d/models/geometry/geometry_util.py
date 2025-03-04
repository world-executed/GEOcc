import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule
import torch.nn.functional as F

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
        
def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    """
    # initialize matrices
    b, _, _ = rot_angle.shape
    R_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)
    T_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1)
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1)

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec

    if invert == True:
        P_mat = torch.matmul(R_mat, T_mat)
    else :
        P_mat = torch.matmul(T_mat, R_mat)
    return P_mat

def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

class Projection(nn.Module):
    """
    This class computes projection and reprojection function. 
    """
    def __init__(self, height, width, device='cuda'):
        super().__init__()
        self.width = width
        self.height = height
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(1, 1, 1) # batch
        img_points = img_points.to(device)
        
        self.to_homo = torch.ones([1, 1, width*height]).to(device) # batch
        self.homo_points = torch.cat([img_points, self.to_homo], 1) # batch

    def backproject(self, invK, depth, post_rots, post_trans):
        """
        This function back-projects 2D image points to 3D.
        """
        bs = invK.shape[0]
        depth = depth.reshape(bs, 1, -1)
        homo_points = self.homo_points.repeat(bs, 1, 1)
        # pixel_grid = pixel_grid - post_trans[:,cam,:].reshape(batch_size, 3, 1)
        # pixel_grid = torch.matmul(torch.inverse(post_rots[:,cam,:,:]), pixel_grid)
        homo_points = homo_points - post_trans.unsqueeze(2)
        homo_points = torch.matmul(torch.inverse(post_rots),homo_points)

        points3D = torch.matmul(invK[:, :3, :3], homo_points)
        points3D = depth*points3D
        to_homo = self.to_homo.repeat(bs, 1, 1)
        return torch.cat([points3D, to_homo], 1)
    
    def reproject(self, K, points3D, T, post_rots, post_trans):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        bs = K.shape[0]
        # project points 
        viewpad = torch.zeros_like(T)
        viewpad[:,:3,:3] = K
        viewpad[:,3,3] = 1
        points2D = (viewpad @ T)[:,:3, :] @ points3D

        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(bs, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)
        # pix_coords = torch.matmul(pix_coords, post_rots.reshape(self.batch_size, 1, 2, 2))
        # pix_coords = pix_coords + post_trans.reshape(self.batch_size, 1, 1, 2)
        post_rots = post_rots[:,:2,:2]
        post_trans = post_trans[:,:2]
        # norm_points2D = torch.matmul(norm_points2D, post_rots.reshape(bs, 1, 2,2))
        # front mul
        norm_points2D = torch.matmul(post_rots.reshape(bs,1,1,2,2,),norm_points2D.unsqueeze(-1)).squeeze()
        norm_points2D = norm_points2D + post_trans.reshape(bs, 1, 1, 2)
        norm_points2D[..., 0 ] /= self.width - 1
        norm_points2D[..., 1 ] /= self.height - 1
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D        

    def forward(self, depth, T, bp_invK, rp_K, tar_rots, tar_trans, src_rots, src_trans):
        cam_points = self.backproject(bp_invK, depth, tar_rots, tar_trans)
        pix_coords = self.reproject(rp_K, cam_points, T, src_rots, src_trans)
        return pix_coords
    

from mmdet.models import BACKBONES
@BACKBONES.register_module()
class VolumeProjector(nn.Module):
    def __init__(self,
                 feat_out_dim=88,
                 voxel_unit_size=[0.4, 0.4, 0.4],
                 voxel_size=[100, 100, 8],
                #  voxel_downsample=[1, 2, 4],
                 voxel_str_p=[-40.0, -40.0, -1],
                 voxel_end_p=[40, 40, 5.4],
                #  voxel_pre_dim=[64],
                 bev_feat_dim=192,
                 proj_d_bins=50,
                 proj_d_str=1,
                 proj_d_end=45,
                 height=256, 
                 width=704):
        super(VolumeProjector, self).__init__()
        self.num_cams = 6
        self.voxel_unit_size=voxel_unit_size
        self.voxel_size=voxel_size
        # self.voxel_downsample = voxel_downsample
        self.voxel_str_p=voxel_str_p
        self.voxel_end_p = voxel_end_p
        # self.voxel_pre_dim=voxel_pre_dim
        self.proj_d_bins=proj_d_bins
        self.proj_d_str=proj_d_str
        self.proj_d_end=proj_d_end
        self.height=height
        self.width=width
        self.img_h = self.height // (2 ** (4))
        self.img_w = self.width // (2 ** (4))
        self.num_pix = self.img_h * self.img_w
        depth_bins = torch.linspace(self.proj_d_str, self.proj_d_end, self.proj_d_bins)
        self.pixel_grid = self.create_pixel_grid(self.img_h, self.img_w).cuda()
        self.pixel_ones = torch.ones(1, 1, self.proj_d_bins, self.num_pix).cuda()
        self.depth_grid = self.create_depth_grid(self.num_pix, self.proj_d_bins, depth_bins).cuda()
        encoder_dims = bev_feat_dim*proj_d_bins
        self.reduce_dim = nn.Sequential(
            ConvModule(
                encoder_dims,
                256,
                kernel_size=3,
                stride=1,
                padding=1),
            ConvModule(
                256,
                feat_out_dim,
                kernel_size=3,
                stride=1,
                padding=1))  


    def create_pixel_grid(self, height, width):
        """
        output: [batch, 3, height * width]
        """
        x = torch.linspace(0, self.width-1, width)\
            .view(1, width, 1).expand(1, width, height)
        y = torch.linspace(0, self.height-1, height)\
            .view(1, 1, height).expand(1, width, height)
        # grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height))
        # pix_coords = torch.stack(grid_xy, axis=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = torch.cat([x,y], dim=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(1, 1, 1)
        ones = torch.ones(1, 1, height * width)
        pix_coords = torch.cat([pix_coords, ones], 1)
        return pix_coords

    def create_depth_grid(self, n_pixels, n_depth_bins, depth_bins):
        """
        output: [batch, 3, num_depths, height * width]
        """
        depth_layers = []
        for d in depth_bins:
            depth_layer = torch.ones((1, n_pixels)) * d
            depth_layers.append(depth_layer)
        depth_layers = torch.cat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(1, 3, n_depth_bins, n_pixels)
        return depth_layers
    
    def forward(self, bev_feat, img_inputs):
        """
        This function projects voxels into 2D image coordinate. 
        [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        """        
        (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda) = img_inputs[:7]
        ints = intrins[:,:6,...]
        inv_K = torch.inverse(ints)
        extrinsics = sensor2egos[:,:6,...]
        bev_feat = bev_feat[0]
        bev_feat = bev_feat.permute(0,1,4,3,2) # xyz->zyx
        batch_size = imgs.shape[0]
        
        proj_feats = []
        for cam in range(self.num_cams):
            # construct 3D point grid for each view
            # vf depth version########################
            # pixel_grid = self.pixel_grid.clone()
            # pixel_grid = pixel_grid.repeat(batch_size, 1, 1)
            # pixel_grid = pixel_grid - post_trans[:,cam,:].reshape(batch_size, 3, 1)
            # pixel_grid = torch.matmul(torch.inverse(post_rots[:,cam,:,:]), pixel_grid)
            # cam_points = torch.matmul(inv_K[:, cam, :3, :3], pixel_grid)
            # cam_points = self.depth_grid.expand(batch_size,-1,-1,-1,) * cam_points.view(batch_size, 3, 1, self.num_pix)
            # cam_points = torch.cat([cam_points, self.pixel_ones.expand(batch_size,-1,-1,-1,)], dim=1) # [b, 4, n_depthbins, n_pixels]
            # cam_points = cam_points.view(batch_size, 4, -1) # [b, 4, n_depthbins * n_pixels]
            # vf depth version#########################

            # bev version########################
            frustum = None
            d = torch.linspace(self.proj_d_str,self.proj_d_end,self.proj_d_bins)
            pixel_grid = self.pixel_grid.reshape(1, 3, 1, self.img_h, self.img_w)
            pixel_grid = pixel_grid[:,:2,...].repeat(batch_size, 1, self.proj_d_bins, 1, 1)
            d = self.depth_grid[:,0,...].reshape(1, 1 ,self.proj_d_bins, self.img_h, self.img_w).repeat(batch_size, 1, 1, 1, 1)
            frustum = torch.cat([pixel_grid, d],dim=1)
            frustum = frustum.permute(0,2,3,4,1) # bs, D, H, W, 3

            points = frustum - post_trans[:,cam,:].view(batch_size, 1, 1, 1, 3)
            points = torch.inverse(post_rots[:,cam,:,:]).view(batch_size,1,1,1,3,3).matmul(points.unsqueeze(-1))
            points = torch.cat((points[...,:2,:]*points[...,2:3,:],points[...,2:3, :]), 4)
            points = inv_K[:,cam,:3,:3].view(batch_size, 1, 1, 1, 3, 3).matmul(points)
            cam_points = points.reshape(batch_size, 3, self.img_h*self.img_w*self.proj_d_bins)
            ones = self.pixel_ones.repeat(batch_size, 1, 1, 1)
            cam_points = torch.cat([cam_points,ones.reshape(batch_size, 1, -1)], dim = 1)
            # bev version########################
            
            # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
            points = torch.matmul(extrinsics[:, cam, :3, :], cam_points)

            #bda aug
            points = torch.matmul(bda, points)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = points.permute(0, 2, 1) 
            
            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.
                
            grid = grid.view(batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)            
            proj_feat = F.grid_sample(bev_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            proj_feat = proj_feat.view(batch_size, -1, self.img_h, self.img_w)
            
            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        
        return torch.stack(proj_feats, dim=1).flatten(start_dim=0, end_dim=1)