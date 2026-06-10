import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger as loguru

from utils.data_util import to_numpy, to_tensor, homo

def normalize_depth(depth : torch.Tensor, mask : torch.Tensor):
    if mask is not None:
        depth_min = torch.min(depth[mask.bool()])
        depth_max = torch.max(depth[mask.bool()])
        depth = depth * mask.bool()
    else:
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)
    return (depth - depth_min) / (depth_max - depth_min + 1e-8)


def colorize_depth(depth, valid_mask=None, cmap='turbo'):
    # Normalize depth values to 0-1 range for proper visualization
    if valid_mask is None:
        valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0)
    if not np.any(valid_mask):
        return np.zeros((*depth.shape, 3))
    
    d_min, d_max = np.min(depth[valid_mask]), np.max(depth[valid_mask])
    normalized_depth = np.zeros_like(depth)
    if d_max > d_min:
        normalized_depth[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min + 1e-8)
        normalized_depth[valid_mask] = normalized_depth[valid_mask] / 2 + 0.5
    
    cmap_fn = plt.get_cmap(cmap)
    colored_depth = cmap_fn(normalized_depth)
    if colored_depth.shape[-1] == 4:
        colored_depth = colored_depth[..., :3]
    return colored_depth

def dilate_mask(mask : torch.Tensor, kernel_size=3):
    kernel = torch.ones(kernel_size, kernel_size).to(mask.device)
    mask = nn.functional.conv2d(
        mask.unsqueeze(0).unsqueeze(0), 
        kernel.unsqueeze(0).unsqueeze(0), 
        padding=kernel_size//2
    )
    mask = (mask > 0).long()
    return mask.squeeze(0).squeeze(0)

def erode_mask(mask : torch.Tensor, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device, dtype=torch.float32)
    # convert mask to float for convolution
    m = mask.float().unsqueeze(0).unsqueeze(0)
    # compute asymmetric padding so output spatial dims equal input spatial dims
    pad_total = kernel_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    m = nn.functional.pad(m, (pad_left, pad_right, pad_top, pad_bottom))
    m = nn.functional.conv2d(m, kernel)
    # require full overlap for erosion
    m = (m >= (kernel_size * kernel_size)).long()
    return m.squeeze(0).squeeze(0)

def erode_mask_batch(mask : torch.Tensor, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device, dtype=torch.float32)
    # convert mask to float for convolution
    m = mask.float().unsqueeze(1)  # B, 1, H, W
    # compute asymmetric padding so output spatial dims equal input spatial dims
    pad_total = kernel_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    m = nn.functional.pad(m, (pad_left, pad_right, pad_top, pad_bottom))
    m = nn.functional.conv2d(m, kernel)
    # require full overlap for erosion
    m = (m >= (kernel_size * kernel_size)).long()
    return m.squeeze(1)  # B, H, W

def filter_v3d_by_mask(v3d, mask, K, extrin):
    """
    Args:
        v3d: shape of (N, 3)
        mask: shape of (H, W)
        K: shape of (4, 4)
        extrin: shape of (4, 4)
    """
    if K.shape != (4, 4):
        _K = np.eye(4)
        _K[:3, :3] = K
        K = _K
    if extrin.shape != (4, 4):
        _extrin = np.eye(4)
        _extrin[:3, :3] = extrin
        extrin = _extrin
    
    v3d_homo = np.ones((v3d.shape[0], 4))
    v3d_homo[:, :3] = v3d
    # Project 3D points to 2D
    v2d_homo = K @ extrin @ v3d_homo.T
    v2d = (v2d_homo[:2] / v2d_homo[2]).T
    v2d = np.round(v2d).astype(int)

    # Create validity mask for points within image bounds
    valid_idx = (v2d[:, 0] >= 0) & (v2d[:, 0] < mask.shape[1]) & \
                (v2d[:, 1] >= 0) & (v2d[:, 1] < mask.shape[0])

    # Get mask values at projected points
    mask_values = np.zeros(len(v3d), dtype=bool)
    mask_values[valid_idx] = mask[v2d[valid_idx, 1], v2d[valid_idx, 0]]

    # Filter v3d points
    return v3d[mask_values], mask_values

def mask_IoU(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def maskout_depth(depths, masks):
    """
    Apply masks to depth maps.
    depths: F, H, W or H, W
    masks: F, H, W or H, W
    """
    if depths.shape != masks.shape:
        raise ValueError("Depths and masks must have the same shape.")
    masked = depths * masks
    masked[~masks.bool()] = 50.0  # large value for invalid pixels
    return masked


def project_points(points, K, extrin):
    """
    param points: [BS, N, 3]
    param K: [3, 3]
    param extrin: [4, 4]
    """
    if len(points.shape) == 2:
        BS = -1
    else:
        BS = points.shape[0]
    points = homo(points)
    _K = torch.eye(4).to(K.device)
    _K[:3, :3] = K[:3, :3]
    P = torch.matmul(_K, extrin)
    if BS > 0:
        points = torch.einsum("ij,BNj->BNi", P, points)
    else:
        points = torch.einsum("ij,Nj->Ni", P, points)
    points = points / points[..., 2:3]
    return points[..., :2]

def project_points_np(points, K, extrin, return_depth=False):
    """
    param points: [BS, N, 3]
    param K: [3, 3]
    param extrin: [4, 4]
    """
    BS = points.shape[0]
    # Add homogeneous coordinate
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    # Calculate projection matrix
    _K = np.eye(4)
    _K[:3, :3] = K[:3, :3]
    P = np.matmul(_K, extrin)
    # Tile P to match batch size
    P = np.tile(P[None], (BS, points.shape[1], 1, 1))
    
    # Project points
    points = np.einsum("BNij,BNj->BNi", P, points)
    # Normalize by depth
    points = points / points[..., 2:3]
    points = points / points[..., 2:]
    if return_depth:
        return points[..., :2], points[..., 2]
    return points[..., :2]


def plot_points_in_image(points, image, color=(55, 196, 232), radius=2, thickness=-1):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(pil_image)
    
    # Convert color tuple to a valid hex string for outline
    if isinstance(color, tuple) and len(color) == 3:
        outline_color = '#%02x%02x%02x' % color
    else:
        outline_color = color

    for point in points:
        x, y = point[:2].astype(np.int32)
        # If thickness is -1, fill the circle
        if thickness == -1:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        else:
            # Draw an outline with the specified thickness
            for i in range(thickness):
                draw.ellipse((x-radius-i, y-radius-i, x+radius+i, y+radius+i), outline=outline_color)
    
    # Convert back to numpy array
    return np.array(pil_image)


def proj_pc_on_image(xyzs, img, K, extrin, dot_size=3, output_path=None):
    img = to_numpy(img)
    # if (len(K.shape) == 2):
    #     K = K[None]
    # if (len(extrin.shape) == 2):
    #     extrin = extrin[None]
    # if (len(xyzs.shape) == 2):
    #     xyzs = xyzs[None]
    
    xyz2d = project_points(
        to_tensor(xyzs), 
        to_tensor(K), 
        to_tensor(extrin)
    )
    xyz2d = to_numpy(xyz2d)
    
    if (np.max(img) <= 2.0):
        img = (img * 255).astype(np.uint8)
    # transform mask
    if (len(img.shape) == 2):
        img = np.array(Image.fromarray(img).convert('RGB'))
        loguru.debug(f'Converted mask image to {img.shape}')
    
    synth = plot_points_in_image(xyz2d, img, radius=dot_size)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(synth).save(output_path)
    return synth
