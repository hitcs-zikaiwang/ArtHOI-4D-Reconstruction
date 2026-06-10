import os
import cv2
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from loguru import logger as loguru

def metric_depth_to_pointcloud(
        depth, K, c2w, rgb=None, mask=None, 
        custom_focal=None,
        use_pytorch3d_coord=True,
        return_o3d: bool = True,
        cutoff_outlier = True,
        fast_impl = False,
):
    """
    Convert a depth map to a point cloud using PyTorch for GPU acceleration.
    
    Args:
        depth: depth map [H, W] or [1, H, W]
        K: camera intrinsics matrix [3, 3]
        c2w: camera-to-world transform matrix [4, 4] or [3, 4]
        rgb: optional RGB image [H, W, 3]
        mask: optional mask [H, W]
        custom_focal: optional custom focal length used to scale depth
        use_pytorch3d_coord: whether to use the PyTorch3D coordinate system
        return_o3d: whether to return an Open3D point cloud object; otherwise return PyTorch tensors for points and colors
        
    Returns:
        Point cloud as an Open3D PointCloud when return_o3d=True; otherwise a tuple (points, colors).
    """
    device = depth.device
    
    # Ensure depth uses [H, W] format.
    if len(depth.shape) == 3:
        depth = depth.squeeze(0)
    
    # Ensure all inputs are PyTorch tensors.
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    if not isinstance(c2w, torch.Tensor):
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    if rgb is not None and not isinstance(rgb, torch.Tensor):
        rgb = torch.tensor(rgb, dtype=torch.float32, device=device)
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
    
    height, width = depth.shape
    focal_length_x = K[0, 0]
    focal_length_y = K[1, 1]
    
    if custom_focal is not None:
        scale_factor = (custom_focal / K[0, 0])
        loguru.debug(f'depth z min: {depth.min().item()}, z max: {depth.max().item()}')
        depth = depth / scale_factor
        loguru.info(f'scaled depth by factor {scale_factor}')
        loguru.debug(f'depth z min: {depth.min().item()}, z max: {depth.max().item()}')
    
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    # M converts from source coordinate to PyTorch3D's coordinate system
    M = torch.eye(3, device=device)
    if use_pytorch3d_coord:
        M[0, 0] = -1.0
        M[1, 1] = -1.0
    
    # convert R, t from source to Py3D coordinate system
    R = M @ R @ torch.inverse(M)
    t = M @ t
    
    # Generate mesh grid and calculate point cloud coordinates
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    x = (x - width / 2) / focal_length_x
    y = (y - height / 2) / focal_length_y
    
    if mask is not None:
        depth = torch.where(mask > 0, depth, torch.tensor(-1.0, device=device))
    
    z = depth
    
    # Stack coordinates
    points = torch.stack((x * z, y * z, z), dim=-1)
    points = points.reshape(-1, 3)
    
    # Filter valid points (z > 0)
    valid_mask = points[:, 2] > 0
    valid_points = points[valid_mask]
    
    # Apply coordinate system transform
    valid_points = torch.matmul(M, valid_points.t()).t()
    
    # Camera to world transformation
    pts_world = torch.matmul(R, valid_points.t()).t() + t
    
    # Cutoff z values at mask boundaries
    if cutoff_outlier:
        if fast_impl:
            pts_world, outlier_mask = cutoff_z_fast(
                pts_world, K, torch.inverse(c2w),
                height, width,
                sensitivity=1.8, kernel_size=3
            )
        else:
            pts_world, outlier_mask = cutoff_z(
                pts_world, K, torch.inverse(c2w),
                height, width,
                sensitivity=1.8, kernel_size=3
            )
    
    if return_o3d:
        # Convert back to numpy and create Open3D point cloud
        pts_world_np = pts_world.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world_np)
        
        if rgb is not None:
            colors = rgb.reshape(-1, 3)
            valid_colors = colors[valid_mask]
            valid_colors = valid_colors[outlier_mask] if cutoff_outlier else valid_colors
            pcd.colors = o3d.utility.Vector3dVector(valid_colors.cpu().numpy())
        
        return pcd
    else:
        # Return PyTorch tensors
        if rgb is not None:
            colors = rgb.reshape(-1, 3)
            valid_colors = colors[valid_mask]
            valid_colors = valid_colors[outlier_mask] if cutoff_outlier else valid_colors
            return pts_world, valid_colors
        else:
            return pts_world, None

def metric_depth_to_pointcloud_np(
        depth, K, c2w, rgb=None, mask=None, 
        custom_focal=None,
        use_pytorch3d_coord=True,
) -> o3d.geometry.PointCloud:
    if len(depth.shape) == 3:
        depth = depth.squeeze(0)
    if type(depth) == torch.Tensor:
        depth = depth.cpu().numpy()
    if type(rgb) == torch.Tensor:
        rgb = rgb.cpu().numpy()
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()
    if type(c2w) == torch.Tensor:
        c2w = c2w.cpu().numpy()
    if type(K) == torch.Tensor:
        K = K.cpu().numpy()
    
    height, width = int(depth.shape[0]), int(depth.shape[1])
    focal_length_x = K[0, 0]
    focal_length_y = K[1, 1]
    if custom_focal is None:
        scale_factor = 1.0
    else:
        scale_factor = (custom_focal / K[0, 0])
        loguru.debug(f'depth z min: {depth.min()}, z max: {depth.max()}')
        depth = depth / scale_factor
        loguru.info(f'scaled depth by factor {scale_factor}')
        loguru.debug(f'depth z min: {depth.min()}, z max: {depth.max()}')
    
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    # M converts from source coordinate to PyTorch3D's coordinate system
    # see: https://github.com/open-mmlab/mmhuman3d/blob/main/docs_zh-CN/cameras.md
    M = np.eye(3)
    if use_pytorch3d_coord:
        M[0, 0] = -1.0
        M[1, 1] = -1.0
    # convert R, t from source to Py3D coordinate system
    R = M @ R @ np.linalg.inv(M)
    t = M @ t
    
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / focal_length_x
    y = (y - height / 2) / focal_length_y
    if mask is not None:
        depth = np.where(mask > 0, depth, -1)
    z = np.array(depth)
    
    # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)
    points = points.reshape(-1, 3)
    # points with z > 0 are not masked
    valid_mask = points[:, 2] > 0
    valid_points = points[valid_mask]
    # loguru.debug(f"valid points shape: {valid_points.shape}")
    
    valid_points = np.einsum('ij,Pj->Pi', M, valid_points)
    # camera to world
    pts_world_pt3d_format = np.einsum('ij,Pj->Pi', R, valid_points) + t

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world_pt3d_format)
    
    if rgb is not None:
        # bgr_to_rgb
        # rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_ = np.copy(rgb)
        colors = np.array(rgb_).reshape(-1, 3)
        valid_colors = colors[valid_mask]
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    
    return pcd

def cutoff_z(
    points: torch.Tensor,    # [N, 3] points in world space
    K: torch.Tensor,         # [3, 3] camera intrinsics
    w2c: torch.Tensor,       # [4, 4] world to camera transform
    height, width,
    sensitivity: float = 3.0, # higher value means more strict filtering
    kernel_size: int = 3     # size of neighborhood to check
) -> torch.Tensor:
    """
    Filter out points with abnormal z values at mask boundaries.
    Implemented with PyTorch so the computation can be GPU accelerated.
    
    Args:
        points: [N, 3] points in world space
        K: [3, 3] camera intrinsics
        w2c: [4, 4] world to camera transform
        sensitivity: threshold multiplier for z-value difference
        kernel_size: size of neighborhood window
    
    Returns:
        filtered_points: [M, 3] points after removing abnormal z points
    """
    N_raw = points.shape[0]
    device = points.device
    
    # Ensure all inputs are PyTorch tensors.
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    if not isinstance(w2c, torch.Tensor):
        w2c = torch.tensor(w2c, dtype=torch.float32, device=device)
    
    # Transform points to camera space
    points_homo = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=1)
    points_cam = torch.matmul(w2c, points_homo.t()).t()[:, :3]
    
    # Project points to image plane
    points_2d = torch.matmul(K, points_cam.t()).t()
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    points_2d = torch.round(points_2d).long()
    
    # Filter points outside image bounds
    h, w = height, width
    valid_idx = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    points_2d = points_2d[valid_idx]
    points_cam = points_cam[valid_idx]
    points = points[valid_idx]
    
    # Create z-buffer
    z_buffer = torch.full((h, w), -1.0, device=device)
    for i, (x, y) in enumerate(points_2d):
        if z_buffer[y, x] < 0:
            z_buffer[y, x] = points_cam[i, 2]
        else:
            z_buffer[y, x] = torch.min(z_buffer[y, x], points_cam[i, 2])
    
    # Calculate local z statistics
    pad = kernel_size // 2
    z_padded = torch.nn.functional.pad(z_buffer, (pad, pad, pad, pad), mode='constant', value=-1)
    
    # Use unfold to batch-extract neighborhoods instead of looping.
    # First create a mask image for point locations.
    point_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    point_mask_flat = torch.zeros((h*w), dtype=torch.bool, device=device)
    
    # Map valid 2D point locations onto the mask.
    valid_indices = points_2d[:, 1] * w + points_2d[:, 0]
    point_mask_flat[valid_indices] = True
    point_mask = point_mask_flat.reshape(h, w)
    
    # Use unfold to extract neighborhoods around each valid point.
    z_padded_unfold = torch.nn.functional.unfold(
        z_padded.unsqueeze(0).unsqueeze(0),
        kernel_size=(kernel_size, kernel_size),
        stride=1
    ).squeeze(0)  # shape: [kernel_size*kernel_size, h*w]
    
    # Reshape to [kernel_size*kernel_size, h, w].
    z_patches = z_padded_unfold.reshape(kernel_size*kernel_size, h, w)
    
    # Keep only neighborhoods at valid point locations.
    z_patches = z_patches[:, point_mask]  # shape: [kernel_size*kernel_size, num_valid_points]
    
    # Create a valid-point mask (patch > 0).
    patch_valid = z_patches > 0  # shape: [kernel_size*kernel_size, num_valid_points]
    
    # Count valid points in each point's neighborhood.
    valid_count = torch.sum(patch_valid, dim=0)  # shape: [num_valid_points]
    
    # Initialize a tensor that stores median differences for all points.
    z_diffs = torch.full((points_2d.shape[0],), float('inf'), device=device)
    
    # Find points with enough valid neighbors.
    valid_points_idx = torch.where(valid_count > 1)[0]
    
    # Process only points with enough valid neighbors.
    if len(valid_points_idx) > 0:
        # Get the center z values for these points.
        center_z = points_cam[valid_points_idx, 2]
        
        # Compute the median neighborhood difference for each such point.
        for i, idx in enumerate(valid_points_idx):
            patch_values = z_patches[:, idx]
            patch_valid_mask = patch_values > 0
            patch_valid_vals = patch_values[patch_valid_mask]
            z_diff = torch.abs(patch_valid_vals - center_z[i])
            z_diffs[idx] = torch.median(z_diff)
    
    # Calculate adaptive threshold
    valid_diffs_mask = z_diffs < float('inf')
    if not torch.any(valid_diffs_mask):
        return points, torch.ones(len(points), dtype=torch.bool, device=device)
    
    valid_diffs = z_diffs[valid_diffs_mask]
    threshold = torch.median(valid_diffs) * sensitivity
    valid_points = z_diffs <= threshold
    
    valid_points_full = torch.zeros(N_raw, dtype=torch.bool, device=device)
    valid_points_full[:len(valid_points)] = valid_points
    return points[valid_points], valid_points_full

def cutoff_z_fast(
    points: torch.Tensor,    # [N, 3] points in world space
    K: torch.Tensor,         # [3, 3] camera intrinsics
    w2c: torch.Tensor,       # [4, 4] world to camera transform
    height, width,
    sensitivity: float = 3.0, # higher value means more strict filtering
    kernel_size: int = 3     # size of neighborhood to check
) -> torch.Tensor:
    """
    Fast GPU-parallelized version using statistical thresholding instead of median calculation.
    Uses statistical thresholding instead of median calculation to improve GPU parallel efficiency.
    
    Args:
        points: [N, 3] points in world space
        K: [3, 3] camera intrinsics  
        w2c: [4, 4] world to camera transform
        height, width: image dimensions
        sensitivity: threshold multiplier for z-value difference
        kernel_size: size of neighborhood window
    
    Returns:
        filtered_points: [M, 3] points after removing abnormal z points
        valid_mask: [N,] boolean mask indicating which points are kept
    """
    N_raw = points.shape[0]
    device = points.device
    
    # Ensure all inputs are PyTorch tensors.
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    if not isinstance(w2c, torch.Tensor):
        w2c = torch.tensor(w2c, dtype=torch.float32, device=device)
    
    # Transform points to camera space
    points_homo = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=1)
    points_cam = torch.matmul(w2c, points_homo.t()).t()[:, :3]
    
    # Project points to image plane
    points_2d = torch.matmul(K, points_cam.t()).t()
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    points_2d = torch.round(points_2d).long()
    
    # Filter points outside image bounds
    h, w = height, width
    valid_idx = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    
    if not torch.any(valid_idx):
        return points[:0], torch.zeros(N_raw, dtype=torch.bool, device=device)
    
    points_2d_valid = points_2d[valid_idx]
    points_cam_valid = points_cam[valid_idx]
    points_valid = points[valid_idx]
    
    # Create z-buffer using scatter operations
    linear_indices = points_2d_valid[:, 1] * w + points_2d_valid[:, 0]
    z_buffer = torch.full((h * w,), float('inf'), device=device)
    z_buffer.scatter_reduce_(0, linear_indices, points_cam_valid[:, 2], reduce='amin')
    z_buffer = z_buffer.reshape(h, w)
    z_buffer = torch.where(z_buffer == float('inf'), -1.0, z_buffer)
    
    # Compute global z statistics as the baseline for adaptive thresholding.
    valid_z = points_cam_valid[:, 2]
    global_mean = torch.mean(valid_z)
    global_std = torch.std(valid_z)
    base_threshold = global_std * sensitivity
    
    # Use convolution to quickly compute neighborhood means and valid-point counts.
    pad = kernel_size // 2
    z_padded = torch.nn.functional.pad(z_buffer.unsqueeze(0).unsqueeze(0), 
                                       (pad, pad, pad, pad), mode='constant', value=-1)
    
    # Create a convolution kernel for neighborhood statistics.
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
    
    # Create a valid-point mask for locations where z > 0.
    valid_mask_2d = (z_buffer > 0).float().unsqueeze(0).unsqueeze(0)
    valid_padded = torch.nn.functional.pad(valid_mask_2d, (pad, pad, pad, pad), mode='constant', value=0)
    
    # Compute the number of valid neighbors at each location.
    neighbor_count = torch.nn.functional.conv2d(valid_padded, kernel, padding=0).squeeze() * (kernel_size * kernel_size)
    
    # Set z values in invalid regions to 0 for convolution.
    z_for_conv = torch.where(z_buffer > 0, z_buffer, 0.0).unsqueeze(0).unsqueeze(0)
    z_for_conv_padded = torch.nn.functional.pad(z_for_conv, (pad, pad, pad, pad), mode='constant', value=0)
    
    # Compute the neighborhood z-value sum.
    neighbor_sum = torch.nn.functional.conv2d(z_for_conv_padded, kernel, padding=0).squeeze() * (kernel_size * kernel_size)
    
    # Compute neighborhood mean z values while avoiding division by zero.
    neighbor_mean = torch.where(neighbor_count > 0, neighbor_sum / neighbor_count, torch.tensor(0.0, device=device))
    
    # Batch-compute the difference between each valid point and its neighborhood mean.
    y_coords, x_coords = points_2d_valid[:, 1], points_2d_valid[:, 0]
    center_z = points_cam_valid[:, 2]
    
    # Fetch neighborhood statistics at each point location.
    point_neighbor_mean = neighbor_mean[y_coords, x_coords]
    point_neighbor_count = neighbor_count[y_coords, x_coords]
    
    # Compute z-value differences.
    z_diff = torch.abs(center_z - point_neighbor_mean)
    
    # Adaptive threshold: combine global statistics with local neighbor counts.
    # More neighbors make the threshold stricter; fewer neighbors make it looser.
    adaptive_threshold = base_threshold * torch.clamp(3.0 / torch.sqrt(point_neighbor_count + 1), 0.5, 2.0)
    
    # Filter outliers: keep points whose difference is below the adaptive threshold and have enough neighbors.
    valid_points_mask = (z_diff <= adaptive_threshold) & (point_neighbor_count >= 2)
    
    # Construct the full valid-point mask.
    valid_points_full = torch.zeros(N_raw, dtype=torch.bool, device=device)
    valid_points_full[valid_idx] = valid_points_mask
    
    return points_valid[valid_points_mask], valid_points_full
