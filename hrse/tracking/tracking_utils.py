import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import os

from hrse.utils.image_utils import erode_mask, project_points

def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0

def parse_cotracker(visibility_2d):
    """
    Parse the visibility output from CoTracker.
    :param visibility_2d: [T, N, 1]
    :return: visibles, invisibles, confidences
    """
    visibles = visibility_2d > 0.5
    invisibles = visibility_2d < 0.5
    # confidences = visibility_2d.squeeze(-1)
    confidences = None
    return visibles, invisibles, confidences

def lift_track_to3d_oneframe(
    query_index: int,
    query_img: torch.Tensor,
    tracks_2d: torch.Tensor,
    visibility_2d: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    Ks: torch.Tensor,
    c2ws: torch.Tensor,
):
    """
    :param query_index (int)
    :param query_img [H, W, 3]
    :param tracks_2d [T, N, 2]
    :param visibility_2d [T, N]
    :param depths [T, H, W]
    :param masks [T, H, W]
    :param Ks [T, 3, 3]
    :param c2ws [T, 4, 4]
    returns (
        tracks_3d [N, T, 3]
        track_colors [N, 3]
        visibles [N, T]
        invisibles [N, T]
        confidences [N, T]
    )
    """
    T, H, W = depths.shape
    query_img = query_img[None].permute(0, 3, 1, 2)  # (1, 3, H, W)
    # tracks_2d = tracks_2d.swapaxes(0, 1)  # (T, N, 2)
    
    # (T, N), (T, N), (T, N)
    visibles, invisibles, confidences = parse_cotracker(visibility_2d)
    
    # Unproject 2D tracks to 3D.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    track_depths = F.grid_sample(
        depths[:, None],
        normalize_coords(tracks_2d[:, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[:, 0, 0]
    inv_Ks = torch.linalg.inv(Ks)
    tracks_3d = (
        torch.einsum(
            "nij,npj->npi",
            inv_Ks,
            F.pad(tracks_2d, (0, 1), value=1.0),
        )
        *
        track_depths[..., None]
    )
    
    # M converts from source coordinate to PyTorch3D's coordinate system
    # see: https://github.com/open-mmlab/mmhuman3d/blob/main/docs_zh-CN/cameras.md
    M = torch.eye(3).to(c2ws.device)
    if True:
        M[0, 0] = -1.0
        M[1, 1] = -1.0
        R = c2ws[..., :3, :3]
        t = c2ws[..., :3, 3]
        # convert R, t from source to Py3D coordinate system
        R = torch.einsum("ij,njk,kl->nil", M, R, torch.linalg.inv(M))
        t = torch.einsum("ij,nj->ni", M, t)
        c2ws[..., :3, :3] = R
        c2ws[..., :3, 3] = t
        tracks_3d = torch.einsum('ij,nPj->nPi', M, tracks_3d)
    # in [T, N, 3] -> [T, N, 4] -> [T, N, 3]
    tracks_4d = F.pad(tracks_3d, (0, 1), value=1.0)
    tracks_3d = torch.einsum("nij,npj->npi", c2ws, tracks_4d)[..., :3] 
    # Filter out out-of-mask tracks.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    is_in_masks = (
        F.grid_sample(
            masks[:, None],
            normalize_coords(tracks_2d[:, None], H, W),
            align_corners=True,
        )[:, 0, 0]
        == 1
    )
    visibles *= is_in_masks
    invisibles *= is_in_masks
    # confidences *= is_in_masks.float()

    # valid if in the fg mask at least 40% of the time
    # in_mask_counts = is_in_masks.sum(0)
    # t = 0.25
    # thresh = min(t * T, in_mask_counts.float().quantile(t).item())
    # valid = in_mask_counts > thresh
    valid = is_in_masks[query_index]
    # valid if visible 5% of the time
    visible_counts = visibles.sum(0)
    valid = valid & (
        visible_counts
        >= min(
            int(0.05 * T),
            visible_counts.float().quantile(0.1).item(),
        )
    )

    # Get track's color from the query frame.
    # (1, 3, H, W), (1, 1, N, 2) -> (1, 3, 1, N) -> (N, 3)
    track_colors = F.grid_sample(
        query_img,
        normalize_coords(tracks_2d[query_index : query_index + 1, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[0, :, 0].T
    return (
        tracks_3d[:, valid].swapdims(0, 1),
        track_colors[valid],
        visibles[:, valid].swapdims(0, 1),
        invisibles[:, valid].swapdims(0, 1),
        # confidences[:, valid].swapdims(0, 1),
        None,
    )

def vis_2d_tracks(out_path, frames, pred_tracks, pred_visibility, dump_name=None):
    import cotracker.utils.visualizer as covis
    os.makedirs(out_path, exist_ok=True)
    video = frames.permute(0, 3, 1, 2)[None].float().to(pred_tracks.device)
    if video.max() <= 1.0:
        video = video * 255 # denorm
    viser = covis.Visualizer(
        save_dir=out_path,
        linewidth=4,
        mode='cool',
        tracks_leave_trace=-1 
    )
    if dump_name is not None:
        filename=f'track2d_{dump_name}'
    else:
        filename=f'track2d_{datetime.now().strftime("%H%M%S")}'
    viser.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=filename,
    )

def query_grid_over_mask(
    grid_size, mask, erode_radius=5,
):
    eroded_mask = erode_mask(mask, kernel_size=erode_radius)
    positive_pixels = torch.nonzero(eroded_mask)
    if len(positive_pixels) == 0:
        return torch.empty((0, 2), device=mask.device)
    
    min_y, min_x = positive_pixels.min(dim=0).values
    max_y, max_x = positive_pixels.max(dim=0).values
    
    if isinstance(grid_size, (list, tuple)):
        grid_h, grid_w = grid_size
    else:
        grid_h = grid_w = grid_size
    
    y_step = max(1, (max_y - min_y) // (grid_h - 1))
    x_step = max(1, (max_x - min_x) // (grid_w - 1))
    
    # Sample points from grid
    grid_points = []
    for y in range(min_y, max_y + 1, y_step):
        for x in range(min_x, max_x + 1, x_step):
            if y < eroded_mask.shape[0] and x < eroded_mask.shape[1] and eroded_mask[y, x]:
                grid_points.append([x, y]) # cotracker do (x, y)==(W, H) instead of (y, x)
    return torch.tensor(grid_points).float()

def get_projected_tracks(
    tracks_3d: torch.Tensor,
    K: torch.Tensor,
    extr: torch.Tensor,
):
    """
    K: [3, 3]
    extr: [4, 4]
    """
    tracks_2d = project_points(
        tracks_3d,
        K,
        extr,
    )
    tracks_depth = tracks_3d[..., 2]
    return tracks_2d, tracks_depth

def plot_visibility_by_time(visibles, save_path):
    """
    Plot visibility over time.
    :param visibles: [N, T]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import matplotlib.pyplot as plt
    visibles = visibles.swapdims(0, 1)
    visibles = visibles.sum(axis=1)  # [T]
    plt.plot(visibles.cpu().numpy())
    plt.xlabel("Time")
    plt.ylabel("Visibility")
    plt.title("Visibility over Time")
    plt.savefig(save_path)
    plt.close()
