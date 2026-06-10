import torch
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F
from geomloss import SamplesLoss
from loguru import logger as loguru
from arrgh import arrgh

from hrse.dataclass.primitives import MANOParams, TransformParams


def cd_loss_fn(pred_verts, target_pcd):
    # pred_verts: [N, 3], target_pcd: [M, 3]
    # Unsqueeze to add batch dimension
    loss, _ = chamfer_distance(pred_verts.unsqueeze(0), target_pcd.unsqueeze(0))
    return loss

def icp_loss_fn(source_points, target_points):
    """
    Compute ICP (Iterative Closest Point) loss.
    source_points: [N, 3] tensor, source point cloud from predicted mesh vertices.
    target_points: [M, 3] tensor, target point cloud back-projected from the depth map.
    """
    # Use single_directional to compute only the source-to-target distance.
    loss, _ = chamfer_distance(
        source_points.unsqueeze(0), 
        target_points.unsqueeze(0),
        single_directional=True  # Compute only the source-to-target distance.
    )
    return loss

def velocity_loss_fn(
        params, 
        t_now, 
        direction,
        rot_weight=0.4,  # NOTE those weights are experimental
        trans_weight=2.0,):
    if direction == 'backward':
        t_now = t_now + 1
    trans_diff = torch.abs(params[t_now].translation - params[t_now - 1].translation)
    rot_diff = torch.abs(params[t_now].rotation - params[t_now - 1].rotation)
    loss = rot_weight * torch.mean(rot_diff) + trans_weight * torch.mean(trans_diff)
    return loss

def accel_loss_fn(
        current_param,
        t_now,
        direction,
        rot_weight=1.0,
        trans_weight=1.2,
        drop=0.0):
    if direction == 'backward':
        t_now = t_now + 2
    trans_diff = torch.abs(current_param[t_now].translation - 
                           2 * current_param[t_now - 1].translation + 
                           current_param[t_now - 2].translation)
    rot_diff = torch.abs(current_param[t_now].rotation - 
                         2 * current_param[t_now - 1].rotation + 
                         current_param[t_now - 2].rotation)
    loss = rot_weight * torch.mean(rot_diff) + trans_weight * torch.mean(trans_diff)
    if loss < drop:
        return torch.tensor(0.0).to(loss.device)
    return loss

def giou_loss_fn(pred_mask, gt_mask):
    """
    Generalized IoU loss between predicted and ground truth masks
    Args:
        pred_mask: (b, 1, h, w) tensor
        gt_mask: (b, 1, h, w) tensor
    Returns:
        loss: (1,) tensor
    """
    pred_mask_flat = pred_mask.flatten()
    gt_mask_flat = gt_mask.flatten()
    
    intersection = (pred_mask_flat * gt_mask_flat).sum()
    union = pred_mask_flat.sum() + gt_mask_flat.sum() - intersection
    
    # Calculate the area of the smallest enclosing box (convex hull)
    # In the case of masks, we can approximate this as the combined area
    enclosing_area = torch.max(pred_mask_flat.sum(), gt_mask_flat.sum())
    
    iou = intersection / (union + 1e-6)
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    
    # Return 1-GIoU as loss (to minimize)
    return 1 - giou


def masked_l1_loss(pred, target, mask, normalize=True):
    """
    pred: shape (N, ...) or (B, N, ...)
    target: shape (N, ...) or (B, N, ...)
    mask: shape (N,) or (B, N)
    dim: int, the dimension to reduce
    """
    sum_loss = F.l1_loss(pred, target, reduction='none').mean(dim=-1, keepdim=True)
    masked_loss = sum_loss * mask
    if normalize:
        return torch.mean(masked_loss) / (mask.sum() + 1e-8)
    else:
        return torch.mean(masked_loss)

def masked_l1_loss_batch(pred, target, mask, normalize=True):
    """
    pred: shape (B, N, ...)
    target: shape (B, N, ...)
    mask: shape (B, N)
    """
    sum_loss = F.l1_loss(pred, target, reduction='none').mean(dim=-1)
    masked_loss = sum_loss * mask
    if normalize:
        return masked_loss.sum() / (mask.sum() + 1e-8)
    else:
        return torch.mean(masked_loss)

def normed_depth_l1(pred, target):
    """
    Compute normalized depth L1 loss.
    pred: shape (H, W)
    target: shape (H, W)
    """
    pred_mean = pred.mean()
    pred_std = pred.std() + 1e-8
    pred_normalized = (pred - pred_mean) / pred_std
    target_mean = target.mean()
    target_std = target.std() + 1e-8
    target_normalized = (target - target_mean) / target_std

    # Compute L1 loss between normalized values
    loss = F.l1_loss(pred_normalized, target_normalized)
    return loss

def depth_ranking_loss_batch(
    preds, gts, 
    masks, 
    sample_permask=400
):
    """
    notice: mask must be eroded before calling! or else we'll get heavy misalignment
    preds: shape (B, H, W)
    gts: shape (B, H, W)
    masks: shape (B, H, W)
    """
    B = preds.shape[0]
    losses = torch.zeros(B).to(preds.device)
    for i in range(B):
        pred_depth = preds[i]
        gt_depth = gts[i]
        mask = masks[i]
        valid_ids = torch.where(mask)
        if len(valid_ids[0]) == 0:
            losses[i] = 0
            continue
        
        rand_samples = torch.randint(
            0, len(valid_ids[0]), (sample_permask,), 
            device=mask.device
        )
        rand_samples = (
            valid_ids[0][rand_samples], 
            valid_ids[1][rand_samples]
        )
        losses[i] = depth_ranking_fn(pred_depth[rand_samples], gt_depth[rand_samples])
    return losses.mean()

def depth_ranking_fn(rendered_depth, gt_depth):
    """
    taken from nerfstudio's implementation
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    
    modified from [H, W, 1] to [H, W]
    """
    m = 1e-4
    if rendered_depth.shape[0] % 2 != 0:
        # chop off one index
        rendered_depth = rendered_depth[:-1]
        gt_depth = gt_depth[:-1]
    dpt_diff = gt_depth[::2] - gt_depth[1::2]
    out_diff = rendered_depth[::2] - rendered_depth[1::2] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    return torch.nanmean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))

def two_phrase_mask_loss_batch(
        preds, gts
):
    """
    preds: shape (B, H, W)
    gts: shape (B, H, W)
    # NOTICE: this won't skip optimal transport loss
    """
    B = preds.shape[0]
    losses = torch.zeros(B).to(preds.device)
    for i in range(B):
        pred_mask = preds[i]
        gt_mask = gts[i]
        iou_loss = soft_iou_loss_fn(pred_mask, gt_mask)
        # tried to add OT loss no matter what IoU is, but still has very limited effect.
        # if iou_loss > 0.9:
        #     optimal_transport = compute_sinkhorn_loss(pred_mask, gt_mask)
        #     losses[i] = optimal_transport + iou_loss
        # else:
        #     losses[i] = iou_loss
        losses[i] = iou_loss
    return losses.mean()

def two_phrase_mask_loss(pred_mask, gt_mask):
    iou_loss = soft_iou_loss_fn(pred_mask, gt_mask)
    if iou_loss > 0.9:
        optimal_transport = compute_sinkhorn_loss(pred_mask, gt_mask)
        loss = optimal_transport + iou_loss
    else:
        loss = iou_loss
    return loss

def soft_iou_loss_fn(pred_mask, target_mask, smooth=1e-6):
    """
    taken from EasyHOI
    """
    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask + target_mask) - intersection 
    iou_loss = (intersection + smooth) / (union + smooth)
    return 1 - iou_loss

sinkhorn_loss = SamplesLoss('sinkhorn')
def compute_sinkhorn_loss(pred_mask, gt_mask, max_num = 1000, normalize=True):
    reg_loss = torch.abs(pred_mask.sum() - gt_mask.sum())
    # print("reg_loss: ", reg_loss.item())
    
    # Convert masks to "point clouds" with weights
    pred_pts = pred_mask.nonzero(as_tuple=False).float()
    pred_w = pred_mask[pred_mask != 0]
    
    pred_num = pred_pts.shape[0]
    if pred_num > max_num:
        indices = torch.randperm(pred_num)[:max_num]
        pred_pts = pred_pts[indices]
        pred_w = pred_w[indices]
        pred_num = max_num

    gt_pts = gt_mask.nonzero(as_tuple=False).float()
    gt_w = gt_mask[gt_mask != 0]
    
    gt_num = gt_pts.shape[0]
    if gt_num > max_num:
        indices = torch.randperm(gt_num)[:max_num]
        gt_pts = gt_pts[indices]
        gt_w = gt_w[indices]
        gt_num = max_num

    # Normalize weights
    pred_w /= pred_num
    gt_w /= gt_num
    
    if pred_w.numel() == 0 or pred_pts.numel() == 0:
        # Handle empty inputs explicitly
        main_loss = 0
        # print("main_loss: ", 0)
    else:
        # Compute Sinkhorn loss
        main_loss = sinkhorn_loss(pred_w.contiguous(), pred_pts.contiguous(), 
                                gt_w.contiguous(), gt_pts.contiguous())
        main_loss /= gt_num
        # print("main_loss: ", main_loss.item())
    
    loss = main_loss + 10 * reg_loss
    
    if normalize:
        loss = loss / (pred_mask.numel())
    return loss

def dino_2d_loss(preds, gts, masks=None):
    """
    compute dino 2d loss similar to CoRL'24-RSRD
    """
    if masks is not None:
        loss = (preds - gts)[masks.int()].norm(dim=-1).mean()
    loss = (preds - gts).norm(dim=-1).mean()
    return loss

def batch_velocity_loss_fn(params, rot_weight=1.0, trans_weight=2.0):
    n = len(params)
    if n < 2:
        return torch.tensor(0.0).to(params[0].translation.device)
    
    # Collect rotation and translation parameters for all timesteps.
    translations = torch.stack([p.translation for p in params])
    rotations = torch.stack([p.rotation for p in params])
    
    # Compute velocity differences: (t) - (t-1).
    trans_diffs = torch.abs(translations[1:] - translations[:-1])
    rot_diffs = torch.abs(rotations[1:] - rotations[:-1])
    
    losses = rot_weight * torch.mean(rot_diffs, dim=1) + trans_weight * torch.mean(trans_diffs, dim=1)
    
    return torch.sum(losses)

def batch_accel_loss_fn(
        params : list[TransformParams], 
        rot_weight=0.4, trans_weight=1.0, drop=0.0
):
    n = len(params)
    if n < 3:
        return torch.tensor(0.0).to(params[0].translation.device)
    
    # Collect rotation and translation parameters for all timesteps.
    translations = torch.stack([p.translation for p in params])
    rotations = torch.stack([p.rotation for p in params])
    
    # Compute acceleration differences: (t) - 2*(t-1) + (t-2).
    trans_diffs = translations[2:] - 2 * translations[1:-1] + translations[:-2]
    trans_loss = torch.norm(trans_diffs, dim=-1)
    rot_diffs = rotations[2:] - 2 * rotations[1:-1] + rotations[:-2]
    rot_loss = torch.norm(rot_diffs, dim=-1)
    losses = rot_weight * rot_loss + trans_weight * trans_loss
    
    if drop > 0:
        losses = losses[losses >= drop]
    if losses.numel() == 0:
        return torch.tensor(0.0).to(params[0].translation.device)
    return torch.mean(losses)

## ================= HO Losses =================

def hand_accel_batchloss_fn(
        mano_params : MANOParams,
        drop=0.0,
):
    transl = mano_params.transl  # [T, 3]
    T = transl.shape[0]
    if T < 3:
        return torch.tensor(0.0, device=transl.device)
    accel = transl[2:] - 2 * transl[1:-1] + transl[:-2]  # [T-2, 3]
    loss = torch.norm(accel, dim=-1)
    if drop > 0:
        loss = loss[loss >= drop]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=transl.device)
    return loss.mean()

def keypoint_gmof(
    pred : torch.Tensor, gt : torch.Tensor, 
    sigma=25.0
):
    """
    pred/gt : shape of (B, N, vdim), length of dims can be any positive integer
    """
    diff_x = (pred - gt).flatten(start_dim=0)
    x_squared = diff_x**2
    sigma_squared = sigma**2
    loss_val = (sigma_squared * x_squared) / (sigma_squared + x_squared + 1e-8)
    return loss_val

def keypoint_l2_loss(
    pred : torch.Tensor, gt : torch.Tensor, 
    normalize=True
):
    """
    pred/gt : shape of (B, N, vdim), length of dims can be any positive integer
    """
    diff = (pred - gt).flatten(start_dim=0)
    loss_val = diff.norm(dim=-1)
    if normalize:
        return loss_val.mean()
    else:
        return loss_val.sum()

def keypoint_l1_loss(
    pred : torch.Tensor, gt : torch.Tensor, 
    normalize=True
):
    """
    pred/gt : shape of (B, N, vdim), length of dims can be any positive integer
    """
    diff = (pred - gt).flatten(start_dim=0)
    loss_val = torch.abs(diff).sum(dim=-1)
    if normalize:
        return loss_val.mean()
    else:
        return loss_val.sum()

def HO_interaction_loss(v_tips, v_obj):
    """
    sum of each fingertip to the nearest object point
    v_tips: [ft, 3] tensor, fingertip positions
    v_obj: [obj, 3] tensor, object point positions
    """
    dist = torch.cdist(v_tips, v_obj)
    min_dist = torch.min(dist, dim=1)[0]  # shape [ft,]
    loss = torch.mean(min_dist) * 100.0  # NOTE scale to [cm], to a larger value
    return loss

def HO_interaction_hands_loss(v_hands, v_obj):
    """
    min of hand to the nearest object point
    v_tips: [ft, 3] tensor, fingertip positions
    v_obj: [obj, 3] tensor, object point positions
    """
    dist = torch.cdist(v_hands, v_obj)
    loss = dist.min() * 100.0  # NOTE scale to [cm], to a larger value
    return loss

# ================= Articulation Model Losses =================
def E_1DoF(T_relax, T_art):
    """
    1-DoF articulation energy
    T_relax: [F, 4, 4] tensor, the relative transform from parent to child in relaxed state
    T_art: [F, 4, 4] tensor, the relative transform from parent to child in articulated state
    return: [F,] tensor, the Frobenius norm squared of  T_relax and T_art difference
    """
    frames = T_relax.shape[0]
    T_relax_inv = torch.inverse(T_relax)  # [F, 4, 4]
    T_diff = torch.bmm(T_relax_inv, T_art)  # [F, 4, 4]
    I = torch.eye(4, device=T_relax.device, dtype=T_diff.dtype).unsqueeze(0).repeat(frames, 1, 1)  # [F, 4, 4]

    frobenius_cost = F.mse_loss(T_diff, I, reduction='none').sum(dim=(-2, -1))
    return frobenius_cost
