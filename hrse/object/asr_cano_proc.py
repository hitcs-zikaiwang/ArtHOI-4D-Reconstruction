import os
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from tqdm import tqdm
from loguru import logger as loguru
from arrgh import arrgh
from datetime import datetime
from typing import Tuple

from hrse.dataclass.primitives import TransformParams, ArtiPart
from hrse.dataclass.dataset import ArtiDataset
import hrse.utils.image_utils as imaging

DEVICE = torch.device("cuda")

def cut_mesh_by_vertex_mask(
    verts : torch.Tensor, 
    faces : torch.Tensor,
    colors : torch.Tensor,
    mask : torch.Tensor,
    K : torch.Tensor,
    w2c : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    loguru.debug(f"Processing mesh with {len(verts)} verts")
    loguru.debug(f"Image mask shape: {mask.shape}, valid: {torch.sum(mask)}")
    
    # reverse back xy axis from pytorch3d coord system back to open3d/opengl
    verts = verts.clone().detach()
    faces = faces.clone().detach()
    colors = colors.clone().detach()
    verts[:, :2] *= -1
    
    verts_homo = torch.cat([verts, torch.ones((len(verts), 1), device=verts.device)], dim=1)
    verts_cam = (w2c @ verts_homo.T).T[:, :3]
    verts_proj = verts_cam @ K.T
    verts_proj = verts_proj[:, :2] / verts_proj[:, 2:3]
    verts_proj = verts_proj.round().long()
    
    h, w = mask.shape
    valid_proj = (verts_proj[:, 0] >= 0) & (verts_proj[:, 0] < w) & \
                (verts_proj[:, 1] >= 0) & (verts_proj[:, 1] < h)
    vertex_mask = torch.zeros(len(verts), dtype=torch.bool, device=verts.device)
    valid_idx = torch.where(valid_proj)[0]
    vertex_mask[valid_idx] = mask[
        verts_proj[valid_idx][:, 1], 
        verts_proj[valid_idx][:, 0]
    ].bool()
    
    # Use z > 0 to filter out points behind the camera.
    vertex_mask = vertex_mask & (verts_cam[:, 2] > 0)
    
    # loguru.debug(f"Keeping {torch.sum(vertex_mask)} verts after projection and masking")
    
    # Collect vertices to keep.
    kept_verts = verts[vertex_mask]
    loguru.debug(f"Kept {len(kept_verts)} verts after masking")
    
    # Build a mapping from old vertex indices to new vertex indices.
    old_to_new = torch.full((len(verts),), -1, dtype=torch.long, device=verts.device)
    new_indices = torch.arange(torch.sum(vertex_mask), device=verts.device)
    old_to_new[vertex_mask] = new_indices
    
    # Find faces whose vertices are all kept.
    valid_faces_mask = torch.all(vertex_mask[faces], dim=1)
    kept_faces = faces[valid_faces_mask]
    loguru.debug(f"Kept {len(kept_faces)} faces after filtering")
    # Update vertex indices in faces.
    kept_faces = old_to_new[kept_faces]
    
    kept_colors = colors[vertex_mask] if colors is not None else None

    # Check if kept_faces is a valid mesh structure
    if kept_verts.shape[0] < 3 or kept_faces.shape[0] < 1:
        loguru.warning("Resulting mesh has too few vertices or faces to be valid.")
        raise ValueError("Resulting mesh is invalid after cutting by vertex mask.")

    # Ensure all face indices are within the range of kept_verts
    if kept_faces.max() >= kept_verts.shape[0] or kept_faces.min() < 0:
        loguru.error("Invalid face indices detected in kept_faces.")
        raise ValueError("Face indices are out of bounds after cutting by vertex mask.")
    
    ## reverse back verts from opengl to pytorch3d
    kept_verts[:, :2] *= -1
    
    return kept_verts, kept_faces, kept_colors, vertex_mask


def save_parts(
    obj : ArtiPart,
    args,
    suffix="partdata",
):
    parts = []
    vmasks = []
    for i in range(args.part_cnt):
        mask = seq.masks[f'part_{i}'][args.cano_frame]
        mask = imaging.dilate_mask(mask, kernel_size=args.c_dilate) 
        verts_part, faces_part, colors_part, vmask = cut_mesh_by_vertex_mask(
            obj.verts, obj.faces, obj.colors,
            mask,
            seq.K[args.cano_frame],
            seq.w2c[args.cano_frame],
        )
        part = ArtiPart(
            verts=verts_part,
            faces=faces_part,
            colors=colors_part,
            device=obj.device
        )
        parts.append(part)
        vmasks.append(vmask)
    obj.subparts = parts
    obj.vertex_masks = vmasks
    for i in range(len(obj.subparts)):
        part = obj.subparts[i]
        part.cano_normalization()
    for i in range(len(obj.subparts)):
        np.save(f"{args.output_path}/part_{i}_{suffix}.npy", 
                obj.subparts[i].dump_dict(), allow_pickle=True
        )
    np.save(f"{args.output_path}/part_obj_{suffix}.npy",
            obj.dump_dict(), allow_pickle=True
    )


def cano_register(
    seq : ArtiDataset,
    args : edict,
    save=True
):
    CF = args.cano_frame
    seq_name = args.seq_path.strip('/').split('/')[-1]
    mesh_path = f'{args.asr}/{seq_name}/final_mesh_demo.obj'
    obj = ArtiPart(None, None, None, DEVICE, name='obj')
    obj.load_from_mesh(mesh_path)
    
    # load pose
    cano_pose = np.load(f'{args.asr}/{seq_name}/pred_pose.npy', allow_pickle=True)
    
    pt3d_flip = torch.diag(torch.tensor([-1, -1, 1], device=DEVICE)).float()
    rot = torch.tensor(cano_pose[:3, :3]).to(DEVICE).float()
    transl = torch.tensor(cano_pose[:3, 3]).to(DEVICE).float()
    rot = pt3d_flip @ rot
    transl = pt3d_flip @ transl
    
    # apply transformation to mesh
    obj.verts = (rot @ obj.verts.T).T + transl.unsqueeze(0)
    
    try:
        save_parts(obj, args)
    except Exception as e:
        loguru.error(f"Failed to save parts: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='HOI 3x3 visualization')
    parser.add_argument('--seq-path', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--output-path', type=str, 
                        default=f'./outputs/{datetime.now().strftime("%m%d/%H%M%S")}/')
    parser.add_argument('--asr', type=str)
    parser.add_argument('--c_dilate', type=int, default=5)
    args = parser.parse_args()
    args = edict(vars(args))
    conf = edict(OmegaConf.load(args.conf))
    args.update(conf)
    
    seq_name = args.seq_path.strip('/').split('/')[-1]
    args.output_path = os.path.join(args.output_path, seq_name)
    os.makedirs(args.output_path, exist_ok=True)

    seq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=True,
        fix_intr=True, 
        fix_extr=True
    )
    seq._to_device(DEVICE)
    cano_register(
        seq=seq,
        args=args,
    )


"""
Proceed 2d-mask based canonical split on ASR result.
"""
