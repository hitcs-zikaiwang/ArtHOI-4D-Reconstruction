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

DEVICE = torch.device("cuda")

from hrse.dataclass.primitives import TransformParams, ArtiPart
from hrse.dataclass.dataset import ArtiDataset
from hrse.object.asr_cano_proc import cut_mesh_by_vertex_mask
import hrse.utils.image_utils as imaging

VMAP_VOTE_THRESHOLD = 0.75

def faces_in_vmap(faces, vmap):
    """
    Keep faces fully inside vmap and remap vertex indices to local [0, len(vmap)-1].
    Args:
        faces: (M, 3) array-like of int (np.ndarray or torch.Tensor)
        vmap:  (K,) array-like of int, global vertex indices kept in the part
    Returns:
        np.ndarray shape (M_kept, 3) with reindexed faces
    """
    vmap = np.asarray(vmap).astype(np.int64)
    if torch.is_tensor(faces):
        faces_np = faces.detach().cpu().numpy().astype(np.int64)
    else:
        faces_np = np.asarray(faces).astype(np.int64)

    if faces_np.size == 0 or vmap.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # filter faces whose three vertices are all in vmap
    in_mask = np.isin(faces_np, vmap)
    keep = in_mask.all(axis=1)
    faces_kept = faces_np[keep]
    if faces_kept.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # remap global vertex indices -> local indices [0..K-1] according to vmap order
    max_vidx = int(faces_np.max())
    remap = np.full(max_vidx + 1, -1, dtype=np.int64)
    remap[vmap] = np.arange(vmap.shape[0], dtype=np.int64)
    faces_local = remap[faces_kept]
    return faces_local

def save_parts_by_vertex_mask(obj, seq, args):
    """Fallback to the original projection-based mesh cut."""
    for i in range(args.part_cnt):
        mask = seq.masks[f'part_{i}'][args.cano_frame]
        mask = imaging.dilate_mask(mask, kernel_size=getattr(args, 'c_dilate', 5))
        verts_part, faces_part, colors_part, _ = cut_mesh_by_vertex_mask(
            obj.verts, obj.faces, obj.colors,
            mask,
            seq.K[args.cano_frame],
            seq.w2c[args.cano_frame],
        )

        part_mesh = trimesh.Trimesh(
            vertices=verts_part.cpu().numpy(),
            faces=faces_part.cpu().numpy(),
            vertex_colors=colors_part.cpu().numpy(),
            process=False,
        )
        part_mesh.export(f"{args.output_path}/part_{i}_partmesh.obj")

        part = ArtiPart(
            verts=verts_part,
            faces=faces_part,
            colors=colors_part,
            device=DEVICE
        )
        part.cano_normalization()
        np.save(f"{args.output_path}/part_{i}_partdata.npy",
                part.dump_dict(), allow_pickle=True
        )


def save_parts_by_vmap_vote(obj, seq, args, vmaps, threshold=VMAP_VOTE_THRESHOLD):
    """Assign PartField clusters to part masks by vertex voting and export them."""
    # Project the ASR mesh onto the canonical frame once for all part masks.
    verts = obj.verts.clone().detach()
    verts[:, :2] *= -1
    verts_homo = torch.cat([
        verts,
        torch.ones((len(verts), 1), device=verts.device),
    ], dim=1)
    verts_cam = (seq.w2c[args.cano_frame] @ verts_homo.T).T[:, :3]
    verts_proj = verts_cam @ seq.K[args.cano_frame].T
    verts_proj = (verts_proj[:, :2] / verts_proj[:, 2:3]).round().long()

    # Record which mesh vertices fall inside each projected part mask.
    mask_vertex_votes = []
    for i in range(args.part_cnt):
        mask = seq.masks[f'part_{i}'][args.cano_frame]
        mask = imaging.dilate_mask(mask, kernel_size=getattr(args, 'c_dilate', 5))
        h, w = mask.shape
        valid_proj = (
            (verts_proj[:, 0] >= 0) & (verts_proj[:, 0] < w) &
            (verts_proj[:, 1] >= 0) & (verts_proj[:, 1] < h) &
            (verts_cam[:, 2] > 0)
        )
        vertex_vote = torch.zeros(len(verts), dtype=torch.bool, device=verts.device)
        valid_idx = torch.where(valid_proj)[0]
        vertex_vote[valid_idx] = mask[
            verts_proj[valid_idx][:, 1],
            verts_proj[valid_idx][:, 0],
        ].bool()
        mask_vertex_votes.append(vertex_vote)

    # Assign each vmap to the mask containing at least threshold of its vertices.
    buckets = [[] for _ in range(args.part_cnt)]
    for vmap_idx, vmap in enumerate(vmaps):
        vmap = np.asarray(vmap, dtype=np.int64)
        if vmap.size == 0 or np.any(vmap < 0) or int(vmap.max()) >= len(verts):
            loguru.warning(f"Invalid vmap {vmap_idx}; fallback to projection split.")
            save_parts_by_vertex_mask(obj, seq, args)
            return

        ratios = [vertex_vote[vmap].float().mean().item()
                  for vertex_vote in mask_vertex_votes]
        mask_idx = int(np.argmax(ratios))
        if ratios[mask_idx] < threshold:
            loguru.critical(
                f"Vmap {vmap_idx} best vote ratio {ratios[mask_idx]:.3f} is below "
                f"{threshold:.3f}; fallback to projection split.\n"
                f"Please check and adjust PartField cluster count if you want to proceed"
                f"with vmap-based separation."
            )
            save_parts_by_vertex_mask(obj, seq, args)
            return
        buckets[mask_idx].append(vmap)

    # An empty bucket is also ambiguous, so use the projection fallback.
    if args.use_2d_mask or any(len(bucket) == 0 for bucket in buckets):
        loguru.warning("Empty mask bucket; fallback to projection split.")
        save_parts_by_vertex_mask(obj, seq, args)
        return

    # Merge the clusters assigned to each mask and export one mesh per part.
    for i, bucket in enumerate(buckets):
        vmap = np.unique(np.concatenate(bucket, axis=0)).astype(np.int64)
        verts_part = obj.verts[vmap]
        colors_part = obj.colors[vmap]
        faces_part = faces_in_vmap(obj.faces, vmap)

        part_mesh = trimesh.Trimesh(
            vertices=verts_part.cpu().numpy(),
            faces=faces_part,
            vertex_colors=colors_part.cpu().numpy(),
            process=False,
        )
        part_mesh.export(f"{args.output_path}/part_{i}_partmesh.obj")

        part = ArtiPart(
            verts=verts_part,
            faces=faces_part,
            colors=colors_part,
            device=DEVICE
        )
        part.cano_normalization()
        np.save(f"{args.output_path}/part_{i}_partdata.npy",
                part.dump_dict(), allow_pickle=True
        )

def cano_register(
    args : edict,
    seq : ArtiDataset = None,
):
    seq_name = args.seq_path.strip('/').split('/')[-1]
    mesh_path = f'{args.asr}/final_mesh_asr.obj'
    obj = ArtiPart(None, None, None, DEVICE, name='obj')
    obj.load_from_mesh(mesh_path)

    # load pose
    cano_pose = np.load(f'{args.asr}/pred_pose.npy', allow_pickle=True)

    pt3d_flip = torch.diag(torch.tensor([-1, -1, 1], device=DEVICE)).float()
    rot = torch.tensor(cano_pose[:3, :3]).to(DEVICE).float()
    transl = torch.tensor(cano_pose[:3, 3]).to(DEVICE).float()
    rot = pt3d_flip @ rot
    transl = pt3d_flip @ transl

    # apply transformation to mesh
    obj.verts = (rot @ obj.verts.T).T + transl.unsqueeze(0)

    # load vmap
    vmap_paths = os.listdir(f'{args.pfsep}')
    vmap_paths = sorted(vmap_paths)
    vmaps = []
    for vm in vmap_paths:
        if not vm.endswith('_vmap.npy'):
            continue
        vmap = np.load(f'{args.pfsep}/{vm}', allow_pickle=True)
        vmaps.append(vmap)

    if args.part_order != 'default':
        order = [int(i) for i in args.part_order.split(',')]
        vmaps = [vmaps[i] for i in order]
    if args.part_merge != 'default':
        merge_groups = args.part_merge.split(';')
        loguru.debug(f"merge into {len(merge_groups)} parts: {merge_groups}")
        new_vmaps = []
        for group in merge_groups:
            part_ids = [int(i) for i in group.split(',')]
            merged_vmap = np.concatenate([vmaps[i] for i in part_ids], axis=0)
            new_vmaps.append(merged_vmap)
        vmaps = new_vmaps

    ## Manual Match
    # for i in range(args.part_cnt):
    #     verts_part = obj.verts[vmaps[i]]
    #     colors_part = obj.colors[vmaps[i]]
    #     faces_part = faces_in_vmap(obj.faces, vmaps[i])
    #
    #     part_mesh = trimesh.Trimesh(
    #         vertices=verts_part.cpu().numpy(),
    #         faces=faces_part,
    #         vertex_colors=colors_part.cpu().numpy(),
    #         process=False,
    #     )
    #     part_mesh.export(f"{args.output_path}/part_{i}_partmesh.obj")
    #
    #     part = ArtiPart(
    #         verts=verts_part,
    #         faces=faces_part,
    #         colors=colors_part,
    #         device=DEVICE
    #     )
    #     part.cano_normalization()
    #     np.save(f"{args.output_path}/part_{i}_partdata.npy",
    #             part.dump_dict(), allow_pickle=True
    #     )
    save_parts_by_vmap_vote(obj, seq, args, vmaps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Part Seperation')
    parser.add_argument('--seq-path', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--output-path', type=str, 
                        default=f'./outputs/{datetime.now().strftime("%m%d/%H%M%S")}/')
    parser.add_argument('--asr', type=str)
    parser.add_argument('--pfsep', type=str)
    parser.add_argument('--part_order', type=str, default='default')
    parser.add_argument('--part_merge', type=str, default='default')
    parser.add_argument('--c_dilate', type=int, default=5)
    parser.add_argument('--use_2d_mask', action='store_true', 
                        help='Use 2D mask based separation instead of PartField separation')
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
        args=args,
        seq=seq,
    )
