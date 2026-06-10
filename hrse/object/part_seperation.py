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

def cano_register(
    args : edict,
):
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
    
    for i in range(args.part_cnt):
        verts_part = obj.verts[vmaps[i]]
        colors_part = obj.colors[vmaps[i]]
        faces_part = faces_in_vmap(obj.faces, vmaps[i])
        
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
    args = parser.parse_args()
    args = edict(vars(args))
    conf = edict(OmegaConf.load(args.conf))
    args.update(conf)
    
    seq_name = args.seq_path.strip('/').split('/')[-1]
    args.output_path = os.path.join(args.output_path, seq_name)
    os.makedirs(args.output_path, exist_ok=True)

    cano_register(
        args=args,
    )

