import os
import trimesh
import numpy as np
import cv2
import yaml
import torch
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from PIL import Image
import argparse
from loguru import logger as loguru
from pytorch_lightning import seed_everything
import nvdiffrast.torch as dr

from hrse.dataclass.dataset import ArtiDataset
from hrse.object.asr_estimator import ASR
from utils.data_util import to_tensor
from hrse.dataclass.primitives import ArtiPart

import sys
__cur_path__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{__cur_path__}/../third_party")
from foundationpose.Utils import visualize_frame_results, align_mesh_to_coordinate

glctx = dr.RasterizeCudaContext()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda")

def load_mesh_arthoi(mesh_path, device):
    mesh_c = trimesh.load_mesh(mesh_path, process=False)
    verts = to_tensor(mesh_c.vertices, device=device)
    faces = to_tensor(mesh_c.faces, device=device)
    # trimesh stores vertex colors in a VertexColor object, access via .data
    if hasattr(mesh_c.visual, 'vertex_colors'):
        colors = to_tensor(mesh_c.visual.vertex_colors[:, :3], device=device) # take only RGB, ignore A if present
    else: # if no vertex colors, create a default white color
        loguru.warning(f"No vertex colors found in mesh {mesh_path}, using default color.")
        colors = torch.ones_like(verts, device=device) * 0.8
    return verts, faces, colors, mesh_c

def synth_tf(obbR, center):
    """
    obbR: np.ndarray (3, 3)
    center: np.ndarray (3,)
    mesh_o3d.rotate(np.linalg.inv(obb.R), center=center)
    new_mesh = (old_mesh - center) @ obb.R.T + center
    ->
    tf: new_mesh = tf @ old_mesh
    """
    R_inv = np.linalg.inv(obbR)
    t = center - R_inv @ center
    tf = np.eye(4)
    tf[:3, :3] = R_inv
    tf[:3, 3] = t
    return tf


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


if __name__=='__main__':
    seed_everything(0)
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--seq-path", type=str, default="", help="Path to the input sequence")
    parser.add_argument("--conf", type=str)
    parser.add_argument("--out", type=str, default='results')
    args = parser.parse_args()
    args = edict(vars(args))
    conf = edict(OmegaConf.load(args.conf))
    args.update(conf)
    
    seq_name = args.seq_path.strip('/').split('/')[-1]
    loguru.info(f"Processing sequence: {seq_name}")
    seq = ArtiDataset(args.seq_path, use_video_inpainting=True, fix_intr=True, fix_extr=True)
    
    save_path = f'{args.out}/{seq_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    CF = int(args.cano_frame)
    im_color = (seq.rgbs[CF] * 255).astype(np.uint8)
    im_depth = seq.depths[CF] 
    # load mesh seperately from dataset to avoid any processing
    (mesh_v, mesh_f, mesh_c, mesh) = load_mesh_arthoi(
        f"{args.seq_path}/build/mesh/{CF:05d}.obj", device
    )
    r_mesh, obbR, center = align_mesh_to_coordinate(mesh)
    mesh_tf = synth_tf(obbR, center)
    # other data
    intrinsic = seq.K[CF].cpu().numpy()
    mask = seq.masks['obj'][CF].astype(np.bool_)
    
    est = ASR(symmetry_tfs=None, mesh=r_mesh, debug_dir=save_path, debug=2)

    loguru.debug(f'im_rgb: {im_color.shape}, im_depth: {im_depth.shape}, mask: {mask.shape}, K: {intrinsic}')
    pred_pose, pred_score, pred_scale, IoU, pt3d_rendered = est.register_asr(
        K=intrinsic, 
        rgb=im_color, 
        depth=im_depth, 
        ob_mask=mask, 
        asr_iter=args.asr_iter if hasattr(args, 'asr_iter') else 20,
        name=f'asr'
    )
    # pred_pose_raw = pred_pose @ mesh_tf
    
    # rot = to_tensor(pred_pose[:3, :3], device=device)
    # transl = to_tensor(pred_pose[:3, 3], device=device)
    # # flip to pytorch3d coordinate
    # flip_mat = torch.tensor(
    #     [[-1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, 1]],
    #     device=device,
    #     dtype=torch.float32
    # )
    # rot = flip_mat @ rot
    # transl = flip_mat @ transl
    
    # cano_tf = TransformParams(device=device, fit_scale=False, train=False)
    # cano_tf.rotation = torch.nn.Parameter(a6d_arti_ds.rmat_to_cont_6d(rot))
    # cano_tf.translation = torch.nn.Parameter(transl)
    # np.save(os.path.join(save_path, 'cano_tf.npy'), [cano_tf.get_param()])
    # loguru.info(f"Saved canonical transform to {os.path.join(save_path, 'cano_tf.npy')}")
    
    np.save(os.path.join(save_path, 'pred_pose.npy'), pred_pose)
    
    if False:
        with open(os.path.join(save_path, 'iou.txt'), 'w') as f:
            f.write(f'IoU: {IoU}\n')
            f.write(f'Foundation Pose Score: {pred_score}\n')
            f.write(f'Predicted Scale: {pred_scale}\n')
    
    visualize_frame_results(
        color=im_color,
        gt_mesh=None,
        est=est,
        K=intrinsic,
        gt_pose=None,
        pred_pose=pred_pose,
        metric=None,
        obj_f=f"{seq_name}",
        frame_idx=CF,
        save_path=save_path,
        glctx=glctx,
        name=f"demo",
        nocs_metric=False,
        est_only=True,
        save_on_folder=True
    )
    
    # assign part seperation to mask
    # run `process_pfsep.py`
    """
    python object/part_seperation.py \
        --seq-path /hrse-ds/ytb_waffle2 --conf conf/ytb_waffle2_100c.yml \
        --output-path /hdd1/exp/hrse_asr --asr /hdd1/exp/art_cano/asr_v2/ \
        --pfsep /hdd1/exp/arti/hrse-ds/ytb_waffle2/processed/partseps
    """