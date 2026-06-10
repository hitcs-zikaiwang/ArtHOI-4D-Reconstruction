import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from loguru import logger as loguru
from easydict import EasyDict as edict
import imageio
from arrgh import arrgh

DEVICE = torch.device("cuda")

import hrse.utils.render as render
from hrse.utils.render import (
    create_meshes, 
    create_meshes_merge, 
    PhongRenderer
)
import hrse.utils.image_utils as imaging
from hrse.dataclass.primitives import TransformParams, ArtiPart, MANOParams
from hrse.dataclass.dataset import ArtiDataset
import hrse.utils.plot.vis_plotter as plotter
from args_init import output_args, parse_args
import hrse.utils.tfs.transform_3d as tf


def render_train_views(
        hands : dict[str, MANOParams], # {'left': MANOParams, 'right': MANOParams}
        part_tfs : list[list[TransformParams]], # [part_cnt, F] of TransformParams
        parts : list[ArtiPart],
        seq : ArtiDataset,
        args,
        out_label='',
):
    K = seq.K[0]
    F = seq.frame_cnt
    H, W = seq.img_h, seq.img_w
    renderer = render.PhongRenderer(K, (H, W), device=DEVICE)
    os.makedirs(f"{args.output_path}/{out_label}/ho_vis", exist_ok=True)
    os.makedirs(f"{args.output_path}/{out_label}/ho_vis_depth", exist_ok=True)
    
    ## prepare & deform every part mesh
    obj_verts = []  ## [F, PN, (verts)]
    obj_faces = []  ## [PN, (faces)]
    obj_colors = [] ## [PN, (3)]
    for i in range(args.part_cnt):
        obj_faces.append(parts[i].faces)
        obj_colors.append((0.7, 0.7, 1.0))
    for f in range(F):
        verts_part = []
        for i in range(args.part_cnt):
            verts = parts[i].verts
            vp = part_tfs[i][f].forward(verts)
            vp = parts[i].apply_denorm(vp)
            verts_part.append(vp)
        obj_verts.append(verts_part)
    
    ## prepare or calc hand mesh real-time
    for f in tqdm(range(F), desc='Rendering HO views'):
        vs, fs, cs = [], [], []
        for h in hands:
            # cancel batch dim caused by index-forward
            v3d_p = hands[h].forward(f)['v3d'][0]
            # reverse x, y as pytorch3d's axis coords
            # v3d_p[..., :2] = -v3d_p[..., :2]  
            vs.append(v3d_p)
            fs.append(hands[h].faces)
            cs.append((0.4, 0.7, 1.0) if h == 'left' else (1.0, 0.4, 0.7))
            
        # add object mesh
        vs.extend(obj_verts[f])
        fs.extend(obj_faces)
        cs.extend(obj_colors)
        meshes = render.create_meshes_merge(
            vs, fs, cs, DEVICE
        )
        
        ## rend and save
        pred = renderer.render_mesh(meshes)
        pred_rgb = pred['rgb']
        pred_mask = pred['mask']
        pred_depth = pred['depth']
        normalized_depth = imaging.normalize_depth(pred_depth, pred_mask)
        normalized_gt_depth = imaging.normalize_depth(
            seq.depths[f], 
            imaging.erode_mask(seq.masks['composite'][f], args.depth_erode)
        )
        plotter.plot_rgb_masks_fast(
            seq.rgbs[f], seq.masks['composite'][f],
            pred_rgb, pred_mask,
            save=f"{args.output_path}/{out_label}/ho_vis/pred_{f}.png",
        )
        plotter.plot_depth_masks_fast(
            normalized_gt_depth, seq.masks['composite'][f],
            # nviseq.depths[f], nviseq.masks['composite'][f],
            normalized_depth, pred_mask,
            erode=args.depth_erode,
            save=f"{args.output_path}/{out_label}/ho_vis_depth/pred_{f}.png",
        )

def dump_video(
    preds, gts, F, save_name, fps=30,
):
    img_dir = f"{save_name}/imgs"
    os.makedirs(save_name, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    loguru.info(f'Saving to {save_name}')
    
    rgb_plots = []
    depth_plots = []
    for f in tqdm(range(F), desc="rendering plots"):
        pred_rgb, pred_depth, pred_mask = preds['rgb'][f], preds['depth'][f], preds['mask'][f]
        gt_rgb, gt_depth, gt_mask = gts['rgb'][f], gts['depth'][f], gts['mask'][f]
        # depth_plot = plotter.plot_depth_masks(
        #     gt_depth, gt_mask,
        #     pred_depth, pred_mask,
        #     ret_pic=True,
        # )
        rgb_plot = plotter.plot_rgb_masks_fast(
            gt_rgb, gt_mask,
            pred_rgb, pred_mask,
            ret_pic=True,
        )
        rgb_plots.append(rgb_plot)
        # depth_plots.append(depth_plot)
        rgb_plot.save(f"{img_dir}/{f:04d}.png")

    rgb_video_path = f"{save_name}/rgb.mp4"
    with imageio.get_writer(rgb_video_path, fps=fps) as writer:
        for frame in rgb_plots:
            writer.append_data(np.asanyarray(frame))
    # depth_video_path = f"{save_name}/depth.mp4"
    # with imageio.get_writer(depth_video_path, fps=fps) as writer:
    #     for frame in depth_plots:
    #         writer.append_data(np.asanyarray(frame))


def composite_vis(
    seq : ArtiDataset,
    parts : list[ArtiPart],
    params : list[list[TransformParams]],
    args,
):
    masks = torch.stack(seq.masks['obj'], dim=0).to(DEVICE)
    depths = torch.stack(seq.depths, dim=0).to(DEVICE)
    rgbs = torch.stack(seq.rgbs, dim=0).to(DEVICE)
    K = seq.K
    c2w = seq.c2w
    F = seq.frame_cnt
    renderer = PhongRenderer(K[0], seq.img_size, device=DEVICE)
    
    rend_frames = []
    rend_trunk = 100 # ~20GB
    
    num_trunks = (F + rend_trunk - 1) // rend_trunk
    for trunk_idx in range(num_trunks):
        start_frame = trunk_idx * rend_trunk
        end_frame = min((trunk_idx + 1) * rend_trunk, F)
        
        loguru.info(f"Rendering trunk {trunk_idx + 1}/{num_trunks}: frames [{start_frame}, {end_frame})")
        
        for f in tqdm(range(start_frame, end_frame), 
                      desc=f"Trunk {trunk_idx+1}/{num_trunks} frames", 
                      leave=False):
            vps, faces, colors = [], [], []
            for pid, part in enumerate(parts):
                # deal with per-part mesh
                param = params[pid][f]
                vp = param.forward(part.verts)
                vp = part.apply_denorm(vp)
                vps.append(vp)
                faces.append(part.faces)
                colors.append(part.colors)
            
            mesh_oneframe = create_meshes_merge(
                vps, faces, colors, DEVICE
            )
            rend = renderer.render_mesh(mesh_oneframe)
            rend = {
                'mask': rend['mask'].detach().cpu().numpy(),
                'rgb': rend['rgb'].detach().cpu().numpy(),
                'depth': rend['depth'].detach().cpu().numpy(),
            }
            rend_frames.append(rend)

        # Clear CUDA cache after processing each trunk to manage memory
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    # list of dicts -> dict of lists
    preds = {
        'mask': np.stack([rend['mask'] for rend in rend_frames], axis=0),
        'rgb': np.stack([rend['rgb'] for rend in rend_frames], axis=0),
        'depth': np.stack([rend['depth'] for rend in rend_frames], axis=0),
    }
    gts = {
        'mask': masks,
        'rgb': rgbs,
        'depth': depths,
    }
    dump_video(
        preds, gts,
        F, 
        f"{args.output_path}/vis",
        fps=10,
    )

def render_deform_oneframe(
    rgb, mask, K,
    parts : list[ArtiPart],
    params : list[TransformParams],
    args=None,
    save=None,
):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    H, W = rgb.shape[:2]
    renderer = PhongRenderer(K, (H, W), device=DEVICE)
    vps, faces, colors = [], [], []
    for pid, part in enumerate(parts):
        # deal with per-part mesh
        param = params[pid]
        vp = param.forward(part.verts)
        vp = part.apply_denorm(vp)
        vps.append(vp)
        faces.append(part.faces)
        colors.append(part.colors)
    
    mesh_oneframe = create_meshes_merge(
        vps, faces, colors, DEVICE
    )
    rend = renderer.render_mesh(mesh_oneframe)
    rend = {
        'mask': rend['mask'].detach().cpu().numpy(),
        'rgb': rend['rgb'].detach().cpu().numpy(),
        'depth': rend['depth'].detach().cpu().numpy(),
    }
    plotter.plot_rgb_masks(
        rgb, mask, 
        rend['rgb'], rend['mask'],
        save=save,
    )
    return rend

def main():
    args = parse_args()
    args = edict(vars(args))
    flip_fpose = True
    
    seq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=args.no_vi,
        fix_intr=True, 
        fix_extr=True
    )
    seq._to_device(DEVICE)
    
    part_cnt = int(input("input the part count: "))
    parts = []
    allpart_params = []
    for i in range(part_cnt):
        data_path = input(f"input the path to part_{i} part data / mesh file: ")
        if data_path.endswith('.obj') or data_path.endswith('.ply'):
            part = ArtiPart(None, None, None, DEVICE, name=f'part_{i}')
            part.load_from_mesh(data_path)
            # norm / denorm mats will be identities so it's okay.
        else:  # `.npy`
            flip_fpose = False
            part_data = np.load(
                data_path, 
                allow_pickle=True
            ).item()
            part = ArtiPart(None, None, None, DEVICE, name=f'part_{i}')
            part.load_dict(part_data)
            # part.denorm_mat = torch.eye(4, device=DEVICE)
            # part.norm_mat = torch.eye(4, device=DEVICE)
        parts.append(part)
        
        data_path = input(f"input the path to part_{i} motion params: ")
        param_dump = np.load(
            data_path, 
            allow_pickle=True
        )
        part_param = [TransformParams(device=DEVICE, fit_scale=False, train=False) 
                        for _ in range(len(param_dump))]
        for i, p in enumerate(param_dump):
            part_param[i].set_param(p)
        allpart_params.append(part_param)
        assert len(part_param) == seq.frame_cnt, \
            f"part {i} params count {len(part_param)} != seq frame count {seq.frame_cnt}"
    
    if flip_fpose:
        # The "pytorch3d coords flip"
        flip_mat = torch.tensor(
            [[-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]],
            device=DEVICE
        ).float()
        for params in allpart_params:
            for param in params:
                R = tf.cont_6d_to_rmat(param.rotation)
                R = flip_mat @ R
                param.rotation = torch.nn.Parameter(tf.rmat_to_cont_6d(R))
                T = param.translation.clone().detach()
                T = flip_mat @ T
                param.translation = torch.nn.Parameter(T)
    
    composite_vis(
        seq=seq,
        parts=parts,
        params=allpart_params,
        args=args,
    )


if __name__ == '__main__':
    main()