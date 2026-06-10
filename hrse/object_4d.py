import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from loguru import logger as loguru
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import shutil
from arrgh import arrgh

import utils.data_util as data_util
from hrse.utils.image_utils import proj_pc_on_image

from hrse.dataclass.dataset import ArtiDataset
from hrse.dataclass.primitives import TransformParams, ArtiPart
from hrse.utils.render import (
    create_meshes, 
    PhongRenderer
)
import hrse.utils.render as render

import hrse.tracking.run_cotracker as run_cotracker
from args_init import output_args, parse_args
import hrse.losses as losses
import hrse.utils.plot.vis_plotter as plotter

DEVICE = torch.device("cuda")

def part_loss(
        preds, gts, 
        params, frame_t, direction, frame_cnt,
        step, args, writer=None):
    enabled = args.use
    loss_dict = {}
    
    final_loss = torch.zeros(1).to(DEVICE)
    if 'track' in enabled:
        w, since, until = args.track
        if since <= step and step < until:
            track_loss = losses.masked_l1_loss(
                preds["tracks_3d"], gts["tracks_3d"],
                gts["tracks_visibles"],
            )
            loss_dict['track'] = track_loss
            final_loss += w * track_loss
    
    if 'tpmask' in enabled:
        w, since, until = args.tpmask
        if since <= step and step < until:
            tpmask_loss = losses.two_phrase_mask_loss(
                preds["mask"], gts["mask"],
            )
            loss_dict['tpmask'] = tpmask_loss
            final_loss += w * tpmask_loss
    
    if 'depth_ranking' in enabled:
        w, since, until = args.depth_ranking
        if since <= step and step < until:
            depth_loss = losses.depth_ranking_loss_batch(
                preds['depth'][None], gts['depth'][None],
                gts['mask'][None],
            )
            loss_dict['depth_ranking'] = depth_loss
            final_loss += w * depth_loss
    
    if 'depth' in enabled:
        w, since, until = args.depth
        if since <= step and step < until:
            depth_loss = losses.normed_depth_l1(
                preds['depth'], gts['depth'], 
            )
            loss_dict['depth'] = depth_loss
            final_loss += w * depth_loss

    delta = 1 if direction == 'forward' else -1
    r1 = frame_t - (1 * delta)
    r2 = frame_t - (2 * delta)
    if 'velo' in enabled:
        w, since, until = args.velo
        if since <= step and step < until:
            if ((r1 > 0) and (r1 < frame_cnt) and (params[r1] is not None)):
                velo_loss = losses.velocity_loss_fn(params, frame_t, direction)
                loss_dict['velo'] = velo_loss
                final_loss += w * velo_loss
    if 'accel' in enabled:
        w, since, until = args.accel
        if since <= step and step < until:
            if ((r2 > 0) and (r2 < frame_cnt) and (params[r2] is not None)):
                accel_loss = losses.accel_loss_fn(params, frame_t, direction)
                loss_dict['accel'] = accel_loss
                final_loss += w * accel_loss
    
    # Add losses to tensorboard if writer is available
    if writer is not None:
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(f'frame_{frame_t}/{loss_name}', loss_value.item(), step)
    
    return final_loss


def part_perframe_align(
        frame_no, 
        seq : ArtiDataset, 
        part : ArtiPart,
        part_name : str,
        params_list: list, 
        tracks_3d_dict,
        ref_frames,
        args, 
        direction='forward',
        writer=None):
    part_args = args.per_part
    num_iter = part_args.num_iter
    vis_every = part_args.vis_every
    assert num_iter >= part_args.losses.use_best_after, "num_iter should be larger than use_best_after"
        
    dump_rgb_path = f"{args.output_path}/{part_name}/vis_rgb"
    dump_dino_path = f"{args.output_path}/{part_name}/vis_dino"
    dump_depth_path = f"{args.output_path}/{part_name}/vis_depth"
    os.makedirs(dump_rgb_path, exist_ok=True)
    os.makedirs(dump_dino_path, exist_ok=True)
    os.makedirs(dump_depth_path, exist_ok=True)
    
    # prepare imaging data and mesh
    K = seq.K[frame_no]
    rgb = seq.rgbs[frame_no]
    mask = seq.masks[part_name][frame_no]
    depth = seq.depths[frame_no]
    # reset optimizer and scheduler
    param : TransformParams = params_list[frame_no]
    param.train()
    param.defrost()
    optimizer = optim.Adam(param.parameters(), lr=part_args.learning_rate)
    final_lr = part_args.final_learning_rate
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=final_lr / part_args.learning_rate,
        total_iters=int(0.8 * num_iter)
    )
    # exp_gamma = 1.0 - (
    #     torch.exp((
    #         torch.log(torch.Tensor([final_lr]).double() / 
    #                   torch.Tensor([part_args.learning_rate]).double()
    #         ) / num_iter)
    #     )).item()
    # scheduler = optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=exp_gamma,
    #     last_epoch=-1
    # )
        
    renderer = PhongRenderer(K, seq.img_size, device=DEVICE)

    if hasattr(part_args, 'freeze_transl_until'):
        if part_args.freeze_transl_until > 0:
            param.freeze_transl()
    
    depth[1 - mask.int()] = render.RENDER_MAX_DEPTH
    gt = {
        'mask': mask, 
        'rgb': rgb, 
        'depth': depth,
    }
    best_loss = float('inf')
    best_param, best_pred = None, None
    for i in (pbar := tqdm(range(num_iter))):
        optimizer.zero_grad()
        
        if hasattr(part_args, 'freeze_transl_until'):
            if i >= part_args.freeze_transl_until:
                param.defrost_transl()
        
        # here skipped camera-2-world transformation (extr); add if needed
        vert_p = param.forward(part.verts)
        vert_p = part.apply_denorm(vert_p)  # denormalize
        
        # render mask
        meshes = create_meshes(
            vert_p.unsqueeze(0), 
            part.faces, 
            part.colors, 
            DEVICE
        ) 
        pred = renderer.render_mesh(meshes)
        pred['verts'] = vert_p

        # compute tracks motion reference
        if tracks_3d_dict is not None and len(ref_frames) > 0:
            t3ds = tracks_3d_dict['tracks_3d']
            visibles = tracks_3d_dict['visibles'].int()
            
            ref_frame = np.random.choice(ref_frames)
            visibles_x = visibles[:, frame_no] * visibles[:, ref_frame]
            t3d_ref = t3ds[:, ref_frame]
            t3d_ref = params_list[ref_frame].backward(part.apply_norm(t3d_ref))
            t3d_posed = param.forward(t3d_ref)
            t3d_posed= part.apply_denorm(t3d_posed)
            t3d_now = t3ds[:, frame_no]
            gt.update({'tracks_3d': t3d_now, "tracks_visibles": visibles_x})
            pred['tracks_3d'] = t3d_posed
        
        if (vis_every > 0) and (i % vis_every == 0):
            plotter.plot_rgb_masks(
                rgb, mask,
                pred['rgb'], pred['mask'],
            )
        
        loss = part_loss(
            pred, gt, 
            params_list, 
            frame_no, direction, seq.frame_cnt,
            i, part_args.losses, writer=writer
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i >= part_args.losses.use_best_after and loss.item() < best_loss:
            best_loss = loss.item()
            best_param = deepcopy(param.get_param())
            best_pred = data_util.clone_and_detach(pred)
        
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar(f'{part_name}/frame_{frame_no}/learning_rate', current_lr, i)
        writer.add_scalar(f'{part_name}/frame_{frame_no}/best_loss', best_loss, i)
        pbar.set_description(
            f"F{frame_no}|BestLoss={best_loss:.4f}|LR={current_lr:.4f}"
        )
    
    # fill back best param
    param.scale.data = torch.Tensor([best_param['scale']]).to(DEVICE)
    param.rotation.data = torch.Tensor(best_param['rotation']).to(DEVICE)
    param.translation.data = torch.Tensor(best_param['translation']).to(DEVICE)
    param.freeze()
    param.eval()
    
    if args.dump_vis:
        plotter.plot_rgb_masks(
            rgb, mask,
            best_pred['rgb'], best_pred['mask'],
            save=f"{dump_rgb_path}/pred_{frame_no}.png",
        )
        plotter.plot_depth_masks(
            depth, mask,
            best_pred['depth'], best_pred['mask'],
            save=f"{dump_depth_path}/pred_{frame_no}.png",
        )
    
    return {
        "best_loss": best_loss,
    }


def part_alignment(
    seq : ArtiDataset, 
    part : ArtiPart,
    part_id : int,
    track3d_dict,
    cano_param : TransformParams,
    args,
    writer=None,
):
    """
    fit each frame's motion of a part, from the reference of canonical frame
    """
    loguru.info("Starting alignment optimization")
    
    part_name = f'part_{part_id}'

    # prepare params
    params = [TransformParams(device=DEVICE, fit_scale=False, train=False) for i in range(seq.frame_cnt)]
    best_losses = [-1 for i in range(seq.frame_cnt)]
    params[args.cano_frame] = cano_param
    best_losses[args.cano_frame] = 0.0
    
    # forward
    done_frames = [args.cano_frame]
    for frame_no in range(args.cano_frame + 1, seq.frame_cnt):
        if args.per_part.init_from_prev:
            params[frame_no].set_param(params[frame_no - 1].get_param())
        ref_frames = np.random.choice(done_frames, size=max(1, len(done_frames) // 2), replace=False)
        _ret = part_perframe_align(
            frame_no, 
            seq, 
            part,
            part_name,
            params,
            track3d_dict,
            ref_frames,
            args,
            direction='forward',
            writer=writer
        )
        best_losses[frame_no] = _ret['best_loss']
        done_frames.append(frame_no)
            
    # backward
    for frame_no in range(args.cano_frame - 1, -1, -1):
        if args.per_part.init_from_prev:
            params[frame_no].set_param(params[frame_no + 1].get_param())
        ref_frames = np.random.choice(done_frames, size=max(1, len(done_frames) // 2), replace=False)
        _ret = part_perframe_align(
            frame_no, 
            seq, 
            part,
            part_name,
            params,
            track3d_dict,
            ref_frames,
            args,
            direction='backward',
            writer=writer
        )
        best_losses[frame_no] = _ret['best_loss']
        done_frames.append(frame_no)
    
    for i in range(len(params)):
        writer.add_scalar(f'A_overall/best_loss', best_losses[i], i)
    
    loguru.info(f'optimized best params for {len(params)} frames')
    return params


if __name__ == "__main__":
    args = parse_args()
    args = edict(vars(args))
    conf = edict(OmegaConf.load(args.conf))
    args.update(conf)
    
    output_args(args, f"{args.output_path}/args.txt")
    loguru.add(f"{args.output_path}/log.txt")
    tb_log_dir = f"{args.output_path}/tensorboard"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    torch.set_grad_enabled(True)
    
    seq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=args.no_vi,
        fix_intr=args.fix_intr, 
        fix_extr=args.fix_extr
    )
    seq._to_device(DEVICE)
    
    
    ## ===canonical registeration===
    if args.cano_reg_ckpt is None:
        loguru.error("No object canonical frame pose-scale registration provided.")
    obj = ArtiPart(None, None, None, DEVICE, name='obj')
    obj.subparts = []
    for i in range(args.part_cnt):
        part_data = np.load(
            f'{args.cano_reg_ckpt}/part_{i}_partdata.npy', 
            allow_pickle=True
        ).item()
        part = ArtiPart(None, None, None, DEVICE, name=f'part_{i}')
        part.load_dict(part_data)
        obj.subparts.append(part)
        ## copy to output_path
        shutil.copy(
            f'{args.cano_reg_ckpt}/part_{i}_partdata.npy',
            f'{args.output_path}/part_{i}_partdata.npy'
        )
        if torch.equal(part.denorm_mat, torch.eye(4, device=part.denorm_mat.device)):
            part.cano_normalization()
    loguru.critical(f'Loaded part shapes from {args.cano_reg_ckpt} checkpoints')
    
    if args.skip_motion is True:
        exit(0)
    
    ## === per-part fitting ===
    parts = obj.subparts
    allpart_params = []
    allpart_tracks = []
    # get tracking as motion reference
    if args.tracking_ckpt is not None:
        allpart_tracks = np.load(
            f'{args.tracking_ckpt}/perpart_tracks.npy', 
            allow_pickle=True
        )
        shutil.copy(
            f'{args.tracking_ckpt}/perpart_tracks.npy',
            f'{args.output_path}/perpart_tracks.npy'
        )
    else:
        for part_id, part in enumerate(parts):
            track3d_dict = run_cotracker.track3d_on_seq_total(
                seq, part_id, args,
                interval=args.per_part.tracking.interval
            )
            allpart_tracks.append(track3d_dict)
        np.save(
            f"{args.output_path}/perpart_tracks.npy", 
            allpart_tracks, allow_pickle=True
        )
    # fit part
    if args.part_fit_ckpt is not None:
        for part_id, part in enumerate(parts):
            param_dump = np.load(
                f'{args.part_fit_ckpt}/part_{part_id}_best.npy', 
                allow_pickle=True
            )
            part_param = [TransformParams(device=DEVICE, fit_scale=False, train=False) 
                          for _ in range(len(param_dump))]
            for i, p in enumerate(param_dump):
                part_param[i].set_param(p)
            allpart_params.append(part_param)
            ## copy to output_path
            shutil.copy(
                f'{args.cano_reg_ckpt}/part_{part_id}_best.npy',
                f'{args.output_path}/part_{part_id}_best.npy'
            )
        loguru.critical(f'Loaded part params from {args.part_fit_ckpt} checkpoints')
    else:
        for part_id, part in enumerate(parts):
            """# debug
            bg_mask = torch.stack(seq.masks['obj'], dim=0)
            bg_mask = (1 - bg_mask).to(DEVICE)  # invert mask to get background
            vis_3d_o3d.vis_tracks_composite(
                seq, allpart_tracks[part_id], args,
                remove_fg_mask=bg_mask,
            )
            input('press enter to continue')
            """
            
            loguru.critical(f"Part {part_id} alignment")
            cano_param = TransformParams(device=DEVICE, fit_scale=False, train=False)
            part_best_params = part_alignment(
                seq, 
                part, part_id, 
                allpart_tracks[part_id],
                cano_param, 
                args, writer=writer,
            )
            allpart_params.append(part_best_params)
        
        for part_id, part in enumerate(parts):
            params = allpart_params[part_id]
            param_dump = []
            for i, p in enumerate(params):
                param_dump.append(p.get_param())
            np.save(f"{args.output_path}/part_{part_id}_best.npy", 
                    param_dump, allow_pickle=True)

