import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
from loguru import logger as loguru
from copy import deepcopy
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import shutil
from arrgh import arrgh
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    loguru.warning("trimesh not found; penetration stage will be skipped.")

# hack numpy for SMPLX
np.bool = np.bool_
np.int = np.int32

from hrse.utils.image_utils import proj_pc_on_image, project_points
import hrse.utils.plot.vis_composite_fitted as vis_composite
import hrse.dataclass.model_io as model_io
from hrse.dataclass.primitives import MANOParams, TransformParams, ArtiPart
from hrse.dataclass.dataset import ArtiDataset
import hrse.losses as losses

DEVICE = torch.device("cuda")

HAND_TIPS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}
ALL_SIDES = ['left', 'right']

def compute_loss(
        preds, gts, 
        sides : list[str],
        contacts,  # marks for per-frame contact t/f
        residual_hands : dict[str, TransformParams],
        hparams : dict[str, MANOParams],
        enabled,
        step, device, args, writer=None,
):
    F = preds["part_0"]['v2d'].shape[0]
    
    loss_dict = {}
    
    # projection error
    if 'projection_error' in enabled:
        pjerr_keys = [f'part_{i}' for i in range(args.part_cnt)]
        pj_err = torch.zeros(1, device=device)
        for k in pjerr_keys:
            # l2 = losses.keypoint_gmof(
            #     preds[k]['v2d'],
            #     gts[k]['v2d'],
            # ).mean(dim=-1)
            l2 = losses.keypoint_l2_loss(
                preds[k]['v2d'],
                gts[k]['v2d'],
                normalize=True,
            )
            pj_err += l2
            loss_dict[f'pj_err/{k}'] = l2.item()
        for s in sides:
            l2 = losses.keypoint_l2_loss(
                preds[s]['j2d'],
                gts[s]['j2d'],
                normalize=True,
            )
            pj_err += l2
            loss_dict[f'pj_err/{s}'] = l2.item()
    
    if 'contact_dis' in enabled:
        contact_err = {
            "left" : [torch.tensor(0.0, device=device)],
            "right": [torch.tensor(0.0, device=device)],
        }
        for hand_side in sides:
            for f in range(F):
                if contacts[hand_side][f]:
                    active_fingers = contacts[f"{hand_side}_fingers"][f]
                    active_fingers.remove('pinky') if 'pinky' in active_fingers else None
                    active_fingers.remove('ring') if 'ring' in active_fingers else None
                    tips = [preds[hand_side]['v3d'][f, HAND_TIPS[finger]] \
                            for finger in active_fingers]
                    cerr_f = losses.HO_interaction_loss(
                        torch.stack(tips, dim=0),
                        preds['obj']['v3d'][f],
                    )
                    contact_err[hand_side].append(cerr_f)
        contact_err["left"] = torch.stack(contact_err["left"], dim=0).mean()
        contact_err["right"] = torch.stack(contact_err["right"], dim=0).mean()
        loss_dict['contact_l'] = contact_err['left']
        loss_dict['contact_r'] = contact_err['right']
    
    # L2 regularization of hand transl residual
    if 'reg_transl' in enabled:
        reg_transl = torch.tensor(0.0, device=device)
        for s in sides:
            l2_dist = torch.norm(
                hparams[s].transl - gts[s]['transl'], dim=-1
            )
            reg_transl += l2_dist.mean()
        loss_dict['reg_transl'] = reg_transl
    # L1 reg of hand finger(joint) pose
    reg_pose_weight = 50.0
    if 'reg_pose' in enabled:
        reg_pose = torch.tensor(0.0, device=device)
        for s in sides:
            diff = hparams[s].hand_pose - gts[s]['hand_pose']
            reg_pose_s = diff.abs().sum(dim=-1).mean()
            reg_pose += reg_pose_s
        reg_pose *= reg_pose_weight
        loss_dict['reg_pose'] = reg_pose
    
    if 'accel' in enabled:
        # accel = torch.tensor(0.0, device=device)
        for s in sides:
            transl_z = hparams[s].transl[:, 2]
            accel_z = transl_z[2:] - 2 * transl_z[1:-1] + transl_z[:-2]
            accel_z = accel_z.abs().mean()
            loss_dict[f'accel_z_{s}'] = accel_z
    
    if 'velo' in enabled:
        for s in sides:
            transl_z = hparams[s].transl[:, 2]
            velo = transl_z[1:] - transl_z[:-1]
            velo_l1 = velo.abs().mean()
            loss_dict[f'velo_l1_{s}'] = velo_l1
    
    if writer is not None:
        for k, v in loss_dict.items():
            writer.add_scalar(f'loss/{k}', v, step)
    
    all_loss = torch.tensor(0.0, device=device)
    for k, v in loss_dict.items():
        all_loss += v
    writer.add_scalar('loss/total', all_loss.item(), step)
    return all_loss

def make_linear_decay(optimizer, total_steps, final_lrs):
    """
    Args:
        optimizer: torch optimizer with multiple param groups
        total_steps: total training steps
        final_lrs: list of final_lr values per group
    """
    init_lrs = [pg['lr'] for pg in optimizer.param_groups]
    assert len(init_lrs) == len(final_lrs), "final_lrs length must match optimizer.param_groups length"

    def make_fn(init_lr, final_lr):
        # Return a function that maps a step to this parameter group's scale factor.
        def fn(step):
            if total_steps <= 0:
                return 1.0
            ratio = max(0.0, min(step / float(total_steps), 1.0))
            curr = (1 - ratio) * init_lr + ratio * final_lr
            return curr / init_lr
        return fn

    lr_fns = [make_fn(init_lr, final_lr) for init_lr, final_lr in zip(init_lrs, final_lrs)]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fns)

def refined_opt(
    hp : dict[str, MANOParams], # {'left': MANOParams, 'right': MANOParams}
    viseq : ArtiDataset, 
    nviseq : ArtiDataset,
    gts, 
    parts_tfs : list[list[TransformParams]], # [part_cnt, F] of TransformParams
    parts : list[ArtiPart],
    sides,
    contacts,
    args,
    writer=None
):
    device = DEVICE
    K = nviseq.K[0]
    w2c = nviseq.w2c[0]
    F = nviseq.frame_cnt
    H, W = nviseq.img_h, nviseq.img_w

    # setup residual param
    torch.set_grad_enabled(True)
    res_h : dict[str, TransformParams] = {}
    for s in sides:
        res_h[s] = TransformParams(device=device, fit_scale=False, train=True)
        res_h[s].freeze()
        hp[s].hand_pose.requires_grad = False
    res_objscale = TransformParams(device=device, fit_scale=True, train=True)
    res_objscale.freeze_rot()
    res_objscale.freeze_transl()
    
    ## FIXME: override LR and steps in config file
    _O_LR = 0.01
    _O_FLR = 0.0005
    _O_STEPS = 800
    args.ho_align.lr = _O_LR
    args.ho_align.final_lr = _O_FLR
    args.ho_align.steps = _O_STEPS
    
    opt_gr1 = []
    for s in sides:
        opt_gr1.append(res_h[s].translation)
        # opt_tensors.append(hp[s].hand_pose)
    opt_gr1.append(res_objscale.scale)
    opt_gr2 = []
    for s in sides:
        opt_gr2.append(hp[s].hand_pose)
        opt_gr2.append(hp[s].transl)
    
    group_settings = [
        {'params': opt_gr1, 'lr': 0.01, 'final_lr': 0.0005},  # group 0
        {'params': opt_gr2, 'lr': 0.001, 'final_lr': 0.0001},  # group 1
    ]
    optimizer = torch.optim.Adam(
        group_settings,
    )
    final_lrs = [cfg['final_lr'] for cfg in group_settings]
    scheduler = make_linear_decay(optimizer, _O_STEPS, final_lrs)

    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor= args.ho_align.final_lr / args.ho_align.lr,
    #     total_iters=int(0.75 * args.ho_align.steps)
    # )
    optimizing_info = optimizer.param_groups[0]['params']
    loguru.info(f'Optimizing params: {optimizing_info}')
    
    loss_enabled = ['projection_error', 'contact_dis', 'reg_pose']
    for step in (pbar := tqdm(range(args.ho_align.steps))):
        optimizer.zero_grad()
        preds = {}
        
        # deform part verts
        v3d_o = []
        v2d_o = []
        for i in range(args.part_cnt):
            v3d = []
            v2d = []
            for f in range(F):
                vp = parts_tfs[i][f].forward(parts[i].verts)
                vp = parts[i].apply_denorm(vp)
                vp = vp * res_objscale.scale
                v3d.append(vp)
                v2d.append(project_points(vp, K, w2c))
            v3d_o.append(torch.stack(v3d, dim=0))
            v2d_o.append(torch.stack(v2d, dim=0))
            preds[f'part_{i}'] = {
                "v2d": v2d_o[i],
                "v3d": v3d_o[i],
            }
        
        preds['obj'] = {
            "v2d": torch.cat(v2d_o, dim=1),  # [F, PN, 3]
            "v3d": torch.cat(v3d_o, dim=1),  # [F, PN, 3]
        }
        
        # deform hand verts
        for s in sides:
            hv3d_p = []
            hv2d_p = []
            hj3d_p = []
            hj2d_p = []
            hand_dict = hp[s].forward()
            for f in range(F):
                hx = hand_dict['v3d'][f]  # [778, 3]
                jx = hand_dict['j3d'][f] # [21, 3]
                hx[..., 2] += res_h[s].translation[2]
                jx[..., 2] += res_h[s].translation[2]
                # hj = res_h[s].forward(hj3d_gt[s][f])
                hv3d_p.append(hx)
                hv2d_p.append(project_points(hx, K, w2c))
                hj3d_p.append(jx)
                hj2d_p.append(project_points(jx, K, w2c))
            preds[s] = {
                "v2d": torch.stack(hv2d_p, dim=0),
                "v3d": torch.stack(hv3d_p, dim=0),
                "j3d": torch.stack(hj3d_p, dim=0),
                "j2d": torch.stack(hj2d_p, dim=0),
            }
        
        loss = compute_loss(
            preds, gts, 
            sides,
            contacts,
            res_h,
            hp,
            enabled=loss_enabled,
            step=step, device=device, args=args, writer=writer,
        )
        loss.backward()
        for s in hp.keys():
            if hp[s].transl.grad is not None:
                mask = torch.zeros_like(hp[s].transl.grad)
                mask[..., 2] = 1.0
                # mask[f] = 1.0
                hp[s].transl.grad = hp[s].transl.grad * mask
        
        optimizer.step()
        scheduler.step()
        
        curr_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(
            f"Step {step+1}/{args.ho_align.steps} | Loss={loss.item():.6f} | LR={curr_lr:.5f}"
        )
        writer.add_scalar(f'vars/parts_scale', res_objscale.scale.item(), step)
        for s in sides:
            writer.add_scalar(f'vars/{s}_transl', res_h[s].translation.norm().item(), step)
            writer.add_scalar(f'vars/{s}_transZ', res_h[s].translation[2].item(), step)
        
        # learn object scale for first 150 steps
        # if step == args.ho_align.steps * 0.3 or step == args.ho_align.steps * 0.6:
        if step == 150 or step == 400:
            hp_copy = deepcopy(hp)
            partsc = deepcopy(parts)
            for s in sides:
                hp_copy[s].transl.data += res_h[s].translation.data
            for i in range(args.part_cnt):
                demat = partsc[i].denorm_mat
                demat[:3, :3] *= res_objscale.scale
                demat[:3, 3] *= res_objscale.scale
                partsc[i].denorm_mat = demat
            model_io.dump_ho_data(
                hp_copy, parts_tfs, partsc, args,
                out_iter=step+1,
            )
            if step == 150:
                res_objscale.freeze()
                for s in sides:
                    res_h[s].defrost_transl()
            if step == 400:
                for s in sides:
                    res_h[s].freeze()
                    hp[s].hand_pose.requires_grad = True
                    hp[s].transl.requires_grad = True
                loss_enabled.append('accel')
                loss_enabled.append('reg_transl')
                if args.hard_velo:
                    loss_enabled.append('velo')
                    loss_enabled.remove('reg_transl')
        
    torch.save(
        {
            'res_h': {s: rh.state_dict() for s, rh in res_h.items()},
            'res_objscale': res_objscale.state_dict(),
        }, 
        f"{args.output_path}/residuals.pth"
    )
    
    for s in sides:
        hp[s].transl.data += res_h[s].translation.data
    for i in range(args.part_cnt):
        demat = parts[i].denorm_mat
        demat[:3, :3] *= res_objscale.scale
        demat[:3, 3] *= res_objscale.scale
        parts[i].denorm_mat = demat
    
    with torch.no_grad():
        model_io.dump_ho_data(
            hp, parts_tfs, parts, args, out_iter=args.ho_align.steps
        )
    return hp, parts_tfs, parts

def main():
    parser = argparse.ArgumentParser(description='Hand Object Alignment')
    parser.add_argument('--seq-path', type=str, required=True)
    parser.add_argument('--fit-ckpt', type=str, required=True)
    parser.add_argument('--conf', '--f', type=str, default='./conf/ho.yaml',)
    parser.add_argument('--output-path', type=str, 
                        default=f'./outputs/{datetime.now().strftime("%m%d/%H%M%S")}/')
    parser.add_argument('--hand-prior', type=str, default='wilor_af')  # wilor_af or hamer
    parser.add_argument('--initial-dump', action='store_true',)
    parser.add_argument('--dump-final', action='store_true',)
    parser.add_argument('--hard-velo', action='store_true',)
    args = parser.parse_args()
    if not args.output_path.startswith('./outputs/'):
        args.output_path = f'{args.output_path}/{args.seq_path.split("/")[-1]}'
    os.makedirs(args.output_path, exist_ok=True)
    args = edict(vars(args))
    conf = edict(OmegaConf.load(args.conf))
    args.update(conf)
    tb_log_dir = f"{args.output_path}/tensorboard"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    
    viseq, nviseq, parts, parts_tfs = model_io.load_parts_data(args)
    F = viseq.frame_cnt
    K = viseq.K[0]
    w2c = viseq.w2c[0]
    H, W = viseq.img_h, viseq.img_w
    
    # Load hand-object contact and side information
    with open(f'{args.seq_path}/processed/ho_contact.json', 'r') as f:
        contact_data = json.load(f)
    appeared = contact_data['appeared']
    con_frames_min = min([d['frame'] for d in contact_data['contacts']])
    contact_l, contact_r = [False for i in range(F)], [False for i in range(F)]
    finger_l = [[] for i in range(F)]
    finger_r = [[] for i in range(F)]
    missing = [i for i in range(F)]
    for d in contact_data['contacts']:
        f = d['frame'] - con_frames_min
        contact_l[f] = d['l_contact']
        contact_r[f] = d['r_contact']
        finger_l[f] = d['l_fingers']
        finger_r[f] = d['r_fingers']
        missing.remove(f)
    
    contacts = {
        'left': contact_l,
        'right': contact_r,
        'left_fingers': finger_l,
        'right_fingers': finger_r,
    }
    
    mano_params = np.load(
        f'{args.seq_path}/processed/{args.hand_prior}/manoparam_fit.slerp.npy', 
        allow_pickle=True
    ).item()
    hand_params : dict[str, MANOParams] = {}
    hand_avails = {}
    for s in ALL_SIDES:
        if s in appeared:
            h_param = mano_params[s]
            hand_avails[s] = h_param["is_valid"].astype(np.int8)
            ## solve mean pose
            h_param['betas'] = np.mean(h_param["betas"], axis=0)[None, :].repeat(F, axis=0)
            hp = MANOParams(
                h_param,
                hand_type=s,
                K=K, w2c=w2c,
                device=DEVICE
            )
            hp.freeze_global()
            hand_params[s] = hp
    
    # ================= Freeze all motion/time relavant params =================
    for i in range(args.part_cnt):
        for f in range(F):
            parts_tfs[i][f].freeze()
    
    if args.initial_dump:
        vis_composite.render_train_views(
            hand_params, 
            parts_tfs, parts,
            nviseq,
            args,
            out_label='initial_dump',
        )
    
    # ===fine-level adjustments===
    gts = {}
    for s in appeared:
        with torch.no_grad():  # Detach GT data from computation graph
            hdata = hand_params[s].forward()
            gts[s] = {
                "v2d": hdata['v2d'].detach(),
                "v3d": hdata['v3d'].detach(),
                "hand_pose": hand_params[s].hand_pose.clone().detach(),
                "transl": hand_params[s].transl.clone().detach(),
                "j3d": hdata['j3d'].detach(),
                "j2d": hdata['j2d'].detach(),
            }
    for i in range(args.part_cnt):
        assert parts[i].scale.requires_grad == False, "Part scale should not be optimized"
        v3d = []
        v2d = []
        with torch.no_grad():  # Detach part GT data from computation graph
            for f in range(F):
                vp = parts_tfs[i][f].forward(parts[i].verts)
                vp = parts[i].apply_denorm(vp)
                v3d.append(vp)
                v2d.append(project_points(vp, K, w2c))
            gts[f'part_{i}'] = {
                "v2d": torch.stack(v2d, dim=0).detach(),
                "v3d": torch.stack(v3d, dim=0).detach(),
            }

    hp, parts_tfs, parts = refined_opt(
        hand_params, 
        viseq, nviseq,
        gts,
        parts_tfs, parts,
        sides=appeared,
        contacts=contacts,
        args=args,
        writer=writer
    )
    
    if args.dump_final:
        vis_composite.render_train_views(
            hp, 
            parts_tfs, parts,
            nviseq,
            args,
            out_label='final_opt',
        )
    

if __name__ == '__main__':
    main()