import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from arrgh import arrgh
from loguru import logger as loguru
from datetime import datetime

DEVICE = torch.device("cuda")

# hack numpy for SMPLX
np.bool = np.bool_
np.int = np.int32

import hrse.utils.render as render
from hrse.utils import image_utils
import hrse.utils.plot.vis_plotter as plotter
from utils.data_util import to_tensor, to_numpy
from hrse.dataclass.primitives import ArtiDataset, TransformParams, MANOParams, ArtiPart

def hand_z_depth_align_rough(
    hands : dict[str, MANOParams], # {'left': MANOParams, 'right': MANOParams}
    v2d, 
    depths,
    masks_h,
    seq : ArtiDataset,
    args,
    front_side='right',
):
    """
    Sample keypoints from depth so that the hand is roughly aligned with the object in depth.
    """
    assert len(v2d) == len(depths) and len(v2d) == len(masks_h)
    assert len(v2d) == seq.frame_cnt
    v2d = v2d.clone().detach()
    
    front_param = hands[front_side]
    mano_zs = (front_param.transl.clone().detach())[:, 2]
    sampled_avg = torch.zeros(seq.frame_cnt).to(v2d.device)
    valid_frames = []
    
    # sample depths from v2d coord
    for _i in tqdm(range(seq.frame_cnt)):
        ds = []
        for _pt in v2d[_i]:
            x, y = int(_pt[0]), int(_pt[1])
            if (0 <= x < depths[_i].shape[1] and 
                0 <= y < depths[_i].shape[0] and
                masks_h[_i][y, x]):
                ds.append(depths[_i][y, x])
        if len(ds) > 0:
            ds = torch.tensor(ds).to(v2d.device)
            sampled_avg[_i] = torch.mean(ds)
            valid_frames.append(_i)
    mean = torch.mean(sampled_avg)
    std = torch.std(sampled_avg)
    loguru.debug(f'align mean: {mean}, std: {std}')
    
    # calc diff
    mano_zs, sampled_avg = mano_zs[valid_frames], sampled_avg[valid_frames]
    diff = torch.mean(mano_zs - sampled_avg)
    diff_std = torch.std(mano_zs - sampled_avg)
    loguru.debug(f'align diff: {diff}, diff_std: {diff_std}')
    
    for hp in hands.values():
        hp.transl[:, 2] -= diff
    return hands


def render_occupancy(
    renderer : render.PhongRendererSoft,
    hand_meshes : dict[str, render.Meshes], # {'left': Meshes, 'right': Meshes}
    obj_mesh : render.Meshes,
    seq : ArtiDataset, 
    f,
    args, dump=None,
):
    meshes = [obj_mesh]
    if 'left' in hand_meshes:
        meshes.append(hand_meshes['left'])
    if 'right' in hand_meshes:
        meshes.append(hand_meshes['right'])
    scene_mesh = render.join_meshes_as_scene(meshes)
    face_counts = [len(m.faces_list()[0]) for m in meshes]
    face_offsets = torch.tensor([0] + list(
        torch.cumsum(torch.tensor(face_counts), dim=0)
    ), device=renderer.device)
    
    rend = renderer.render_mesh_semmask(
        scene_mesh, face_offsets=face_offsets, target=0
    )
    if dump is not None:
        os.makedirs(dump, exist_ok=True)
        all_semmask_vis = render.vis_part_semmask(
            rend['soft_parts'],
            return_pil=True,
        )
        soft_hand_mask = (rend['soft_parts'][1:].sum(dim=0) > 0.01)
        plotted = plotter.plot_depth_masks(
            seq.depths[f],
            seq.masks['human'][f],
            rend['soft_depth'],
            soft_hand_mask,
            ret_pic=True
        )
        rgb_plotted = plotter.plot_rgb_masks(
            seq.rgbs[f], seq.masks['human'][f],
            rend['rgb'], rend['mask'],
            ret_pic=True
        )
        soft_depth_colored = image_utils.colorize_depth(
            to_numpy(rend['soft_depth']),
            valid_mask=(rend['soft_depth'] < render.RENDER_MAX_DEPTH).cpu().numpy()
        )
        sd_colored_pil = Image.fromarray((soft_depth_colored * 255).astype(np.uint8))
        plotted.save(os.path.join(dump, f'hand_semmask_{f:04d}.png'))
        rgb_plotted.save(os.path.join(dump, f'rgb_rendered_{f:04d}.png'))
        all_semmask_vis.save(os.path.join(dump, f'all_semmask_{f:04d}.png'))
        sd_colored_pil.save(os.path.join(dump, f'all_soft_depth_{f:04d}.png'))
    
    return rend

def deform_HO(
    hp : dict[str, MANOParams], # {'left': MANOParams, 'right': MANOParams}
    parts_tfs : list[list[TransformParams]], # [part_cnt, F] of TransformParams
    parts : list[ArtiPart],
    ids : list[int],
    seq,
    g_objtf : TransformParams | None = None,
    g_handtf : dict[str, TransformParams] | None = None, 
    device=DEVICE,
    return_meshes=False
):
    """
    g_tf: global adjustments in world coord
    """
    K, w2c = seq.K[0], seq.w2c[0]
    ret = {}
    v3d_o = []
    v2d_o = []
    for i in range(len(parts)):
        v3d = []
        v2d = []
        for f in ids:
            vp = parts_tfs[i][f].forward(parts[i].verts)
            vp = parts[i].apply_denorm(vp)
            if g_objtf is not None:
                vp = g_objtf.forward(vp)
                # vp = vp * res_objscale.scale
            v3d.append(vp)
            v2d.append(image_utils.project_points(vp, K, w2c))
        v3d_o.append(torch.stack(v3d, dim=0))
        v2d_o.append(torch.stack(v2d, dim=0))
        ret[f'part_{i}'] = {
            "v2d": v2d_o[i],
            "v3d": v3d_o[i],
        }
    ret['obj'] = {
        "v2d": torch.cat(v2d_o, dim=1),  # [F, sum(PN), 3]
        "v3d": torch.cat(v3d_o, dim=1),  # [F, sum(PN), 3]
    }
    for s in hp.keys():
        hand_dict = hp[s].forward(ids)
        hand_dict['hand_pose'] = hp[s].hand_pose[ids].clone()
        hand_dict['transl'] = hp[s].transl[ids].clone()
        hand_dict['rotation'] = hp[s].global_orient[ids].clone()
        if g_handtf is not None and s in g_handtf:
            hand_dict['v3d'] = g_handtf[s].forward(hand_dict['v3d'])
            hand_dict['j3d'] = g_handtf[s].forward(hand_dict['j3d'])
            hand_dict['v2d'] = image_utils.project_points(hand_dict['v3d'], K, w2c)
            hand_dict['j2d'] = image_utils.project_points(hand_dict['j3d'], K, w2c)
        ret[s] = hand_dict
    ## debug
    # rand_f = np.random.choice(len(ids), 1)[0]
    # sampled_vs = ret['obj']['v2d'][rand_f][::10].cpu()
    # arrgh(sampled_vs)
    # kp_on_rgb_pil = plotter.plot_kp_onto_image(
    #     sampled_vs,
    #     to_numpy(nviseq.rgbs[rand_f]),
    #     radius=2,
    #     return_pil=True,
    # )
    # kp_on_rgb_pil.show()
    # input(f'press enter to continue...')
    if return_meshes:
        raise NotImplementedError
    return ret