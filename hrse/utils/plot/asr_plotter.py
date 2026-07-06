import numpy as np
import torch

DEVICE = torch.device("cuda")

from hrse.utils.render import (
    create_meshes, 
    create_meshes_merge, 
    PhongRenderer
)
import hrse.utils.plot.vis_plotter as plotter

def render_deform_oneframe(
    rgb, mask, K,
    mesh_v, mesh_f, mesh_c,
    tf_mat,
    save=None,
):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    H, W = rgb.shape[:2]
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).to(DEVICE).float()
    else:
        K = K.to(DEVICE).float()
    
    pt3d_flip = torch.diag(torch.tensor([-1, -1, 1], device=DEVICE)).float()

    rot = torch.tensor(tf_mat[:3, :3]).to(DEVICE).float()
    transl = torch.tensor(tf_mat[:3, 3]).to(DEVICE).float()
    rot = pt3d_flip @ rot
    transl = pt3d_flip @ transl
    
    
    verts = torch.from_numpy(mesh_v).to(DEVICE).float()
    faces = torch.from_numpy(mesh_f).to(DEVICE).long()
    colors = torch.from_numpy(mesh_c).to(DEVICE).float()
    colors = colors / 255.0 if colors.max() > 2.0 else colors
    
    vps = (rot @ verts.T).T + transl.unsqueeze(0)
    
    mesh_pt3d = create_meshes(
        vps.unsqueeze(0), faces, colors, DEVICE
    )
    renderer = PhongRenderer(K, (H, W), device=DEVICE)
    rend = renderer.render_mesh(mesh_pt3d)
    rend = {
        'mask': rend['mask'].detach().cpu().numpy(),
        'rgb': rend['rgb'].detach().cpu().numpy(),
        'depth': rend['depth'].detach().cpu().numpy(),
    }
    if save is not None:
        plotter.plot_rgb_masks(
            rgb, mask, 
            rend['rgb'], rend['mask'],
            save=save,
        )
    return rend
