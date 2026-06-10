import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as loguru

from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    SoftPhongShader,
    DirectionalLights,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

RENDER_MAX_DEPTH = 50.0

class PhongRenderer:
    def __init__(self, 
                 K: torch.Tensor,  # [4, 4] camera intrinsics
                 render_size: tuple,  # (height, width)
                 camera_override=None,
                 device=torch.device("cuda")):
        self.device = device
        self.img_h, self.img_w = render_size
        
        # create camera
        if camera_override is not None:
            self.cameras = camera_override
        else:
            self.cameras = create_camera(K.clone(), None, render_size, device)
        
        # create rasterizer
        self.raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
        )
    
        self.lights = DirectionalLights(
            device=device,
            ambient_color=((0.7, 0.7, 0.7),),   # lower ambient light
            diffuse_color=((0.5, 0.5, 0.5),),   # increase diffuse intensity
            specular_color=((0.2, 0.2, 0.2),),  # add slight specular highlight
            direction=((1, -1, -0.5),),         # light from top-right
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=self.lights,
                device=device,
            )
        )
    
    def set_camera(self, camera):
        self.cameras = camera
        self.renderer.rasterizer.cameras = camera
    
    def render_mesh(self, mesh: Meshes):
        fragments = self.renderer.rasterizer(mesh)
        images = self.renderer.shader(fragments, mesh)
        zbuf = fragments.zbuf[0, ..., 0]  # [H, W]
        
        rgb = images[0, ..., :3]  # [H, W, 3]
        alpha = images[0, ..., 3:]  # [H, W, 1]
        mask = (alpha > 0).float().squeeze(-1)  # [H, W]
        zbuf[zbuf < 0] = RENDER_MAX_DEPTH
        
        return {
            "mask": mask, 
            "rgb": rgb,
            "alpha": alpha,  # [H, W, 1]
            "depth": zbuf,  # [H, W]
        }


class PhongRendererSoft:
    def __init__(
            self, 
            K: torch.Tensor,  # [4, 4] camera intrinsics
            render_size: tuple,  # (height, width)
            soft_gamma=1e-4,
            soft_sigma=1e-4,
            soft_faces_per_pixel=64,
            camera_override=None,
            device=torch.device("cuda")
        ):
        """
        Adjust soft rasterization parameters to enhance/weaken boundary gradients:
        - sigma controls boundary smoothness (larger is smoother)
        - faces_per_pixel controls candidate triangles per pixel (larger is more stable, more memory) 
        - gamma controls "sharpness" of depth soft selection (smaller is more biased towards nearest face)
        """
        self.device = device
        self.img_h, self.img_w = render_size
        
        # create camera
        if camera_override is not None:
            self.cameras = camera_override
        else:
            self.cameras = create_camera(K.clone(), None, render_size, device)
        
        # create rasterizer
        self.raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
        )
    
        self.lights = DirectionalLights(
            device=device,
            ambient_color=((0.7, 0.7, 0.7),), 
            diffuse_color=((0.5, 0.5, 0.5),), 
            specular_color=((0.2, 0.2, 0.2),),
            direction=((1, -1, -0.5),),       
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=self.lights,
                device=device,
            )
        )
        self.set_softness(
            sigma=soft_sigma,
            faces_per_pixel=soft_faces_per_pixel,
            gamma=soft_gamma,
        )
    
    # === Soft rasterizer for differentiable silhouettes (occlusion-aware) ===
    def set_softness(self, sigma: float = 1e-4, faces_per_pixel: int = 64, gamma: float = 1e-4):
        self.soft_blend_params = BlendParams(sigma=sigma, gamma=gamma)
        self.soft_raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=50000,
            # cull_backfaces=False,
            # clip_barycentric_coords=True,
            # max_faces_per_bin=0,
            # bin_size=0,  # NOTE slow but stable
        )
        self.soft_rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.soft_raster_settings
        )
    
    def set_camera(self, camera):
        self.cameras = camera
        self.renderer.rasterizer.cameras = camera
    
    def render_mesh_semmask(self, mesh: Meshes, face_offsets, target=None):
        fragments_hard = self.renderer.rasterizer(mesh)
        images_hard = self.renderer.shader(fragments_hard, mesh)
        zbuf = fragments_hard.zbuf[0, ..., 0]   # [H, W]
        rgb = images_hard[0, ..., :3]           # [H, W, 3]
        alpha_hard = images_hard[0, ..., 3:]    # [H, W, 1]
        mask = (alpha_hard > 0).float().squeeze(-1)
        zbuf = torch.where(zbuf < 0, torch.full_like(zbuf, RENDER_MAX_DEPTH), zbuf)
        face_offsets_t = torch.tensor(face_offsets, device=zbuf.device)
        pix_to_face_1 = fragments_hard.pix_to_face[0, ..., 0]  # [H, W]
        semmask = torch.bucketize(pix_to_face_1, face_offsets_t[1:], right=False)
        semmask = torch.where(pix_to_face_1 >= 0, semmask, -1)

        fragments = self.soft_rasterizer(mesh)             # [H, W, K]
        faces_k = fragments.pix_to_face[0]                 # [H, W, K]
        zbuf_k = fragments.zbuf[0]                         # [H, W, K]
        dists_k = fragments.dists[0]                       # [H, W, K]
        valid_k = (faces_k >= 0)

        sigma = self.soft_blend_params.sigma + 1e-12
        c_k = torch.sigmoid(-dists_k / sigma) * valid_k.float()   # [H, W, K]

        z_fill = torch.full_like(zbuf_k, 1e6)
        z_for_softmax = torch.where(valid_k, zbuf_k, z_fill)
        z_min = z_for_softmax.amin(dim=-1, keepdim=True)
        gamma = self.soft_blend_params.gamma + 1e-12
        w_depth = torch.softmax(-(z_for_softmax - z_min) / gamma, dim=-1)  # [H, W, K]

        w_all = w_depth * c_k  # [H, W, K]

        alpha = 1.0 - torch.exp(
            torch.clamp(torch.log(torch.clamp(1.0 - w_all, min=1e-6)).sum(dim=-1), min=-50.0)
        )  # [H, W]

        P = len(face_offsets_t) - 1
        faces_k_clamped = faces_k.clamp_min(0)
        part_ids_k = torch.bucketize(faces_k_clamped, face_offsets_t[1:], right=False)  # [H, W, K]
        one_hot = F.one_hot(part_ids_k, num_classes=P).float() * valid_k.unsqueeze(-1).float()  # [H, W, K, P]

        w_kp = w_all.unsqueeze(-1) * one_hot                        # [H, W, K, P]
        soft_parts = 1.0 - torch.exp(
            torch.clamp(
                torch.log(torch.clamp(1.0 - w_kp, min=1e-6)).sum(dim=-2),
                min=-50.0
            )
        )  # [H, W, P]
        # [H, W, P] -> [P, H, W]
        soft_parts = soft_parts.permute(2, 0, 1)
        
        denom = w_all.sum(dim=-1, keepdim=True)                           # [H, W, 1]
        soft_depth_raw = (w_all * zbuf_k).sum(dim=-1, keepdim=False)      # [H, W]
        soft_depth = soft_depth_raw / (denom.squeeze(-1) + 1e-8)          # [H, W]
        soft_depth = torch.where(
            denom.squeeze(-1) > 0, soft_depth,
            torch.full_like(soft_depth, RENDER_MAX_DEPTH)
        )

        if target is not None:
            soft_target = soft_parts[int(target)]  # [H, W]
        else:
            soft_target = alpha 
        
        return {
            "rgb": rgb,
            "depth": zbuf,              # hard depth
            "mask": mask,               # hard mask
            "semmask": semmask,         # hard semantics
            "alpha": alpha,             # [H, W] differentiable alpha
            "soft_target": soft_target, # [H, W] differentiable (target only, occlusion-aware)
            "soft_parts": soft_parts,   # [P, H, W] differentiable (all parts, occlusion-aware)
            "soft_depth": soft_depth,   # [H, W] differentiable depth
        }


def create_camera(K, w2c, imsize, device):
    """
    Create a PyTorch3D camera from intrinsic matrix K and world-to-camera transform w2c.
    
    Args:
        K: Camera intrinsic matrix of shape [3, 3] or [1, 3, 3]
        w2c: World-to-camera transformation matrix of shape [4, 4] or None
        imsize: Image size as (height, width) or ((height, width),)
        device: Torch device
        
    Returns:
        cameras: PyTorch3D PerspectiveCameras object
    """
    if w2c is not None:
        cam_R = w2c[:3, :3].unsqueeze(0).to(device)
        cam_T = w2c[:3, 3].unsqueeze(0).to(device)
    else:
        cam_R = torch.eye(3).to(device).unsqueeze(0)
        cam_T = torch.zeros((1, 3)).to(device).float()
    
    if len(K.shape) == 3 and K.shape[0] == 1:
        K = K.squeeze(0)
    
    fx, fy, px, py = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((px, py),),
        R=cam_R,
        T=cam_T,
        in_ndc=False,  # important!!
        image_size=(imsize,),
        device=device,
    )

    return cameras

def create_camera_rect(w2c, device):
    cam_R = w2c[:3, :3].unsqueeze(0).to(device)
    cam_T = w2c[:3, 3].unsqueeze(0).to(device)
    cameras = FoVPerspectiveCameras(
        R=cam_R,
        T=cam_T,
        device=device,
    )
    return cameras

def create_meshes(v3d, faces, color, device):
    verts = v3d  # 1, N, 3
    verts_rgb = torch.Tensor(color).float().expand(verts.shape[0], verts.shape[1], -1)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    batch_size = verts.shape[0]
    meshes = Meshes(
        verts=[vv for vv in verts],
        faces=[faces for _ in range(batch_size)],
        textures=textures,
    )
    return meshes

def create_meshes_glb():
    pass

def create_meshes_batch(verts_list, faces_list, colors_list, device):
    """
    args:
    - verts_list: vert list with B elements, each of shape [N_i, 3]
    - faces_list: face list with B elements, each of shape [F_i, 3]
    - colors_list: color list with B elements, each can be a single RGB value 
                   or a tensor matching the corresponding vertices
    returns:
    - meshes: PyTorch3D Meshes object containing a batch of B meshes
    """
    verts_rgb_list = []
    for i, (verts, color) in enumerate(zip(verts_list, colors_list)):
        if color.dim() == 1 or (color.dim() == 2 and color.shape[0] == 1):
            verts_rgb = color.view(1, -1).expand(verts.shape[0], -1).to(device)
        elif color.shape[0] == verts.shape[0]:
            verts_rgb = color.to(device)
        else:
            raise ValueError(f"Invalid color shape: {color.shape}, expected [N_i, 3] or (3)")
        verts_rgb_list.append(verts_rgb)
    
    textures = TexturesVertex(verts_features=verts_rgb_list)
    meshes = Meshes(
        verts=verts_list,
        faces=faces_list,
        textures=textures,
    )
    return meshes

def create_meshes_merge(verts_list, faces_list, colors_list, device):
    """
    args:
    - verts_list: vert list with B elements, each of shape [N_i, 3]
    - faces_list: face list with B elements, each of shape [F_i, 3]
    - colors_list: color list with B elements, each can be a single RGB value 
                   or a tensor matching the corresponding vertices
    - device: Torch device
    
    returns:
    - meshes: PyTorch3D Meshes object containing a merged mesh
    """
    merged_verts = []
    merged_faces = []
    merged_colors = []
    vert_offset = 0
    
    for i, (verts, faces, color) in enumerate(zip(verts_list, faces_list, colors_list)):
        verts = verts.to(device)
        faces = faces.to(device)
        merged_verts.append(verts)
        
        adjusted_faces = faces + vert_offset
        merged_faces.append(adjusted_faces)
        
        if not isinstance(color, torch.Tensor):
            color = torch.tensor(color, dtype=torch.float32)
        if color.dim() == 1 or (color.dim() == 2 and color.shape[0] == 1):
            verts_rgb = color.view(1, -1).expand(verts.shape[0], -1).to(device)
        elif color.shape[0] == verts.shape[0]:
            verts_rgb = color.to(device)
        else:
            raise ValueError(f"Invalid color shape: {color.shape}, expected [N_i, 3] or (3)")
        
        merged_colors.append(verts_rgb)
        
        vert_offset += verts.shape[0]
    
    merged_verts = torch.cat(merged_verts, dim=0)
    merged_faces = torch.cat(merged_faces, dim=0)
    merged_colors = torch.cat(merged_colors, dim=0)
    
    textures = TexturesVertex(verts_features=merged_colors.unsqueeze(0))
    meshes = Meshes(
        verts=[merged_verts],
        faces=[merged_faces],
        textures=textures,
    )
    return meshes

def vis_semmask(semmask, return_pil=True):
    """
    semmask: [H, W], -1 is background, (0,1,...) are mesh indices based on input offset
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    semmask_np = semmask.cpu().numpy()
    num_classes = int(semmask_np.max()) + 1
    cmap = plt.get_cmap('tab20', num_classes)
    colored_mask = np.zeros((*semmask_np.shape, 3), dtype=np.float32)
    
    for i in range(num_classes):
        colored_mask[semmask_np == i] = cmap(i)[:3]  # Ignore alpha channel
    colored_mask[semmask_np == -1] = [0.5, 0.5, 0.5]  # Background to gray
    colored_mask = (colored_mask * 255).astype(np.uint8)
    
    mask_vis = Image.fromarray(colored_mask)
    if return_pil:
        return mask_vis
    else:
        mask_vis.show()

def vis_part_semmask(soft_parts: torch.Tensor, return_pil: bool = True, bg_thresh: float = 0.1):
    import matplotlib.pyplot as plt
    from PIL import Image
    assert soft_parts.dim() == 3, f"soft_parts shape should be [P, H, W], got {soft_parts.shape}"
    sp = soft_parts.detach().cpu()
    maxv, argmax = torch.max(sp, dim=0)  # [H, W]
    part_map = argmax.numpy().astype(np.int32)
    part_map[maxv.numpy() < bg_thresh] = -1

    P = sp.shape[0]
    cmap = plt.get_cmap('tab20b', P)
    colored = np.zeros((*part_map.shape, 3), dtype=np.float32)
    for i in range(P):
        colored[part_map == i] = cmap(i)[:3]
    colored[part_map == -1] = [0.3, 0.3, 0.3]  # Background gray
    colored = (colored * 255).astype(np.uint8)

    img = Image.fromarray(colored)
    if return_pil:
        return img
    else:
        img.show()