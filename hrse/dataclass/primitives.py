import os
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger as loguru
import trimesh
import open3d as o3d
import smplx
from arrgh import arrgh

import hrse.utils.tfs.transform_3d as tf
from utils.data_util import to_tensor
from hrse.utils.image_utils import project_points

WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE = torch.device("cuda")
MANO_DIR=f'{WS_ROOT}/third_party/body_models'

HAND_TIPS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}
WRIST_FIXES = [38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234, 92]
JNTS_MANO_TO_OURS = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TF_MANO_TO_OURS = [0, 13, 14, 15, 0, 1, 2, 3, 0, 4, 5, 6, 0, 10, 11, 12, 0, 7, 8, 9]  ## LBS weights

class ArtiPart():
    def __init__(self, verts, faces, colors, device, 
                 pt3d_coord=True, name=None):
        self.device = device
        self.pt3d_coord = pt3d_coord
        self.name = name
        if verts is not None:
            self.verts = to_tensor(verts, device=device)
        if faces is not None:
            self.faces = to_tensor(faces, device=device)
        if colors is not None:
            self.colors = to_tensor(colors, device=device)
        self.norm_mat = torch.eye(4).to(device)
        self.denorm_mat = torch.eye(4).to(device)
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False).to(device)
        self.subparts = None
    
    def load_from_mesh(self, mesh_path):
        file_ext = os.path.splitext(mesh_path)[1].lower()
        
        if file_ext == '.glb':
            # Load GLB scene and extract mesh with material baking
            scene = trimesh.load(mesh_path)
            
            # Extract the first mesh from the scene
            if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
                mesh_name = list(scene.geometry.keys())[0]
                mesh_c = scene.geometry[mesh_name]
            else:
                raise ValueError(f"No mesh found in GLB file: {mesh_path}")
            
            self.verts = to_tensor(mesh_c.vertices, device=self.device)
            self.faces = to_tensor(mesh_c.faces, device=self.device)
            
            # Bake material to vertex colors if available
            if hasattr(mesh_c.visual, 'material') and mesh_c.visual.material is not None:
                # Try to get vertex colors from material
                try:
                    # If the mesh has UV coordinates and material has texture
                    if hasattr(mesh_c.visual, 'uv') and hasattr(mesh_c.visual.material, 'image'):
                        # Bake texture to vertex colors
                        vertex_colors = mesh_c.visual.to_color().vertex_colors
                        self.colors = to_tensor(vertex_colors[:, :3], device=self.device)
                    else:
                        # Use material base color if no texture
                        if hasattr(mesh_c.visual.material, 'baseColorFactor'):
                            base_color = mesh_c.visual.material.baseColorFactor[:3]
                            self.colors = torch.ones_like(self.verts, device=self.device) * torch.tensor(base_color, device=self.device)
                        else:
                            # Fallback to default color
                            self.colors = torch.ones_like(self.verts, device=self.device) * 0.8
                except Exception as e:
                    loguru.warning(f"Failed to bake material from GLB {mesh_path}: {e}, using default color.")
                    self.colors = torch.ones_like(self.verts, device=self.device) * 0.8
            else:
                loguru.warning(f"No material found in GLB {mesh_path}, using default color.")
                self.colors = torch.ones_like(self.verts, device=self.device) * 0.8
        else:
            mesh_c = trimesh.load_mesh(mesh_path, process=False)
            self.verts = to_tensor(mesh_c.vertices, device=self.device)
            self.faces = to_tensor(mesh_c.faces, device=self.device)
            # trimesh stores vertex colors in a VertexColor object, access via .data
            if hasattr(mesh_c.visual, 'vertex_colors'):
                self.colors = to_tensor(mesh_c.visual.vertex_colors[:, :3], device=self.device) # take only RGB, ignore A if present
            else: # if no vertex colors, create a default white color
                loguru.warning(f"No vertex colors found in mesh {mesh_path}, using default color.")
                self.colors = torch.ones_like(self.verts, device=self.device) * 0.8
        
        if self.colors.max() > 1.0:
            self.colors = self.colors / 255.0
        self.verts = self.verts.float()
        self.faces = self.faces.long()
        self.colors = self.colors.float()
    
    def cano_normalization(self):
        verts_raw = self.verts.clone()
        bbox_max = torch.max(self.verts, dim=0)[0]
        bbox_min = torch.min(self.verts, dim=0)[0]
        center = (bbox_max + bbox_min) * 0.5

        radius = torch.norm(self.verts - center, dim=1).max()
        
        # mat below describes the transform from scaled & center to unscaled and not centered 
        denormalize_mat = torch.diag(torch.tensor([radius, radius, radius, 1.0], device=self.device))
        denormalize_mat[:3, 3] = center

        normalize_mat = torch.inverse(denormalize_mat)
        pts_cano = torch.ones((self.verts.shape[0], 4), device=self.device)
        pts_cano[:, :3] = self.verts
        pts_cano = (normalize_mat @ pts_cano.T).T
        pts_cano = pts_cano[:, :3] / pts_cano[:, 3:]
        
        self.denorm_mat = denormalize_mat
        self.norm_mat = normalize_mat
        self.verts = pts_cano
        
        assert torch.allclose(verts_raw, self.apply_denorm(self.verts), atol=1e-5)
    
    def apply_norm(self, verts):
        pts = verts * self.scale
        pts = F.pad(pts, (0, 1), value=1.0)
        return (self.norm_mat @ pts.T).T[:, :3]
    
    def apply_denorm(self, verts):
        pts = verts * self.scale
        pts = F.pad(pts, (0, 1), value=1.0)
        return (self.denorm_mat @ pts.T).T[:, :3]
    
    def defrost_scale(self):
        self.scale.requires_grad = True
    def freeze_scale(self):
        self.scale.requires_grad = False
    
    def dump_dict(self):
        dino_feats = self.dino_feats if hasattr(self, 'dino_feats') else None
        dino_validity = self.dino_validity if hasattr(self, 'dino_validity') else None
        vmask = self.vertex_masks if hasattr(self, 'vertex_masks') else None
        return {
            'verts': self.verts,
            'faces': self.faces,
            'colors': self.colors,
            'scale': self.scale.item(),
            'norm_mat': self.norm_mat,
            'denorm_mat': self.denorm_mat,
            'name': self.name,
            'vertex_masks': vmask,
            'dino_feats': dino_feats,
            'dino_validity': dino_validity,
        }
    
    def load_dict(self, data):
        self.verts = to_tensor(data['verts'], device=self.device)
        self.faces = to_tensor(data['faces'], device=self.device, dtype=torch.long)
        self.colors = to_tensor(data['colors'], device=self.device)
        self.scale = torch.nn.Parameter(torch.tensor(data['scale'], device=self.device), requires_grad=False)
        self.norm_mat = to_tensor(data['norm_mat'], device=self.device)
        self.denorm_mat = to_tensor(data['denorm_mat'], device=self.device)
        if 'name' in data.keys():
            self.name = data['name']
        if 'vertex_masks' in data.keys():
            self.vertex_masks = to_tensor(data['vertex_masks'], device=self.device)
        if 'dino_feats' in data.keys():
            self.dino_feats = to_tensor(data['dino_feats'], device=self.device)
        if 'dino_validity' in data.keys():
            self.dino_validity = to_tensor(data['dino_validity'], device=self.device)
    
    def visualize(self, name='mesh'):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.verts.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.cpu().numpy())
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy())
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([mesh], window_name=name)


class TransformParams(nn.Module):
    def __init__(self, device, fit_scale=False, train=True):
        super().__init__()
        self.fit_scale = fit_scale
        if fit_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = torch.ones(1).to(device)
        # self.rotation = nn.Parameter(torch.zeros(3))  # euler angles
        sixd_identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        self.rotation = nn.Parameter(sixd_identity)
        self.translation = nn.Parameter(torch.zeros(3))
        self.device = device
        self.to(device)
        if not train:
            self.eval()
            self.freeze()
        else:
            self.train()
            self.defrost()
    
    def forward(self, verts : torch.Tensor) \
        -> torch.Tensor:
        vp = verts * self.scale
        if self.rotation.shape[0] == 3:
            R = tf.axis_angle_to_matrix(self.rotation)
        else:
            R = tf.cont_6d_to_rmat(self.rotation.unsqueeze(0)).squeeze(0)
        # assert R.shape == (3, 3)
        vp = torch.matmul(vp, R.T)
        vp = vp + self.translation
        return vp
    
    def backward(self, verts : torch.Tensor) -> torch.Tensor:
        if self.rotation.shape[0] == 3:
            R = tf.axis_angle_to_matrix(self.rotation)
        else:
            R = tf.cont_6d_to_rmat(self.rotation.unsqueeze(0)).squeeze(0)
        # assert R.shape == (3, 3)
        Rinv = torch.inverse(R)
        vp = verts - self.translation
        vp = torch.matmul(vp, Rinv.T)
        vp = vp / self.scale
        return vp
    
    def get_transformation_matrix(self, ) -> torch.Tensor:
        if self.rotation.shape[0] == 3:
            R = tf.axis_angle_to_matrix(self.rotation)
        else:
            R = tf.cont_6d_to_rmat(self.rotation.unsqueeze(0)).squeeze(0)
        transform = torch.eye(4).to(self.device)
        transform[:3, :3] = R.detach().clone() * self.scale.item()
        transform[:3, 3] = self.translation.detach().clone()
        return transform
    
    def get_param(self):
        return {
            'scale': self.scale.item(),
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist()
        }
    
    def set_param(self, param_dict):
        if self.fit_scale:
            self.scale = nn.Parameter(torch.tensor(param_dict['scale'], device=self.device))
        else:
            if 'scale' in param_dict.keys():
                self.scale = torch.tensor(param_dict['scale'], device=self.device)
            else:
                self.scale = torch.ones(1, device=self.device)
        self.rotation = nn.Parameter(torch.tensor(param_dict['rotation'], device=self.device))
        self.translation = nn.Parameter(torch.tensor(param_dict['translation'], device=self.device))
    
    def freeze(self):
        self.scale.requires_grad = False
        self.rotation.requires_grad = False
        self.translation.requires_grad = False
    
    def defrost(self):
        if self.fit_scale:
            self.scale.requires_grad = True
        else:
            self.scale.requires_grad = False
        self.rotation.requires_grad = True
        self.translation.requires_grad = True
    
    def freeze_transl(self):
        self.translation.requires_grad = False
    def defrost_transl(self):
        self.translation.requires_grad = True
    def freeze_rot(self):
        self.rotation.requires_grad = False
    def defrost_rot(self):
        self.rotation.requires_grad = True
    
    def print_requires_grad(self):
        loguru.debug("requires_grad status:")
        for param_name, param in self.named_parameters():
            loguru.debug(f"\t{param_name}: {param.requires_grad}")
    
    def print_params(self):
        ret = f"s: {self.scale.item():.2f}"
        ret += f"r: [{', '.join([f'{x:.2f}' for x in self.rotation.tolist()])}]"
        ret += f"t: [{', '.join([f'{x:.2f}' for x in self.translation.tolist()])}]"
        return ret

class MANOParams(nn.Module):
    """
    N frames mano params
    """
    def __init__(self, mano_params, hand_type,
                 K, w2c, 
                 device, pt3d_coord=True):
        super().__init__()
        self.device = device
        self.betas = nn.Parameter(torch.tensor(mano_params['betas'], dtype=torch.float32, device=self.device))
        self.transl = nn.Parameter(torch.tensor(mano_params['transl'], dtype=torch.float32, device=self.device))
        self.global_orient = nn.Parameter(torch.tensor(mano_params['global_orient'], dtype=torch.float32, device=self.device))
        self.hand_pose = nn.Parameter(torch.tensor(mano_params['hand_pose'], dtype=torch.float32, device=self.device))
        
        # init mano layer (->device)
        self.mano_layer = smplx.create(
            model_path=(f'{MANO_DIR}/MANO_RIGHT.pkl' 
                        if hand_type == 'right' else 
                        f'{MANO_DIR}/MANO_LEFT.pkl'),
            mano_type='mano',
            use_pca=False,
            is_right=(hand_type == 'right'),
        ).to(self.device)
        self.wrist_fix = []
        for i in range(1, len(WRIST_FIXES) - 1):
            self.wrist_fix.append(np.array([WRIST_FIXES[0], WRIST_FIXES[i], WRIST_FIXES[i+1]]))
        self.wrist_fix = torch.tensor(np.asanyarray(self.wrist_fix), device=self.device)
        self.faces = torch.tensor(self.mano_layer.faces.astype(np.int32), device=self.device)
        self.faces = torch.cat([self.faces, self.wrist_fix], dim=0)
        
        self.K = torch.tensor(K, device=self.device)
        self.w2c = torch.tensor(w2c, device=self.device)
        
        self.pt3d_coord = pt3d_coord
        # x,y flipped
        self.pt3d_reverse = torch.tensor([-1.0, -1.0, 1.0], device=self.device)
        self.hand_pose.requires_grad = False
        self.betas.requires_grad = False
    
    def unfreeze_global(self):
        self.global_orient.requires_grad = True
        self.transl.requires_grad = True
        self.hand_pose.requires_grad = True
        self.betas.requires_grad = True
    
    def freeze_global(self):
        self.global_orient.requires_grad = False
        self.transl.requires_grad = False
        self.hand_pose.requires_grad = False
        self.betas.requires_grad = False
    
    def freeze_orient(self):
        self.global_orient.requires_grad = False
    
    def unfreeze_orient(self):
        self.global_orient.requires_grad = True
    
    def freeze_transl(self):
        self.transl.requires_grad = False
    
    def unfreeze_transl(self):
        self.transl.requires_grad = True
    
    def get_i_dict(self, i=None):
        if i is None:
            return {
                'betas': self.betas,
                'hand_pose': self.hand_pose,
                'transl': self.transl,
                'global_orient': self.global_orient,
            }
        elif isinstance(i, list):
            return {
                'betas': self.betas[i],
                'hand_pose': self.hand_pose[i],
                'transl': self.transl[i],
                'global_orient': self.global_orient[i],
            }
        elif isinstance(i, int):
            return {
                'betas': self.betas[i].unsqueeze(0),
                'hand_pose': self.hand_pose[i].unsqueeze(0),
                'global_orient': self.global_orient[i].unsqueeze(0),
                'transl': self.transl[i].unsqueeze(0)
            }
        raise ValueError(f"Invalid index type: {type(i)}")
    
    def forward(self, idx=None):
        d = self.get_i_dict(idx)
        out = self.mano_layer(
            betas = d['betas'],
            transl = d['transl'],
            global_orient = d['global_orient'],
            hand_pose = d['hand_pose'],
        )
        
        j3d = out.joints
        v3d = out.vertices
        if self.pt3d_coord:
            v3d = v3d * self.pt3d_reverse
            j3d = j3d * self.pt3d_reverse
        
        j2d = project_points(
            j3d,
            self.K,
            self.w2c,
        )
        v2d = project_points(
            v3d,
            self.K,
            self.w2c,
        )
        rd = {
            'j2d': j2d,
            'v2d': v2d,
            'j3d': j3d,
            'v3d': v3d,
        }
        rd.update(d)
        return rd