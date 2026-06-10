import os
from PIL import Image
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger as loguru
import math
from arrgh import arrgh

WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANO_DIR=f'{WS_ROOT}/third_party/body_models'
DEVICE = torch.device("cuda")


from utils.data_util import to_tensor
from hrse.dataclass.primitives import ArtiPart, TransformParams
from hrse.dataclass.dataset import ArtiDataset
import hrse.utils.tfs.transform_3d as tf_utils

ATYPE = {
    'r': 1,
    'revolute': 1,
    'p': 2,
    'prismatic': 2,
    'f': 3,
    'free': 3,
    'root': 4,
}

class ArtiPartXt():
    def __init__(
            self, 
            verts, faces, colors, 
            articulate_type, frame_cnt,
            parent_part, world_tfs,
            device, pt3d_coord=True, name=None):
        self.device = device
        self.pt3d_coord = pt3d_coord
        self.name = name
        self.F = frame_cnt
        if verts is not None:
            self.verts = to_tensor(verts, device=device)
        if faces is not None:
            self.faces = to_tensor(faces, device=device)
        if colors is not None:
            self.colors = to_tensor(colors, device=device)
        self.norm_mat = torch.eye(4).to(device)
        self.denorm_mat = torch.eye(4).to(device)
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False).to(device) # TODO
        
        self.atype = articulate_type
        self.parent_part = parent_part
        if ATYPE[self.atype] != ATYPE['root']:
            self.calc_part_tf(world_tfs=world_tfs, calc_rel=True)
        else:
            self.calc_part_tf(world_tfs=world_tfs, calc_rel=False)

    def calc_part_tf(self, world_tfs : torch.Tensor, calc_rel=False):
        world_tfs = world_tfs.to(self.device)
        self.w_tfs = world_tfs
        if calc_rel:
            assert hasattr(self, 'parent_part') and hasattr(self.parent_part, 'w_tfs')
            p_w_tfs = self.parent_part.w_tfs
            assert p_w_tfs.shape == self.w_tfs.shape and p_w_tfs.shape == (self.F, 4, 4)
            rel_w_tfs = torch.bmm(self.w_tfs, tf_utils.homo_mat_inverse(p_w_tfs))
            self.rel_w_tfs = rel_w_tfs
            if ATYPE[self.atype] != ATYPE['free']:
                self.a_tfs = OneDoFTransform(
                    articulate_type=self.atype,
                    frame_cnt=self.F,
                    device=self.device,
                    train=True,
                )
                self.a_tfs.init_from_tfs(self.rel_w_tfs)
        else:
            self.rel_w_tfs = self.w_tfs
        loguru.debug(f'Initialized articulate parameters for {self.atype}')
    
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
        
        assert torch.allclose(verts_raw, self.apply_denorm(self.verts))
    
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
        vmask = self.vertex_masks if hasattr(self, 'vertex_masks') else None
        return {
            'verts': self.verts,
            'faces': self.faces,
            'colors': self.colors,
            'scale': self.scale.item(),
            'norm_mat': self.norm_mat,
            'denorm_mat': self.denorm_mat,
            'atype': self.atype,
        }
    
    def load_dict(self, data):
        self.verts = to_tensor(data['verts'], device=self.device)
        self.faces = to_tensor(data['faces'], device=self.device, dtype=torch.long)
        self.colors = to_tensor(data['colors'], device=self.device)
        self.scale = torch.nn.Parameter(torch.tensor(data['scale'], device=self.device), requires_grad=False)
        self.norm_mat = to_tensor(data['norm_mat'], device=self.device)
        self.denorm_mat = to_tensor(data['denorm_mat'], device=self.device)
        self.atype = data['atype']


class OneDoFTransform(nn.Module):
    def __init__(self, articulate_type, frame_cnt, device, train=True):
        super().__init__()
        # init params
        self.device = device
        # articulation type between this part and its parent part
        self.art_type = articulate_type
        self.F = frame_cnt
        # axis and center in object coordinate system
        self.l = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0], device=device), requires_grad=train)
        self.m = torch.nn.Parameter(torch.zeros(3, device=device), requires_grad=train)
        # per-frame motion, t(theta) for rotation angle and d(dist) for translation distance
        self.t = torch.nn.Parameter(torch.zeros((self.F), device=device), requires_grad=train)
        self.d = torch.nn.Parameter(torch.zeros((self.F), device=device), requires_grad=train)
        
        self.to(device)
        if not train:
            self.eval()
        else:
            self.train()
    
    def forward(self, verts) -> torch.Tensor:
        if verts.dim() == 2: # (NV, 3)
            vh = F.pad(verts, (0, 1), value=1.0)  # (NV, 4)
            vh = vh.unsqueeze(0).repeat(self.F, 1, 1)  # (F, NV, 4)
        elif verts.dim() == 3: # (F, NV, 3)
            vh = F.pad(verts, (0, 1), value=1.0)  # (F, NV, 4)
        else:
            raise ValueError("verts shape error: {}".format(verts.shape))
        vp = self.get_transformation_matrix() @ vh.reshape(-1, 4)
        return vp[:, :3].reshape(self.F, -1, 3)
    
    def get_transformation_matrix(self, ) -> torch.Tensor:
        l_frames = self.l.reshape(1, 3).repeat(self.F, 1)
        m_frames = self.m.reshape(1, 3).repeat(self.F, 1)
        log_transform = tf_utils.screw_param_to_exponential_coordinates(
            l_frames, m_frames, self.t, self.d
        )
        tf = tf_utils.transform_from_exponential_coordinates(log_transform)
        assert tf.shape == (self.F, 4, 4), "wrong transformation matrix shape: {}".format(tf.shape)
        return tf
    
    def init_from_tfs(self, trans_list):
        """
        trans_list: (F, 4, 4)
        """
        assert trans_list.dim() == 3 and trans_list.shape[1:] == (4, 4)
        dq = tf_utils.transform_to_dq(trans_list)
        s_axis, moment, theta, dist = tf_utils.dq_to_screw(dq)
        s_axis, moment, theta, dist = s_axis.reshape(-1, 3), moment.reshape(-1, 3), theta.reshape(-1), dist.reshape(-1)
        # compute mean screw param
        mean_axis, mean_moment = compute_mean_screw_param(
            s_axis, moment, theta, dist
        )
        assert mean_axis.shape == (3,) and mean_moment.shape == (3,) and \
              theta.shape == (self.F,) and dist.shape == (self.F,)
        # set type
        if ATYPE[self.art_type] == ATYPE['revolute']:
            self.raw_dist = dist.detach().clone()
            dist = dist.fill_(1e-6)
        if ATYPE[self.art_type] == ATYPE['prismatic']:
            self.raw_theta = theta.detach().clone()
            theta = theta.fill_(1e-6)
            
        self.l = torch.nn.Parameter(mean_axis, requires_grad=True)
        self.m = torch.nn.Parameter(mean_moment, requires_grad=True)
        self.t = torch.nn.Parameter(theta, requires_grad=True)
        self.d = torch.nn.Parameter(dist, requires_grad=True)
            
    
    def get_param(self):
        return {
            'axis': self.l.detach().clone(),
            'moment': self.m.detach().clone(),
            'theta': self.t.detach().clone(),
            'dist': self.d.detach().clone(),
        }
    
    def set_param(self, param_dict):
        grad = self.training
        if 'axis' in param_dict:
            axis = to_tensor(param_dict['axis'], device=self.device).view(-1)
            if axis.numel() != 3:
                raise ValueError(f"axis must have 3 elements, got {axis.shape}")
            self.l = torch.nn.Parameter(axis[:3], requires_grad=grad)
        if 'moment' in param_dict:
            moment = to_tensor(param_dict['moment'], device=self.device).view(-1)
            if moment.numel() != 3:
                raise ValueError(f"moment must have 3 elements, got {moment.shape}")
            self.m = torch.nn.Parameter(moment[:3], requires_grad=grad)
        if 'theta' in param_dict:
            theta = to_tensor(param_dict['theta'], device=self.device).view(-1)
            if theta.numel() != self.F:
                raise ValueError(f"theta must have {self.F} elements, got {theta.shape}")
            self.t = torch.nn.Parameter(theta[:self.F], requires_grad=grad)
        if 'dist' in param_dict:
            dist = to_tensor(param_dict['dist'], device=self.device).view(-1)
            if dist.numel() != self.F:
                raise ValueError(f"dist must have {self.F} elements, got {dist.shape}")
            self.d = torch.nn.Parameter(dist[:self.F], requires_grad=grad)

    def freeze(self):
        self.eval()
        for p in [self.l, self.m, self.t, self.d]:
            p.requires_grad = False
    
    def defrost(self):
        self.train()
        for p in [self.l, self.m, self.t, self.d]:
            p.requires_grad = True

def compute_mean_screw_param(s_axis, moment, theta, distance, eps_tol=1e-5):
    # s_axis, moment: (F, 3), theta, distance: (F, 1)
    # NOTE check identity case, where s_axis could be random
    # eps_tol: eps tolerance, should *larger* than eps in [tf]dual quat calculation: eps
    # return: mean_axis, mean_moment: (3,)
    assert s_axis.dim() == 2 and moment.dim() == 2
    F = s_axis.shape[0]
    no_rot = torch.logical_or(theta.abs() <= eps_tol, (theta - math.pi).abs() <= eps_tol)
    no_trans = distance <= eps_tol
    unit_transform = torch.logical_and(no_rot, no_trans)
    
    if torch.all(unit_transform).item():
        mean_axis = s_axis.mean(dim=0)
        mean_moment = moment.mean(dim=0)
    else:
        mask = ~unit_transform
        mean_axis = s_axis[mask].mean(dim=0)
        mean_moment = moment[mask].mean(dim=0)
    return mean_axis, mean_moment