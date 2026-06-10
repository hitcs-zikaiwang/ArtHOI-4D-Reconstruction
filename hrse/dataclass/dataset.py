import os
from PIL import Image
import numpy as np
import torch
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

class ArtiDataset:
    def __init__(self, seq_path, 
                 use_video_inpainting=True, 
                 fix_intr=True,
                 fix_extr=True):
        self.seq_path = seq_path
        # self.mesh_dir = os.path.join(seq_path, 'build/mesh/mesh_cano.obj')
        self.npz_data_path = f"{seq_path}/packed/visuals.npz"
        # inpainting
        if use_video_inpainting:
            self.npz_data_path = f"{seq_path}/packed/visuals_inpainting.npz"
        
        self._load_data(fix_intr, fix_extr)
    
    
    def _load_data(self, fix_intr=True, fix_extr=True):
        loguru.info(f"Loading data from: {self.npz_data_path}")
        
        data = np.load(self.npz_data_path, allow_pickle=True)['data'].item()
        self.rgbs = data["rgbs"]
        self.masks = data["masks"]
        self.depths = data["depths"]
        self.K = data["K"]
        self.w2c = data["extr"]
        loguru.info(f"Loaded {len(self.rgbs)} frames of data.")
        
        # misc
        self.frame_cnt = len(self.rgbs)
        self.idxs = list(range(self.frame_cnt))
        self.img_w, self.img_h = self.rgbs[0].shape[1], self.rgbs[0].shape[0]
        self.img_size = (self.img_h, self.img_w)
        
        if fix_intr:
            K = torch.FloatTensor(self.K[0]).repeat(self.frame_cnt, 1, 1)
            self.K = K
        
        if fix_extr:
            R = torch.eye(3).repeat(self.frame_cnt, 1, 1)
            t = torch.zeros(3).repeat(self.frame_cnt, 1)
            # R|t -> c2w[4*4]
            self.w2c = torch.eye(4).repeat(self.frame_cnt, 1, 1)
            self.w2c[:, :3, :3] = R
            self.w2c[:, :3, 3] = t
        else:
            self.w2c = torch.FloatTensor(self.w2c)  # (frame_cnt, 4, 4)
        self.c2w = torch.inverse(self.w2c)
        loguru.info(f"All data processed.")
    
    def _to_device(self, device):
        loguru.debug(f"Moving data to device: {device}")
        self.K = to_tensor(self.K, device=device)
        self.w2c = to_tensor(self.w2c, device=device)
        self.c2w = to_tensor(self.c2w, device=device)
        self.rgbs = [to_tensor(x, device=device) for x in self.rgbs]
        for k in self.masks.keys():
            self.masks[k] = [to_tensor(x, device=device) for x in self.masks[k]]
        self.depths = [to_tensor(x, device=device) for x in self.depths]
        
        # self.denorm_mat = to_tensor(self.denorm_mat, device=device)
        # self.norm_mat = to_tensor(self.norm_mat, device=device)
        loguru.debug(f"All data moved to device: {device}")
    
    def get(self, idx, mask_type='composite'):
        rgb = self.rgbs[idx]
        mask = self.masks[mask_type][idx]
        depth = self.depths[idx]
        K = self.K[idx]
        w2c = self.w2c[idx]
        c2w = self.c2w[idx]
        return rgb, mask, depth, K, w2c, c2w
    