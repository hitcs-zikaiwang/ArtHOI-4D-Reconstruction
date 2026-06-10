import os
import argparse
from PIL import Image
import numpy as np
import torch
from loguru import logger as loguru
from natsort import natsorted
import glob

DEVICE = torch.device("cuda")
USE_VI_CAMERA = True

def load_masks_from_path(path, img_size):
    mask_paths = natsorted(glob.glob(f"{path}/*.png"))
    ret = []
    ret_names = []
    for p in mask_paths:
        mask = Image.open(p).convert('L')
        # scale to img_size
        mask = mask.resize(img_size, Image.BILINEAR) # reverse to w, h
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        ret.append(mask)
        ret_names.append(os.path.basename(p))
    return ret, ret_names

def load_rgbs_from_path(path, img_size):
    rgb_files = natsorted(glob.glob(path))
    rgbs = []
    for p in rgb_files:
        img = Image.open(p)
        img = img.resize(img_size, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        rgbs.append(img)
    frame_ids = []
    for i in range(len(rgbs)):
        frame_ids.append(os.path.basename(rgb_files[i]))
    return rgbs, frame_ids

def img_mask_cams_dep_frameID(rgbs, masks, K, extr, depths, frame_names, output):
    out_dict = {
        'rgbs': rgbs,
        'masks': masks,
        'K': K,
        'extr': extr,
        'depths': depths,
        'frame_ids': frame_names
    }
    out_dict = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in out_dict.items()}
    np.savez(output, data=out_dict)
    loguru.info(f"Saved to {output}")
    return out_dict

def main():
    parser = argparse.ArgumentParser(description='Hand Object Alignment')
    parser.add_argument('--seq-path', type=str, required=True)
    parser.add_argument('--part-cnt', type=int, required=True, help='articulated part numbers to optimize')
    parser.add_argument('--use_vi', type=bool, default=True, help='Load video inpainting')
    
    args = parser.parse_args()
    
    args.seq_path = f'{args.seq_path}/build'
    
    ## load Cameras (works for both vi and raw)
    if USE_VI_CAMERA:
        camera_path = os.path.join(args.seq_path, 'inpainting', 'camera_param.npy')
    else:
        camera_path = os.path.join(args.seq_path, 'camera_param.npy')
    cam_params = np.load(camera_path, allow_pickle=True).item()
    img_size = cam_params['from_reso'] # WxH
    frame_cnt = cam_params['intrinsics'].shape[0]
    K = cam_params['intrinsics']  # (frame_cnt, 3, 3)
    w2c = cam_params['extrinsics']  # (frame_cnt, 4, 4)
    
    
    ## load masks
    mask_keys = ['human', 'hand', 'composite', 'obj']
    for i in range(args.part_cnt):
        mask_keys.append(f'part_{i}')
    masks = {}
    for k in mask_keys:
        masks[k], _ = load_masks_from_path(os.path.join(args.seq_path, 'mask', k), img_size)
    ## load rgbs 
    rgb_path = f'{args.seq_path}/image/*.png'
    rgbs, frame_ids = load_rgbs_from_path(rgb_path, img_size)
    # load depth
    depth_path = os.path.join(args.seq_path, 'metric_depth.pkl')
    depths_ = np.load(depth_path, allow_pickle=True)
    depths = []
    # interpolate depths
    for i in range(len(depths_)):
        vd = torch.Tensor(depths_[i])
        if vd.shape != (img_size[1], img_size[0]):
            vd = torch.nn.functional.interpolate(
                vd.unsqueeze(0).unsqueeze(0),
                size=(img_size[1], img_size[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(dim=0).squeeze(dim=0)
        depths.append(vd.cpu().numpy())
    # pack all 
    img_mask_cams_dep_frameID(
        rgbs,
        masks,
        K,
        w2c,
        depths,
        frame_ids,
        f'{args.seq_path}/../packed/visuals.npz'
    )
    
    # vi RGB & masks
    vi_rgb_path = f'{args.seq_path}/inpainting/image/*.png'
    vi_rgbs, vframe_ids = load_rgbs_from_path(vi_rgb_path, img_size)
    vi_masks = {}
    for k in mask_keys:
        if k != 'human' and k != 'composite' and k != 'hand':
            vi_masks[k], _ = load_masks_from_path(os.path.join(args.seq_path, 'inpainting', 'mask', k), img_size)
    # vi depth
    vi_depth_path = os.path.join(args.seq_path, 'inpainting', 'metric_depth.pkl')
    vi_depths_ = np.load(vi_depth_path, allow_pickle=True)
    vi_depths = []
    # interpolate depths (if needed)
    for i in range(len(vi_depths_)):
        vd = torch.Tensor(vi_depths_[i])
        if vd.shape != (img_size[1], img_size[0]):
            vd = torch.nn.functional.interpolate(
                vd.unsqueeze(0).unsqueeze(0),
                size=(img_size[1], img_size[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(dim=0).squeeze(dim=0)
        vi_depths.append(vd.cpu().numpy())
    # save all
    img_mask_cams_dep_frameID(
        vi_rgbs,
        vi_masks,
        K,
        w2c,
        vi_depths,
        vframe_ids,
        f'{args.seq_path}/../packed/visuals_inpainting.npz'
    )

if __name__ == "__main__":
    loguru.critical(f'Using camera parameters from {"vi" if USE_VI_CAMERA else "raw"}')
    main()