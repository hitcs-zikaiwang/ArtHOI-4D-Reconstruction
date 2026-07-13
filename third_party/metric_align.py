import argparse
import numpy as np
import os
import torch
from natsort import natsorted
import pickle
import json
from tqdm import tqdm
import imageio as iio

# load UniDepth-V2
from unidepth.models import UniDepthV2
CKPT_V2 = f"lpiccinelli/unidepth-v2-vitl14"  # ViT-L14 is about ~1.5GiB

def run_model_inference(
        img_dir: str, 
        depth_dir: str, 
        intrins_file: str, 
        save_depth_np: bool = False
):
    img_files = natsorted(os.listdir(img_dir))
    if not intrins_file.endswith(".json"):
        intrins_file = f"{intrins_file}.json"

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(os.path.dirname(intrins_file), exist_ok=True)

    model = UniDepthV2.from_pretrained(CKPT_V2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.resolution_level = 9  # we want better result rather than speed
    print(f"Running on {img_dir} with {len(img_files)} images")
    
    intrins_dict = {}
    intrins = [] # in frame natsort order
    metric_depths = []
    for idx, img_file in enumerate(bar := tqdm(img_files)):
        img_name = os.path.splitext(img_file)[0]
        img = iio.imread(f"{img_dir}/{img_file}")
        
        pred_dict = run_model(model, img)
        
        # pred metric depth
        depth = pred_dict["depth"]
        disp = 1.0 / np.clip(depth, a_min=1e-6, a_max=1e6)  # clip to a reasonable range
        bar.set_description(f"Input {img_file} {depth.min()} {depth.max()}")
        if save_depth_np:
            out_path = f"{depth_dir}/{img_name}.npy"
            np.save(out_path.replace("png", "npy"), disp.squeeze())
        
        # pred intrinsics
        K = pred_dict["intrinsics"]
        intrins_dict[img_name] = (
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )
        
        _K = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]])
        intrins.append(_K)
        metric_depths.append(depth)
    
    with open(intrins_file, "w") as f:
        json.dump(intrins_dict, f, indent=1)
    return metric_depths, intrins


def run_model(model, rgb: np.ndarray, intrinsics: np.ndarray | None = None):
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = None
    if intrinsics is not None:
        intrinsics_torch = torch.from_numpy(intrinsics)

    predictions = model.infer(rgb_torch, intrinsics_torch)
    out_dict = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}
    return out_dict


def align_firstframe(depths, metric_depths, args):
    frames_cnt = depths.shape[0]
    
    mono_disp_map = depths[0]
    metric_disp_map = 1.0 / metric_depths[0]
    mid_metric_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
    mid_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8
    scale = np.median(mid_metric_disp / mid_mono_disp)
    shift = np.median(metric_disp_map - scale * mono_disp_map)
    
    ret_dep = []
    for f in range(frames_cnt):
        mono_disp_map = depths[f]
        aligned_disp = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        max_depth = 1 / np.quantile(aligned_disp, 0.01)
        # set depth values that are too small to invalid (0)
        aligned_disp[aligned_disp < min_thre] = 0.0
        
        aligned_metric = 1.0 / np.clip(aligned_disp, a_min=1e-6, a_max=1e6)
        aligned_metric[aligned_disp < min_thre] = max_depth
        ret_dep.append(aligned_metric)
    
    return ret_dep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--raw', type=str, required=True)
    parser.add_argument('--raw-type', type=str, default='vda', help='vda or rolling')
    parser.add_argument('--rgb', type=str, required=True)
    parser.add_argument('--output', '--o', type=str, required=True)
    args = parser.parse_args()
    
    if args.raw_type == 'vda':
        _f = np.load(args.raw)
        depths = _f['depths']
    else:
        # add your own depth model here
        raise ValueError(f"Unknown raw type {args.raw_type}")
    
    height = depths.shape[1]
    width = depths.shape[2]
    frames_cnt = depths.shape[0]
    print(f'Loaded depth maps with shape {depths.shape}, height {height}, width {width}')
    
    # calc UnidepthV2 metric depths
    metric_depths, intrins = run_model_inference(
        args.rgb, 
        f"{args.output}/depths/", 
        f"{args.output}/intrinsics.json",
        # save_depth_np=True,
    )
    metric_depths = np.asanyarray(metric_depths)
    intrins = np.asanyarray(intrins)
    
    # check data sanity
    assert len(metric_depths) == frames_cnt
    assert len(intrins) == frames_cnt
    assert metric_depths[0].shape == (height, width)
    
    aligned_depths = align_firstframe(depths, metric_depths, args)
    
    # save to pkl
    os.makedirs(args.output, exist_ok=True)
    pickle.dump(aligned_depths, open(os.path.join(
        args.output, f'metric_unidepv2_{args.raw_type}.pkl'),'wb'))

