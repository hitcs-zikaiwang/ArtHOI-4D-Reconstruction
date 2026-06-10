import os
import numpy as np
import torch
import smplx
from tqdm import tqdm
import sys
from loguru import logger as loguru

from utils.MANO.registration import optimize_mano_shape
from utils.data_util import ld2dl
from utils.xdict import xdict
import utils.MANO.slerp as slerp


WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANO_DIR_R=f'{WS_ROOT}/third_party/body_models/MANO_RIGHT.pkl'
MANO_DIR_L=f'{WS_ROOT}/third_party/body_models/MANO_LEFT.pkl'
MODEL_DIR = "wilor_af"

mano_layers = {
    "right": smplx.create(
        model_path=MANO_DIR_R, model_type="mano", use_pca=False, is_rhand=True
    ),
    "left": smplx.create(
        model_path=MANO_DIR_L, model_type="mano", use_pca=False, is_rhand=False
    ),
}

def fit_frame(
    seq_path,
    target_v3d,
    save_mesh,
    iteration,
    is_right,
    init_params=None,
    first_frame=False,
    use_beta_loss=False,
):
    target_v3d = torch.FloatTensor(target_v3d.reshape(1, -1, 3)).cuda()

    vis_dir = f"{seq_path}/processed/mesh_fit_vis/"

    tip_sem_idx = [12, 11, 4, 5, 6]

    if first_frame:
        optim_specs = {
            "epoch_coarse": 10000,
            "epoch_fine": 10000,
            "is_right": is_right,
            "save_mesh": save_mesh,
            "criterion": torch.nn.MSELoss(reduction="none"),
            "seed": 0,
            "vis_dir": vis_dir,
            "sem_idx": tip_sem_idx,
        }
    else:
        optim_specs = {
            "epoch_coarse": 2000,
            "epoch_fine": 2000,
            "is_right": is_right,
            "save_mesh": save_mesh,
            "criterion": torch.nn.MSELoss(reduction="none"),
            "seed": 0,
            "vis_dir": vis_dir,
            "sem_idx": tip_sem_idx,
        }

    os.makedirs(optim_specs["vis_dir"], exist_ok=True)
    params = optimize_mano_shape(
        target_v3d,
        mano_layers,
        optim_specs,
        iteration,
        init_params=init_params,
        use_beta_loss=use_beta_loss,
    )
    return params


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-path", type=str, required=True)
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--no_beta_loss", action="store_false", dest="use_beta_loss")
    parser.add_argument("--hand_type", type=str, default=None)
    parser.add_argument("--interp_only", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def fit_single_hand(v3d_ra_list, seq_path, args, is_right):
    import copy

    pbar = tqdm(enumerate(v3d_ra_list), total=len(v3d_ra_list))
    prev_out = None
    out_list = []
    for iteration, v3d_ra in pbar:
        pbar.set_description(
            "Processing %s [%d/%d]" % (seq_path, iteration + 1, len(v3d_ra_list))
        )
        is_valid = np.isnan(v3d_ra).sum() == 0
        if not is_valid:
            out = {}
            out["global_orient"] = torch.zeros(1, 3).cuda() * np.nan
            out["hand_pose"] = torch.zeros(1, 45).cuda() * np.nan
            out["betas"] = torch.zeros(1, 10).cuda() * np.nan
            out["transl"] = torch.zeros(1, 3).cuda() * np.nan
        else:
            out = fit_frame(
                seq_path,
                v3d_ra,
                save_mesh=args.save_mesh,
                init_params=prev_out,
                iteration=iteration,
                is_right=is_right,
                first_frame=prev_out is None,
                use_beta_loss=args.use_beta_loss,
            )
            prev_out = copy.deepcopy(out)
        out_list.append(out)

    out_dict = ld2dl(out_list)
    out_dict = dict(
        xdict({key: torch.cat(val, axis=0) for key, val in out_dict.items()}).to_np()
    )
    return out_dict


## TODO: add marks on output format
OUTLIER_Z_MAD_MULT = 3.5  # z mean outlier detection multiplier
MIN_VALID_FOR_OUTLIER = 5  # Minimum valid frames required for outlier detection


def interpolate_hand_pose(seq_path):
    """
    Interpolates hand pose parameters for a given sequence using spherical linear interpolation (SLERP).

    Args:
    seq_path (str): The name of the sequence to process.
    """
    # Define paths for input data
    data_path = f"{seq_path}/processed/{MODEL_DIR}/manoparam_fit.raw.npy"
    data_3d_path = f"{seq_path}/processed/{MODEL_DIR}/v3d.npy"
    contact_info = f"{seq_path}/processed/ho_contact.json"
    
    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    data_3d = np.load(data_3d_path, allow_pickle=True).item()
    # if exists contact info
    if os.path.exists(contact_info):
        import json
        with open(contact_info, 'r') as f:
            contact_data = json.load(f)
        avail_sides = contact_data['appeared']
    else:
        avail_sides = data.keys()
    
    # Prepare interpolated data dictionary
    data_interp = {}
    
    num_frames = data_3d[f"v3d.left"].shape[0]
    # Process each hand
    for hand in avail_sides:
        """
        NOTE undetected (left/right) hand result on frame (i) would be `NaN` in v3d.full.npy
        """
        hand_v3d = data_3d[f"v3d.{hand}"]  # Expected shape [F, V, 3]
        if hasattr(hand_v3d, 'cpu'):
            hand_v3d_np = hand_v3d.cpu().numpy()
        else:
            hand_v3d_np = np.asarray(hand_v3d)
        raw_not_valid = np.isnan(hand_v3d_np.reshape(num_frames, -1).mean(axis=1))
        # Compute mean z for each frame for outlier detection
        mean_z = hand_v3d_np[..., 2].mean(axis=1)
        mask_valid_initial = ~raw_not_valid
        # Perform z outlier detection (requires sufficient valid frames)
        if mask_valid_initial.sum() >= MIN_VALID_FOR_OUTLIER:
            valid_z = mean_z[mask_valid_initial]
            median_z = np.median(valid_z)
            mad = np.median(np.abs(valid_z - median_z)) + 1e-6
            z_dev = np.abs(mean_z - median_z)
            outlier_z = z_dev > (OUTLIER_Z_MAD_MULT * mad)
        else:
            outlier_z = np.zeros(num_frames, dtype=bool)
        # Combine NaN and z outlier frames into a single mask
        not_valid = raw_not_valid | outlier_z
        key_frames = np.where(~not_valid)[0]
        outliers = np.where(not_valid)[0]
        loguru.debug(f"{seq_path} {hand}: total={num_frames}, raw_nan={raw_not_valid.sum()}, z_outlier={outlier_z.sum()}, kept={key_frames.shape[0]}")
        # Interpolation  check
        if key_frames.shape[0] < 2:
            loguru.warning(
                f"Not enough valid frames (after outlier removal) for {hand} hand in {seq_path}, skipping interpolation."
            )
            data_interp[hand] = data[hand]
            continue
        hand_interp = slerp.slerp_mano_params(
            outliers, num_frames, key_frames, data[hand]
        )
        hand_interp["is_valid"] = (~not_valid).astype(np.float32)
        data_interp[hand] = hand_interp

    # Define output path and save interpolated data
    out_p = data_path.replace(".raw.", ".slerp.")
    np.save(out_p, data_interp)

    # Print the location of exported files
    print(f"Interpolated data saved to {out_p}")


def main():
    args = parse_args()
    if args.model_dir is not None:
        loguru.info(f"Using model dir: {args.model_dir}")
        global MODEL_DIR
        MODEL_DIR = args.model_dir
    if args.interp_only:
        interpolate_hand_pose(args.seq_path)
        return
    
    seq_path = args.seq_path
    loguru.debug(f"Beta loss: {'on' if args.use_beta_loss else 'off'}")
    
    # ========= register MANO parameters =========
    loguru.critical(f'Fitting {seq_path}/processed/{MODEL_DIR}/v3d.npy to MANO parameters')
    data = np.load(f"{seq_path}/processed/{MODEL_DIR}/v3d.npy", allow_pickle=True).item()
    data = xdict(data).search("v3d.")

    if args.hand_type is not None:
        data = data.search(args.hand_type)

    out_dict = {}
    for key, val in data.items():
        print("Processing " + key)
        flag = key.split(".")[1]
        is_right = flag == "right"

        mydict = fit_single_hand(val, seq_path, args, is_right=is_right)
        out_dict[flag] = mydict
    out_p = f"{seq_path}/processed/{MODEL_DIR}/manoparam_fit.raw.npy"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    np.save(out_p, out_dict)
    print(f"Saved to {out_p}")
    
    # ========= interpolate hand pose =========
    interpolate_hand_pose(seq_path)


if __name__ == "__main__":
    main()

"""
python WiLoR_reg_mano.py  \
  --seq_path ds/rs_cddrive2/ \
  --use_beta_loss 
"""