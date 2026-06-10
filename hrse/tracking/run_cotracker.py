import os
import sys

import torch
from loguru import logger as loguru
from tqdm import tqdm

WS_ROOT = "/home/gnaq/dev/DMT-align"
DEVICE = torch.device("cuda")

sys.path.append(WS_ROOT)

from io_utils.data_util import to_tensor
from arti_ds import ArtiDataset
from io_utils import vis_3d_o3d

from tracking.tracking_utils import (
    lift_track_to3d_oneframe,
    plot_visibility_by_time,
    query_grid_over_mask,
    vis_2d_tracks,
)


def load_cotracker(ckpt="cotracker3_offline"):
    """Load CoTracker model"""
    loguru.info(f"Loading CoTracker model: {ckpt}")
    model = torch.hub.load("facebookresearch/co-tracker", ckpt)
    model.eval()
    loguru.success("CoTracker model loaded successfully")
    return model


def load_cotracker_online(ckpt="cotracker3_online"):
    """Load CoTracker model for online tracking"""
    loguru.info(f"Loading CoTracker model: {ckpt}")
    model = torch.hub.load("facebookresearch/co-tracker", ckpt)
    model.eval()
    loguru.success("CoTracker model loaded successfully")
    return model


def track_quries(model, video, queries, device):
    """
    video: shape of B T C H W, denormalized (by * 255)
    queries: shape of [N, 3] for N queries, MUST be float tensor.
    returns:
        pred_tracks: [B, T, N, 2] T number of frames, N number of queries
        pred_visibility: [B, T, N, 1] bool
    """
    loguru.debug(f"Tracking queries of shape {queries.shape}")
    pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
    loguru.debug(f'pred_tracks: {pred_tracks.shape}, pred_visibility: {pred_visibility.shape}')
    return pred_tracks, pred_visibility


def track_quries_online_sqt(model, video, queries, query_t, device):
    """
    Single-Query-Time tracking
    video: shape of B=1 T C H W, denormalized (by * 255)
    queries: shape of [B=1, N, 2] for N queries, MUST be float tensor.
    returns:
        pred_tracks: [T, N, 2] T number of frames, N number of queries
        pred_visibility: [T, N, 1] bool
    """
    T = video.shape[1]
    N = queries.shape[1]

    pred_tracks_fwd, pred_visibility_fwd = online_one_pass(model, video, queries)
    pred_tracks_fwd = pred_tracks_fwd[0, :, :N]
    pred_visibility_fwd = pred_visibility_fwd[0, :, :N]

    video_inv = video.flip(1)
    queries_inv = queries.clone()
    queries_inv[..., 0] = T - 1 - queries_inv[..., 0]

    pred_tracks_bwd, pred_visibility_bwd = online_one_pass(model, video_inv, queries_inv)
    pred_tracks_bwd = pred_tracks_bwd.flip(1)[0, :, :N]
    pred_visibility_bwd = pred_visibility_bwd.flip(1)[0, :, :N]

    bwd_mask = ((torch.arange(T)[:, None]).to(queries.device)) < queries[0, :, 0][None, :]
    pred_tracks = torch.where(bwd_mask[..., None], pred_tracks_bwd, pred_tracks_fwd)
    pred_visibility = torch.where(bwd_mask, pred_visibility_bwd, pred_visibility_fwd)

    assert pred_tracks.shape == (T, N, 2)
    assert pred_visibility.shape == (T, N)
    return pred_tracks, pred_visibility


def online_one_pass(model, video, queries):
    """
    model, query and video should be on the same (cuda) device
    video: shape of B T C H W, denormalized (by * 255)
    queries: shape of [N, 2] for N queries, MUST be float tensor.
    """
    T = video.shape[1]
    first_flag = True
    for i in tqdm(range(T)):
        if i % model.step == 0 and i > 0:
            video_chunk = video[:, max(0, i - model.step * 2) : i]
            pred_tracks, pred_visibility = model(
                video_chunk,
                is_first_step=first_flag,
                queries=queries,
            )
            first_flag = False
    pred_tracks, pred_visibility = model(
        video[:, -(i % model.step) - model.step - 1 :],
        False,
        queries=queries,
    )
    torch.cuda.empty_cache()
    return pred_tracks, pred_visibility


cotrackerv3 = load_cotracker_online().cuda()


def track_frames_quries(
        frames, quries,
        query_time,
        device=DEVICE,
):
    """
    frames: shape of [T, H, W, 3]
    quries: shape of [N, 2] for N query.
    query_time: int, the frame time of query.
    
    return: results of [T, N, *]:
      pred_tracks: [T, N, 2]
      pred_visibility: [T, N, 1]
    """
    video = frames.permute(0, 3, 1, 2)[None].float().to(device)
    if video.max() <= 1.0:
        video = video * 255
    quries = torch.cat([
        torch.full((quries.shape[0], 1), query_time, device=device),
        quries,
    ], dim=-1)
    quries = quries.unsqueeze(0)
    quries = to_tensor(quries, device=device, dtype=torch.float32)
    pred_tracks, pred_visibility = track_quries_online_sqt(
        cotrackerv3,
        video,
        quries,
        query_time,
        device=device,
    )
    return pred_tracks, pred_visibility


def track3d_on_seq_oneframe(
    seq: ArtiDataset,
    mask_name,
    query_time,
    query_points,
    device=DEVICE,
    args=None,
):
    """
    :param query_time: int
    :param query_points: [N, 2]
    
    return:
        tracks_3d: [N, T, 3]
        track_colors: [N, 3]
        visibles: [N, T]
        invisibles: [N, T]
        # confidences: [N, T]
    """
    rgbs = torch.stack(seq.rgbs, dim=0).to(device)
    masks = torch.stack(seq.masks[mask_name], dim=0).to(device)
    depths = torch.stack(seq.depths, dim=0).to(device)
    K = seq.K.to(device)[..., :3, :3]
    extr = seq.w2c.to(device)
    query_points = to_tensor(query_points, device=device, dtype=torch.float32)

    tracks_2d, visibility_2d = track_frames_quries(
        rgbs,
        query_points,
        query_time=query_time,
        device=device,
    )

    os.makedirs(f'{args.output_path}/track/{mask_name}/', exist_ok=True)
    vis_2d_tracks(
        f'{args.output_path}/track/{mask_name}/',
        rgbs, tracks_2d[None], visibility_2d[None],
        dump_name=f'mask_grid_f{query_time:04d}',
    )
    loguru.debug(f"tracks_2d: {tracks_2d.shape}, visibility_2d: {visibility_2d.shape}")

    tracks_3d, track_colors, visibles, invisibles, confidences = (
        lift_track_to3d_oneframe(
            query_index=query_time,
            query_img=rgbs[query_time],
            tracks_2d=tracks_2d,
            visibility_2d=visibility_2d,
            depths=depths,
            masks=masks,
            Ks=K,
            c2ws=extr,
        )
    )

    plot_visibility_by_time(
        visibles,
        f'{args.output_path}/track/{mask_name}/vis_{query_time:04d}.png',
    )

    return {
        "tracks_2d": tracks_2d.swapdims(0, 1),
        "tracks_3d": tracks_3d,
        "track_colors": track_colors,
        "visibles": visibles,
        "invisibles": invisibles,
    }


def track3d_on_seq_total(seq: ArtiDataset, part_id, args, interval):
    """
    returns:
      [tracks_2d] : [N, T, 2]
      [tracks_3d] : [N, T, 3]
      [track_colors] : [N, 3]
      [visibles] : [N, T]
      [invisibles] : [N, T]
    """
    pick_frames = [i for i in range(0, seq.frame_cnt, interval)]
    pick_frames.append(seq.frame_cnt - 1)
    loguru.info(f"pick frames: {pick_frames}")

    ret_dict = None
    for fid in pick_frames:
        q_mask = seq.masks[f'part_{part_id}'][fid]

        query_points = query_grid_over_mask(
            args.per_part.tracking.grid_size, q_mask,
            erode_radius=args.per_part.tracking.mask_erode_radius,
        ).to(DEVICE)
        if query_points.shape[0] == 0:
            continue

        track3d_dict = track3d_on_seq_oneframe(
            seq,
            f'part_{part_id}',
            fid,
            query_points=query_points,
            args=args,
        )

        if ret_dict is None:
            ret_dict = track3d_dict
        else:
            for k in ret_dict.keys():
                x = torch.cat([ret_dict[k], track3d_dict[k]], dim=0)
                ret_dict[k] = x

        if args.debug and False:
            vis_3d_o3d.vis_tracks_composite(
                seq, track3d_dict, args,
            )
            breakpoint()

    if ret_dict is None:
        raise ValueError("No tracking data found, please check settings")

    plot_visibility_by_time(
        ret_dict['visibles'],
        f'{args.output_path}/track_visibles_part{part_id}.png',
    )
    for k in ret_dict.keys():
        loguru.debug(f'ret_dict[{k}] shape: {ret_dict[k].shape}')
    return ret_dict
