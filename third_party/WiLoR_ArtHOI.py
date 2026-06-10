from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional
from tqdm import tqdm

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from wilor.utils.geometry import batch_rot2aa
from wilor.configs import get_config
from ultralytics import YOLO 
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

def focal_arthoi(path):
    cam_path = path.replace('/build/image', '/build/inpainting/camera_param.npy')
    cam_params = np.load(cam_path, allow_pickle=True).item()
    return float(cam_params['intrinsics'][0, 0, 0])

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()
    # set out_folder to ArtHOI style
    args.out_folder = args.img_folder.replace('/build/image', '/processed/wilor_af')
    wilor_dir = os.path.dirname(os.path.realpath(__file__))
    model_cfg = get_config(os.path.join(wilor_dir, 'pretrained_models/model_config.yaml'), update_cachedir=True)
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    
    # override focal in config
    # setup camera intrinsics
    im0 = cv2.imread(str(img_paths[0]))
    img_size = torch.tensor(im0.shape[:2][::-1]).float()  # (width, height)
    _c_focal = focal_arthoi(args.img_folder)
    model_cfg.defrost()
    scaled_focal_length = torch.tensor(_c_focal).float().to(device)
    model_cfg.EXTRA.FOCAL_LENGTH = _c_focal / img_size.max().item() * model_cfg.MODEL.IMAGE_SIZE
    model_cfg.MANO.DATA_DIR    = os.path.join(wilor_dir, 'mano_data')
    model_cfg.MANO.MODEL_PATH  = os.path.join(wilor_dir, 'mano_data')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(wilor_dir, 'mano_data', 'mano_mean_params.npz')
    model_cfg.freeze()
    
    print(f'Using focal length: {scaled_focal_length} px for all images.')
    # Download and load checkpoints
    model, model_cfg = load_wilor(
        checkpoint_path = os.path.join(wilor_dir, 'pretrained_models/wilor_final.ckpt'), 
        cfg_path= os.path.join(wilor_dir, 'pretrained_models/model_config.yaml'),
        override_cfg=model_cfg
    )
    detector = YOLO(os.path.join(wilor_dir, 'pretrained_models/detector.pt'))
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'vis_mesh_wilor'), exist_ok=True)
    
    # Iterate over all images in folder
    pred_list = []
    pred_mano = []
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))
        # detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        detections = detector(img_cv2, conf = 0.485, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            print(f'Skipping {img_path} as no hands detected')
            _d = {
                'img_path': str(img_path),
                'is_right': True,
                'skip': True,
            }
            pred_list.append(_d)
            pred_mano.append(_d)
            continue
        
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            # scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n] # shape (3,)
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                pred_dict = {}
                pred_dict['cam_t.full'] = cam_t
                pred_dict['verts'] = verts
                pred_dict['jts'] = joints
                pred_dict['is_right'] = is_right
                pred_dict['img_path'] = str(img_path)
                pred_list.append(pred_dict)

                # {'global_orient', 'hand_pose', 'betas'}, transl=cam_t
                mano_param = {}
                mano_param['is_right'] = is_right
                mano_param['img_path'] = str(img_path)
                mano_param['transl'] = cam_t # shape (3,)
                # betas (Bs, -1)
                mano_param['betas'] = out['pred_mano_params']['betas'][n].cpu().numpy()
                # global_orient (Bs, 1, 3, 3) -> (3,)
                mano_param['global_orient'] = batch_rot2aa(
                    out['pred_mano_params']['global_orient'][n]).cpu().numpy().squeeze(0)
                # hand_pose (Bs, 15, 3, 3) -> (45,)
                mano_param['hand_pose'] = batch_rot2aa(
                    out['pred_mano_params']['hand_pose'][n]).cpu().numpy().reshape(-1)
                pred_mano.append(mano_param)
                
                # NOTE
                # shapes = f'betas: {mano_param["betas"].shape}, global_orient: {mano_param["global_orient"].shape}, ' + \
                # f'hand_pose: {mano_param["hand_pose"].shape}, transl: {mano_param["transl"].shape}'
                # print(shapes)
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            cv2.imwrite(
                os.path.join(args.out_folder, 'vis_mesh_wilor', f'{img_fn}.jpg'), 
                255*input_img_overlay[:, :, ::-1]
            )
    
    # finally dump data
    results_3d, mano_d = reform_pred_list(pred_list, pred_mano)
    save_mano_params(args.out_folder, mano_params=mano_d, reformed_pred_list=results_3d)
    print(f'Saved results to {args.out_folder}!')

def reform_pred_list(pred_list, pred_mano):
    im_paths = sorted(list(set([pred_dict['img_path'] for pred_dict in pred_list])))
    num_frames = len(im_paths)
    verts_r = np.zeros((num_frames, 778, 3))*np.nan
    verts_l = np.copy(verts_r)
    for pred_dict in pred_list:
        if 'skip' in pred_dict.keys() and pred_dict['skip'] is True:
            continue
        is_right = bool(pred_dict['is_right'])
        v3d_cam = pred_dict['verts']  + pred_dict['cam_t.full'][None, :]
        idx = im_paths.index(pred_dict['img_path'])
        if is_right:
            verts_r[idx] = v3d_cam
        else:
            verts_l[idx] = v3d_cam
    verts_r = verts_r.astype(np.float32)
    verts_l = verts_l.astype(np.float32)
    verts_r = torch.FloatTensor(verts_r)
    verts_l = torch.FloatTensor(verts_l)
    results_3d = {}
    results_3d['v3d.right'] = verts_r
    results_3d['v3d.left'] = verts_l
    results_3d['im_paths'] = im_paths
    
    not_valid_r_idx = np.all(np.isnan(verts_r.numpy()), axis=(1,2))
    not_valid_l_idx = np.all(np.isnan(verts_l.numpy()), axis=(1,2))
    is_valid = {}
    is_valid['left'] = np.where(~not_valid_l_idx)[0]
    is_valid['right'] = np.where(~not_valid_r_idx)[0]

    # pred mano params
    SIDES = ['right', 'left']
    MANO_KEYS = {'global_orient': 3, 'hand_pose': 45, 'betas': 10, 'transl': 3}
    mano_d = {'right': {}, 'left': {}}
    for s in SIDES:
        for k, v_dim in MANO_KEYS.items():
            mano_d[s][k] = np.zeros((num_frames, v_dim))*np.nan
    for m in pred_mano:
        side = 'right' if is_right else 'left'
        idx = im_paths.index(m['img_path'])
        if 'skip' in m.keys() and m['skip'] is True:
            assert idx in (np.where(not_valid_r_idx)[0] if side=='right' else np.where(not_valid_l_idx)[0]), \
                f"Frame {m['img_path']} marked as skip but has mano params!"
            continue
        is_right = bool(m['is_right'])
        for k in MANO_KEYS.keys():
            mano_d[side][k][idx] = m[k]
        mano_d[side]['is_valid'] = is_valid[side]
    
    return results_3d, mano_d

def save_mano_params(save_path: str, mano_params: dict, reformed_pred_list=None):
    """
    dump `manoparam_fit.raw.npy` :
    {
        "left/right":
        {
            "global_orient": np.ndarray (F, 3) of axis-angle
            "hand_pose":     np.ndarray (F, 45)
            "betas":         np.ndarray (F, 10)
            "transl":        np.ndarray (F, 3)
        }
        with missing frames filled with NaN
    }
    """
    if reformed_pred_list is not None:
        np.save(f'{save_path}/v3d.npy', reformed_pred_list)
    
    SIDES = ['right', 'left']
    d = {'left': {}, 'right': {}}
    for side in SIDES:
        for key, val in mano_params[side].items():
            if isinstance(val, torch.Tensor):
                d[side][key] = val.cpu().numpy()
            else:
                d[side][key] = val

    np.save(f'{save_path}/manoparam_fit.raw.npy', d)

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()

"""
Example 
python WiLoR_ArtHOI.py --img_folder ds/rsrd_usb_plug/build/image 
"""