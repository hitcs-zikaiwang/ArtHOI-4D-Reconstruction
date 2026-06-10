import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
from datetime import datetime
import pickle
from loguru import logger as loguru
import subprocess
from natsort import natsorted
import shutil
import subprocess
from arrgh import arrgh


def make_dir_structure(seq_path, extra_dirs:list[str]=None):
    """
    Create the directory structure for the sequence.
    """
    dirs_to_create = [
        'build/image',
        'build/mask',
        'build/mesh',
        'build/inpainting/image',
        'build/inpainting/mask',
        'packed',  # packed dataset for convenient loading
        'processed/wilor_af', # hand recon
        'processed/mask_raw/gt',
        'processed/mask_raw/vi',
        'processed/partseps',  # PartField separation, may change 
        'processed/vda', 
        'processed/unidepv2/gt',
        'processed/unidepv2/vi',
        'processed/cano_obj', # canonical frame object cutout after inpainting
    ]
    if extra_dirs:
        dirs_to_create.extend(extra_dirs)
    for d in dirs_to_create:
        dir_path = os.path.join(seq_path, d)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            loguru.debug(f"Created directory: {dir_path}")


def call_script_block(script_path, args_list, conda_env=None, **kwargs):
    """
    Call a script with the given arguments and wait for it to finish.
    Redirects stdout and stderr to the current stdout.
    """
    cmd = ['python', script_path] + args_list
    if conda_env:
        cmd = ['conda', 'run', '-n', conda_env, '--live-stream',
               'python', script_path] + args_list
    loguru.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        subprocess.run(
            cmd,
            check=True,
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            **kwargs
        )
        loguru.info("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        loguru.error(f"Script execution failed: {e}")


def call_script_async(script_path, args_list, conda_env=None, **kwargs):
    cmd = ['python', script_path] + args_list
    if conda_env:
        cmd = ['conda', 'run', '-n', conda_env, '--live-stream',
               'python', script_path] + args_list
    loguru.debug(f"Executing command asynchronously: {' '.join(cmd)}")
    child = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs
    )
    t_input = ''
    while t_input.strip() != 'q':
        t_input = input("input 'q' to quit async execution: ")
    child.terminate()
    loguru.info(f"Terminated async script: {script_path}")

def post_process_merge_masks(splits):
    merged = []
    mask_len = len(splits[0])
    mask_num = len(splits)
    for i in range(mask_len):
        cur = np.zeros_like(splits[0][0])
        for idx in range(mask_num):
            cur = np.logical_or(cur, splits[idx][i])
        merged.append(cur)
    return merged

def save_mask(out_dir, masks):
    os.makedirs(out_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask = mask.astype(np.uint8)
        if mask.max() < 2:
            mask = mask * 255
        mask = Image.fromarray(mask)
        mask.save(f'{out_dir}/{idx:05d}.png')
    loguru.info(f'Saved {len(masks)} masks to {out_dir}')

def post_process_sam2_masks(masks_folder, out_folder):
    mask_subs = os.listdir(masks_folder)
    parts = [_ for _ in mask_subs if _.startswith('part_')]
    
    mask_dict = {}
    for mask_name in mask_subs:
        fp = os.path.join(masks_folder, mask_name)
        mask_files = natsorted([os.path.join(fp, f) for f in os.listdir(fp) if f.endswith('.npy')])
        masks = [np.load(f, allow_pickle=True) for f in mask_files]
        mask_dict[mask_name] = masks
        save_mask(os.path.join(out_folder, mask_name), masks)
    
    # merge obj
    to_merge = [_[1] for _ in mask_dict.items() if _[0] in parts]
    merged_obj = post_process_merge_masks(to_merge)
    save_mask(os.path.join(out_folder, 'obj'), merged_obj)
    
    # merge composite
    if 'human' in mask_dict.keys():
        to_merge.append(mask_dict['human'])
        merged_composite = post_process_merge_masks(to_merge)
        save_mask(os.path.join(out_folder, 'composite'), merged_composite)

def convert_png_to_jpg(input_folder):
    has_png = any(f.lower().endswith('.png') for f in os.listdir(input_folder))
    if not has_png:
        return input_folder, False

    jpgs_folder = os.path.join(input_folder, 'jpgs')
    os.makedirs(jpgs_folder, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.lower().endswith('.png'):
            png_path = os.path.join(input_folder, file)
            jpg_path = os.path.join(jpgs_folder, os.path.splitext(file)[0] + '.jpg')
            if not os.path.exists(jpg_path):
                with Image.open(png_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG')
    
    return jpgs_folder, True


def mask_cutout(mask_path, rgb_image):
    """Crop the masked foreground from an RGB image with transparent background."""
    if isinstance(mask_path, str) or isinstance(mask_path, os.PathLike):
        mask = Image.open(mask_path).convert("L")
    else:
        mask = mask_path if isinstance(mask_path, Image.Image) else Image.fromarray(mask_path)
        mask = mask.convert("L")

    if isinstance(rgb_image, str) or isinstance(rgb_image, os.PathLike):
        image = Image.open(rgb_image).convert("RGB")
    elif isinstance(rgb_image, Image.Image):
        image = rgb_image.convert("RGB")
    else:
        image = Image.fromarray(rgb_image).convert("RGB")

    if mask.size != image.size:
        raise ValueError(f"Mask size {mask.size} does not match image size {image.size}")

    mask_array = (np.array(mask) > 127).astype(np.uint8) * 255
    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask has no foreground pixels")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width, height = mask.size
    pad_x = int(0.05 * (x_max - x_min + 1))
    pad_y = int(0.05 * (y_max - y_min + 1))
    x_min = max(x_min - pad_x, 0)
    x_max = min(x_max + pad_x, width - 1)
    y_min = max(y_min - pad_y, 0)
    y_max = min(y_max + pad_y, height - 1)

    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(Image.fromarray(mask_array))
    return image_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))


def split_video(video_path, output_dir, resolution=(960, 540)):
    """
    Split video into frames using ffmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)
    width, height = resolution
    ffmpeg_cmd = f"ffmpeg -i {video_path} -vf scale={width}:{height} "\
                 f"-start_number 0 {output_dir}/%05d.png"
    loguru.info(f"Executing: {ffmpeg_cmd}")
    os.system(ffmpeg_cmd)
    loguru.info(f"Frames extracted to {output_dir}")


def get_video_resolution(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        width, height = map(int, output.split(','))
        return (width, height)
    except subprocess.CalledProcessError as e:
        loguru.error(f"Failed to get video resolution: {e}")
        return None

def synth_video(img_dir, out_video_name, fps=30, target_resolution=(960, 540), clip=None):
    """
    Synthesize video from images using ffmpeg.
    """
    os.makedirs(os.path.dirname(out_video_name), exist_ok=True)
    
    working_folder, need_cleanup = convert_png_to_jpg(img_dir)
    
    image_files = []
    for file in os.listdir(working_folder):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(os.path.join(working_folder, file))
    
    image_files = natsorted(image_files)
    if not image_files:
        print("No image files found!")
        return
    if clip is not None:
        l, r = clip
        image_files = image_files[l:r+1]
    
    with open('temp_files.txt', 'w') as f:
        for img in image_files:
            f.write(f"file '{img}'\n")
    
    if target_resolution is not None:
        width, height = target_resolution
        scale_filter = f"scale={width}:{height}"
        cmd = [
            'ffmpeg',
            '-r', str(fps),
            '-f', 'concat',
            '-safe', '0',
            '-i', 'temp_files.txt',
            '-vf', f'pad=ceil(iw/2)*2:ceil(ih/2)*2,{scale_filter}',
            '-c:v', 'libx264',
            out_video_name
        ]
    else: 
        cmd = [
            'ffmpeg',
            '-r', str(fps),
            '-f', 'concat',
            '-safe', '0',
            '-i', 'temp_files.txt',
            '-c:v', 'libx264',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            out_video_name
        ]
    
    try:
        subprocess.run(cmd, check=True)
        loguru.debug(f"Video generated: {out_video_name}")
    except subprocess.CalledProcessError as e:
        loguru.debug(f"Video failed to generate: {e}")
    finally:
        if os.path.exists('temp_files.txt'):
            os.remove('temp_files.txt')
        if need_cleanup and os.path.exists(working_folder):
            shutil.rmtree(working_folder)
            loguru.debug("Cleaned up temporary converted image files")

def resize_video(source, target, resolution=(960, 540)):
    """
    Resize video to a specific resolution using ffmpeg.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    width, height = resolution
    ffmpeg_cmd = f"ffmpeg -i {source} -vf scale={width}:{height} {target}"
    loguru.info(f"Executing: {ffmpeg_cmd}")
    os.system(ffmpeg_cmd)
    loguru.info(f"Video resized and saved to {target}")


def get_intr_depth_pro(intr_path, cutoff=None):
    """
    fomatted in K matrices
    """
    with open(intr_path, 'rb') as f:
        intr = pickle.load(f)
    if cutoff is not None:
        l, r = cutoff
        intr = intr[l:r+1]
    intr = np.round(intr, 5)
    return intr

def get_intr_unidepth(intr_path, cutoff=None):
    with open(intr_path, 'r') as f:
        intr = json.load(f)
    intr_keys = sorted(intr.keys())
    intr = [intr[k] for k in intr_keys]
    if cutoff is not None:
        l, r = cutoff
        intr = intr[l:r+1]
    return np.asanyarray(intr)

def get_extr_droidslam(extr_path, cutoff=None):
    # from shape-of-motion - DROID-SLAM
    data = np.load(extr_path, allow_pickle=True).item()
    # intr = data[]
    extr = data["traj_c2w"]
    if cutoff is not None:
        l, r = cutoff
        extr = extr[l:r+1]
    return extr

def mergeIntrExtr(
        intr, 
        extr, 
        out_path,
        intr_type='Kmat',  # Kmat or ffcc
        output_format='json',
        cutoff=None, 
        from_reso=None):
    """
    output format: json / npy
    """
    if cutoff is not None:
        l, r = cutoff
        intr = intr[l:r+1]
        extr = extr[l:r+1]
    
    # transform intr from fx, fy, cx, cy to K
    if intr_type != 'Kmat':
        intr_ = intr
        intr = []
        for i in intr_:
            fx, fy, cx, cy = i
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
            intr.append(K)
        intr = np.array(intr).astype(np.float32)
    
    if from_reso is None:
        from_reso = input('Enter the correspoding resolution (w, h): ')
        from_reso = tuple(map(int, from_reso.split(' ')))
    assert len(from_reso) == 2
    
    result = {}
    result['from_reso'] = from_reso
    result['intrinsics'] = intr
    result['extrinsics'] = extr
    loguru.debug(result)
    
    os.makedirs(out_path, exist_ok=True)
    if output_format == 'json':
        with open(f'{out_path}/camera_param.json', 'w') as f:
            json.dump(result, f)
    elif output_format == 'npy':
        np.save(f'{out_path}/camera_param.npy', result, allow_pickle=True)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

def process_camera_params(
    resolution,
    intr_path,
    extr_path=None,
    output_path='.',
):
    
    intr = get_intr_unidepth(intr_path)
    if extr_path is None:
        loguru.info('No extrinsics provided, using identity matrix')
        extr = np.eye(4).astype(np.float32)
        extr = np.repeat(extr[np.newaxis, ...], intr.shape[0], axis=0)
        assert len(extr.shape) == 3 and extr.shape[-1] == 4 and extr.shape[-2] == 4
    else:
        extr = get_extr_droidslam(extr_path)
    
    mergeIntrExtr(intr, extr, output_path, 
                  intr_type='ffcc',
                  from_reso=resolution, 
                  output_format='npy')
    # loguru.info('Done')
