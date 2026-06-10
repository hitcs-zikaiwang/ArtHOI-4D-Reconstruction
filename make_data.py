import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse
from loguru import logger as loguru
import shutil
from ruamel.yaml import YAML

import utils.file_utils as file_utils
import utils.dataset_io as dataset_io

# ============ configurate these environments =============

ROOT = os.path.dirname(os.path.abspath(__file__))
# MAIN Conda env
AHOI_ENV = "arthoi"
# Standalone envs
DIFFUERASER_ENV = 'diffueraser'
WILOR_ENV = 'hamer2'
PARTFIELD_ENV = 'partfield' # could be merged into main
PARTFIELD_ROOT = f'{ROOT}/third_party/PartField/'

# ============ configurate these script paths =============
# It's not a good practice to hardcode these paths here and call them via spawning processes
# rather than importing them as modules. But since it is hard to merge all conda environments, 
# we do this monkey patch for convenience and flexibility.
# We also choose to provide a python script instead of a shell script for similar reason.

SAM2_FILEPATH = f'{ROOT}/third_party/sam2/app_hrse.py'
DIFFUERASER_FILEPATH = f'{ROOT}/third_party/DiffuEraser/run_diffueraser.py'

VDA_FILEPATH = f'{ROOT}/third_party/Video-Depth-Anything/run.py'
UNIDEPV2_FILEPATH = f'{ROOT}/third_party/unidepth/metric_align.py'

HY3D_API_FILEPATH = f'{ROOT}/utils/hy3d/api_caller.py'

WILOR_FILEPATH = f'{ROOT}/third_party/WiLoR/WiLoR_ArtHOI.py'
WILOR_REG_FILEPATH = f'{ROOT}/utils/WiLoR_reg_mano.py'


# =================================================

def get_args():
    parser = argparse.ArgumentParser(description='Generate sequence for DMT-align')
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to the sequence directory')
    parser.add_argument('--seq-name', type=str, default=None,
                        help='Dataset/sequence directory name. Defaults to input video filename stem')
    parser.add_argument('--out', type=str, default=f'{ROOT}/ds/',
                        help='Output directory for the generated sequence')
    # articulate parameters
    parser.add_argument('--part-cnt', type=int, default=2, 
                        help='Number of parts in the sequence')
    parser.add_argument('--cano-frame', type=int, default=0,
                        help='Frame index to use as canonical frame')
    parser.add_argument('--target_reso', type=int, nargs=2, default=(960, 540), 
                        help='Target resolution for the output video (width height)')
    return parser.parse_args()
    
    
def get_seq_name(video_path, seq_name=None):
    return seq_name or os.path.splitext(os.path.basename(video_path))[0]


def ask_to_continue(prompt : str):
    loguru.critical(prompt)
    cmd = input("Press [enter] to continue, or type 's' then [enter] to skip."
                " ('exit' to terminate process): ")
    if cmd.lower() == 's':
        return True
    elif cmd.lower() == 'exit':
        exit(0)
    return False


def Step1_Process_video_and_datadir(seq_path, args, target_resolution=None):
    # Create directory structure and process input video
    skip = ask_to_continue("Stage 1: Create directory structure and process input video.")
    if not skip:
        file_utils.make_dir_structure(seq_path, extra_dirs=None)
        loguru.info(f"Target resolution: {target_resolution}")
        file_utils.split_video(
            args.video, 
            f'{seq_path}/build/image',
            resolution=target_resolution,
        )
        # re-generate video for processing
        file_utils.synth_video(
            f'{seq_path}/build/image',
            f'{seq_path}/processed/gt.mp4',
            fps=30,
            target_resolution=target_resolution,
        )

def Step2_Label_human_mask_and_inpaint(seq_path, args, target_resolution=None):
    skip = ask_to_continue("Stage 2A: Please label human masks using SAM2.")
    if not skip:
        loguru.debug("Starting SAM2 to label human masks...")
        sam2_args = ['--im_dir', f'{seq_path}/build/image',
                     '--mask_root_dir', f'{seq_path}/processed/mask_raw/gt/',]
        loguru.critical("-> http://localhost:23341/, click 'Exit' on Gradio UI to end mask processing")
        file_utils.call_script_block(
            SAM2_FILEPATH,
            args_list=sam2_args,
            conda_env=AHOI_ENV,
        )
        # merge human masks to video
        file_utils.synth_video(
            f'{seq_path}/processed/mask_raw/gt/human',
            f'{seq_path}/processed/human_mask.mp4',
            fps=30,
            target_resolution=target_resolution,
        )
    
    skip = ask_to_continue("Stage 2B: Run video inpainting using DiffuEraser. ")
    if not skip:
        loguru.info("Starting video inpainting, loading weights may take long time...")
        diffueraser_args = [
            '--iv', f'{seq_path}/processed/gt.mp4',
            '--im', f'{seq_path}/processed/human_mask.mp4',
            '--output_path', f'{seq_path}/processed/',  # would save as 'xxx/inpainted_gt.mp4'
        ]
        # Split inpainted video into frames
        file_utils.call_script_block(
            DIFFUERASER_FILEPATH,
            args_list=diffueraser_args,
            conda_env=DIFFUERASER_ENV,
        )
        file_utils.resize_video(
            source=f'{seq_path}/processed/inpainted_gt.mp4',
            target=f'{seq_path}/processed/vi.mp4',
            resolution=target_resolution,
        )
        file_utils.split_video(
            f'{seq_path}/processed/vi.mp4',
            f'{seq_path}/build/inpainting/image',
            resolution=target_resolution,
        )


def Step3_Label_part_masks(seq_path, args):
    skip = ask_to_continue("Stage 3: Please label part-level masks for gt and inpainted video using SAM2.")
    if not skip:
        loguru.info("Starting SAM2 to label masks...")
        sam2_args = ['--im_dir', f'{seq_path}/build/image',
                     '--mask_root_dir', f'{seq_path}/processed/mask_raw/gt/',]
        file_utils.call_script_block(
            SAM2_FILEPATH,
            args_list=sam2_args,
            conda_env=AHOI_ENV,
        )
        loguru.info("Starting SAM2 to label video inpainting masks...")
        sam2_args = ['--im_dir', f'{seq_path}/build/inpainting/image',
                     '--mask_root_dir', f'{seq_path}/processed/mask_raw/vi/',]
        file_utils.call_script_block(
            SAM2_FILEPATH,
            args_list=sam2_args,
            conda_env=AHOI_ENV,
        )
        # merge masks to build dir
        file_utils.post_process_sam2_masks(
            masks_folder=f'{seq_path}/processed/mask_raw/gt/',
            out_folder= f'{seq_path}/build/mask/',
        )
        file_utils.post_process_sam2_masks(
            masks_folder=f'{seq_path}/processed/mask_raw/vi/',
            out_folder= f'{seq_path}/build/inpainting/mask/',
        )

def Step4_Run_depth_estimation(seq_path, args, target_resolution=None):
    skip = ask_to_continue("Stage 4A: Run video depth estimation using Video-Depth-Anything")
    if not skip:
        loguru.info("Starting VDA depth estimation, might take a while to finish...")
        vda_args_gt = [
            '--iv', f'{seq_path}/processed/gt.mp4',
            '--output_dir', f'{seq_path}/processed/vda/', 
            '--save_npz',
        ]
        file_utils.call_script_block(
            VDA_FILEPATH,
            args_list=vda_args_gt,
            conda_env=AHOI_ENV,
        )
        vda_args_vi = [
            '--iv', f'{seq_path}/processed/vi.mp4',
            '--output_dir', f'{seq_path}/processed/vda/', 
            '--save_npz',
        ]
        file_utils.call_script_block(
            VDA_FILEPATH,
            args_list=vda_args_vi,
            conda_env=AHOI_ENV,
        )
    
    skip = ask_to_continue("Stage 4B: Align depth and camera param using Unidepthv2")
    if not skip:
        loguru.info("Starting Unidepthv2...")
        unidepv2_args = [
            '--raw', f'{seq_path}/processed/vda/gt_depths.npz',
            '--raw-type', 'vda',
            '--rgb', f'{seq_path}/build/image',
            '--o', f'{seq_path}/processed/unidepv2/gt/',
        ]
        file_utils.call_script_block(
            UNIDEPV2_FILEPATH,
            args_list=unidepv2_args,
        )
        unidepv2_args = [
            '--raw', f'{seq_path}/processed/vda/vi_depths.npz',
            '--raw-type', 'vda',
            '--rgb', f'{seq_path}/build/inpainting/image',
            '--o', f'{seq_path}/processed/unidepv2/vi/',
        ]
        file_utils.call_script_block(
            UNIDEPV2_FILEPATH,
            args_list=unidepv2_args,
        )
        file_utils.process_camera_params(
            resolution=target_resolution,
            intr_path=f'{seq_path}/processed/unidepv2/gt/intrinsics.json',
            output_path=f'{seq_path}/build/',
        )
        file_utils.process_camera_params(
            resolution=target_resolution,
            intr_path=f'{seq_path}/processed/unidepv2/vi/intrinsics.json',
            output_path=f'{seq_path}/build/inpainting/',
        )
        shutil.move(
            f'{seq_path}/processed/unidepv2/gt/metric_unidepv2_vda.pkl',
            f'{seq_path}/build/metric_depth.pkl',
        )
        shutil.move(
            f'{seq_path}/processed/unidepv2/vi/metric_unidepv2_vda.pkl',
            f'{seq_path}/build/inpainting/metric_depth.pkl',
        )


def Object_recon(seq_path, seq_name, args):
    # cutout object image from canonical frame
    cano_frame = args.cano_frame
    cutout = file_utils.mask_cutout(
        mask_path=f'{seq_path}/build/inpainting/mask/obj/{cano_frame:05d}.png',
        rgb_image=f'{seq_path}/build/inpainting/image/{cano_frame:05d}.png',
    )
    cutout.save(f'{seq_path}/processed/cano_obj/{cano_frame:05d}.png')
    
    # Get Object Mesh
    skip = ask_to_continue("Stage 5A: generate object mesh.")
    glb_path = f'{seq_path}/build/mesh/'
    if not skip:
        loguru.info("Calling Hunyuan3D 3.1, this would take long time...")
        hy3d_api_args = [
            '--im', f'{seq_path}/processed/cano_obj/{cano_frame:05d}.png',
            '--output-path', f'{glb_path}/{cano_frame:05d}.glb',
            '--model', '3.1', # use HY3D 3.1 for better quality.
        ]
        file_utils.call_script_block(
            HY3D_API_FILEPATH,
            args_list=hy3d_api_args,
        )
    # convert glb to vertcolor obj
    from utils.process_hy3d import process_glb_to_vertcolor_obj
    glb_files = [f for f in os.listdir(glb_path) if f.endswith('.glb')]
    for glb_file in glb_files:
        if os.path.exists(os.path.join(glb_path, glb_file.replace('.glb', '.obj'))):
            # loguru.info(f"OBJ file already exists for {glb_file}, skipping conversion.")
            continue
        process_glb_to_vertcolor_obj(
            os.path.join(glb_path, glb_file),
            os.path.join(glb_path, glb_file.replace('.glb', '.obj'))
        )
    loguru.info(f"Processed {len(glb_files)} GLB files to OBJ with vertex colors in {glb_path}")
        
    # Get Object Part Field pre-calculation
    skip = ask_to_continue("Stage 5B: generate PartField pre-calculation results.")
    if not skip:
        loguru.info("Starting PartField pre-calculation...")
        partfield_args = [
            "-c", f'{PARTFIELD_ROOT}/configs/final/demo.yaml',
            "--opts",
            "continue_ckpt", f'{PARTFIELD_ROOT}/model/model_objaverse.ckpt',
            "result_name", f'partfield_features/{seq_name}/',
            "dataset.data_path", f'{seq_path}/build/mesh',
        ] 
        # due to the unflexible design of PartField codebase
        # this will save results under third_party/PartField/exp_results/partfield_features/hrse/
        file_utils.call_script_block(
            f'{PARTFIELD_ROOT}/partfield_inference.py',
            args_list=partfield_args,
            conda_env=PARTFIELD_ENV,
        )
    
    skip = ask_to_continue("Stage 5C: Separate object mesh into parts (PartField's method)")
    if not skip:
        part_sep_args = [
            "--root", f"exp_results/partfield_features/{seq_name}/",
            "--source_model", f"{seq_path}/build/mesh/{cano_frame:05d}.obj",
            "--dump_dir", f"{seq_path}/processed/partseps/",
            "--num_clusters", f"{args.part_cnt * 2}"
            # "--num-clusters", "6"
        ]
        file_utils.call_script_block(
            f'{PARTFIELD_ROOT}/run_seperate.py',
            args_list=part_sep_args,
            conda_env=PARTFIELD_ENV,
        )

def Hand_recon(seq_path, args):
    skip = ask_to_continue("Stage 6A: Run hand reconstruction using WiLoR.")
    if not skip:
        loguru.info("Starting hand mesh reconstruction using WiLoR...")
        wilor_args = [
            '--img_folder', f'{seq_path}/build/image'
        ]
        file_utils.call_script_block(
            WILOR_FILEPATH,
            args_list=wilor_args,
            conda_env=WILOR_ENV,
        )
    
    # NOTE register MANO - 
    #     adapted from HOLD. However, this consumes a lot of time due to opt-based registration.
    #     As a alternative method, We can directly extract MANO params from HaMeR / WiLoR inference
    #     results and apply dim swaps to avoid any registration process.
    #     See https://github.com/zc-alexfan/hold/issues/44;
    #     (Although this works for HAMER, I found unmatched finger axis order using WiLoR MANO param
    #      results, thus I still keep the registration process for now.)
    skip = ask_to_continue("Stage 6B: Register MANO mesh to object mesh.")
    if not skip:
        wilor_reg_args = [
            "--seq-path", seq_path,
        ]
        file_utils.call_script_block(
            WILOR_REG_FILEPATH,
            args_list=wilor_reg_args,
            conda_env=AHOI_ENV,
        )
    

def generate_conf(args, seq_name):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(f'{ROOT}/conf/template.yml', 'r', encoding='utf-8') as f:
        conf_template = yaml.load(f)

    cano_frame = args.cano_frame
    part_cnt = args.part_cnt
    conf_template.cano_frame = cano_frame
    conf_template.part_cnt = part_cnt
    conf_out = f'{ROOT}/conf/{seq_name}_p{part_cnt}c{cano_frame}.yml'

    with open(conf_out, 'w', encoding='utf-8') as f:
        yaml.dump(conf_template, f)

    loguru.info(f"Generated config file at {conf_out}, you can modify it before running ArtHOI.")


def main():
    args = get_args()
    seq_name = get_seq_name(args.video, args.seq_name)
    seq_path = os.path.join(args.out, seq_name)
    
    raw_resolution = file_utils.get_video_resolution(args.video)
    target_resolution = tuple(args.target_reso) if args.target_reso else raw_resolution
    
    Step1_Process_video_and_datadir(
        seq_path, 
        args, 
        target_resolution=target_resolution
    )
    Step2_Label_human_mask_and_inpaint(
        seq_path, 
        args, 
        target_resolution=target_resolution
    )
    Step3_Label_part_masks(
        seq_path, 
        args, 
    )
    Step4_Run_depth_estimation(
        seq_path, 
        args, 
        target_resolution=target_resolution
    )
    ask_to_continue("All video data preprocessing is done. "
                    "Continue to pre-process object and hand reconstruction? ")
    generate_conf(args, seq_name)
    Object_recon(
        seq_path,
        seq_name,
        args,
    )
    Hand_recon(
        seq_path,
        args,
    )
    loguru.critical("All done!")
    

if __name__ == "__main__":
    main()
