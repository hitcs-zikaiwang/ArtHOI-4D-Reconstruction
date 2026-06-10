import cv2
import os
import json
from datetime import datetime
import argparse
from loguru import logger as loguru
from easydict import EasyDict as edict
import omegaconf

def parse_args():
    parser = argparse.ArgumentParser(description='Mesh alignment optimization')
    
    # File path arguments
    parser.add_argument('--seq-path', type=str, required=True)
    parser.add_argument('--conf', type=str, default='./conf/motion.yml',)
    parser.add_argument('--output-path', type=str, 
                        default=f'./outputs/{datetime.now().strftime("%m%d/%H%M%S")}/')
    parser.add_argument('--dump-vis', type=bool, default=True,
                        help='Dump visualization images')
    
    # preprocess choices
    parser.add_argument('--no-vi', '--use-video-inpainting', action='store_false', default=True,
                       help='Do not use video inpainting')
    parser.add_argument('--part-no', type=int, default=None, 
                        help='specific articulated part id to optimize')
    parser.add_argument('--skip_motion', action='store_true', default=False,
                        help='Fit global per-part motion')
    
    # fitting
    parser.add_argument('--cano-reg-ckpt', type=str, default=None,
                        help='Path to canonical register ckpt (skip canonical registeration)')
    parser.add_argument('--tracking-ckpt', type=str, default=None,
                        help='Path to tracking ckpt (skip tracking)')
    parser.add_argument('--part-fit-ckpt', type=str, default=None,
                        help='Path to part fit checkpoint (skip per-part fitting)')
    
    # debugging
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    return args

def output_args(args: edict, file):
    with open(file, 'w') as f:
        f.write("==== Arguments ====\n")
        # Recursively convert edict, DictConfig or dict to regular dict for JSON serialization
        def edict_to_dict(obj):
            if isinstance(obj, (edict, dict)):
                return {k: edict_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'DictConfig':
                # Convert OmegaConf DictConfig to dict
                try:
                    return {k: edict_to_dict(v) for k, v in omegaconf.OmegaConf.to_container(obj, resolve=True).items()}
                except ImportError:
                    # Fall back to dict conversion if omegaconf is not available
                    return {k: edict_to_dict(v) for k, v in dict(obj).items()}
            elif isinstance(obj, list):
                return [edict_to_dict(item) for item in obj]
            else:
                return obj
        args_dict = edict_to_dict(args)
        json.dump(args_dict, f, indent=4, sort_keys=False)
        f.write("\n==================\n")
