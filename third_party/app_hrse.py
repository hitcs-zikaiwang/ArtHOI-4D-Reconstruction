import torch
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

import colorsys
import datetime
import os
import subprocess
import tempfile
import threading
import time

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np

from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor

MASK_TARGETS = (
    "human",
    "obj",
    "part_0",
    "part_1",
    "part_2",
    "part_3",
    "part_4",
)

"""
Usage:
python app.py --root_dir [data_root]
"""

class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.inference_state = None
        self.selected_mask_target = None
        self.selected_mask_dir = ""
        self.init_sam_model()

    def init_sam_model(self):
        if self.sam_model is None:
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
            guru.info(f"loaded model checkpoint {self.checkpoint_dir}")


    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def make_index_mask(self, masks):
        assert len(masks) > 0
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def reset(self):
        if self.inference_state is None:
            return self.get_sam_features()
        message = self.clear_annotation_state()
        return message, self.image

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.inference_state = None
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        
        # Keep supported image formats directly.
        img_paths_final = []
        for name in self.img_paths:
            if name.endswith((".png", ".jpg", ".jpeg")):
                img_paths_final.append(name)
            else:
                guru.error(f"Unsupported image format: {name}")
        
        self.img_paths = img_paths_final
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image

        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        """Ensure all supported image formats can be processed."""
        # Check whether image paths are available.
        if not self.img_paths:
            raise gr.Error("No images found in the selected directory.")
            
        # SAM2 video initialization may require JPG inputs.
        jpg_exists = any(path.lower().endswith(('.jpg', '.jpeg')) for path in self.img_paths)
        
        if not jpg_exists:
            # Create a temporary directory for converted JPG images.
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            guru.info(f"Creating temporary directory for JPG images: {temp_dir}")
            
            # Convert all frames to JPG.
            temp_paths = []
            for i, img_path in enumerate(self.img_paths):
                img = iio.imread(img_path)
                jpg_path = f"{temp_dir}/{i:05d}.jpg"
                iio.imwrite(jpg_path, img)
                temp_paths.append(jpg_path)
            
            # Initialize SAM with the temporary directory.
            self.inference_state = self.sam_model.init_state(video_path=temp_dir)
            
            # Keep the temporary directory alive while the model may use it.
        else:
            # Initialize SAM with the original directory.
            self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
            
        self.sam_model.reset_state(self.inference_state)
        msg = (
            "SAM features extracted. "
            "Click points to update mask, and submit when ready to start tracking"
        )
        return msg, self.image

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, frame_idx, i, j):
        """
        get the index mask of the objects
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )
        mask = self.make_index_mask(masks)

        return mask
    

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_model is not None
        

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return  {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }


    def run_tracker(self) -> tuple[str, str]:
        self.require_mask_target()
        if self.inference_state is None:
            raise gr.Error("SAM features are not ready yet. Reload the image directory or press Reset.")
        if not self.selected_points:
            raise gr.Error("Please add at least one point before submitting the mask.")

        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
            # index_masks_all.append(self.make_index_mask(masks))

        self.index_masks_all = [self.make_index_mask(v) for k, v in video_segments.items()]

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = os.path.join(tempfile.gettempdir(), "tracked_colors.mp4")
        iio.mimwrite(out_vidpath, out_frames)
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        self.require_mask_target()
        if not output_dir:
            raise gr.Error("Please select a mask target before saving masks.")
        if not self.color_masks_all or not self.index_masks_all:
            raise gr.Error("Please submit the mask for tracking before saving masks.")
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask, id_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            
            mono_mask = (clr_mask[:, :, 0] > 0).astype(np.uint8) * 255
            iio.imwrite(out_path, mono_mask)
            np_out_path = f"{output_dir}/{name[:-4]}.npy"
            np.save(np_out_path, id_mask)
        
        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        return message

    def clear_annotation_state(self) -> str:
        self.selected_points.clear()
        self.selected_labels.clear()
        self.cur_label_val = 1.0
        self.cur_mask_idx = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []
        if self.inference_state is not None:
            self.sam_model.reset_state(self.inference_state)
        return "Cleared the current annotation state."

    def set_mask_target(self, mask_root_dir: str, target: str) -> tuple[str, str]:
        self.selected_mask_target = target
        self.selected_mask_dir = os.path.join(mask_root_dir, target)
        return self.selected_mask_dir, f"Ready to annotate {target}. Mask output path: {self.selected_mask_dir}"

    def require_mask_target(self):
        if self.selected_mask_target is None:
            raise gr.Error("Please select a mask target before continuing.")


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(
    checkpoint_dir,
    model_cfg,
    work_dir,
    mask_root_dir=None,
):
    prompts = PromptGUI(checkpoint_dir, model_cfg)
    default_img_dir = f"{work_dir}"
    default_mask_root_dir = mask_root_dir if mask_root_dir is not None else f"{work_dir}/mask"
    
    start_instructions = "Select an image directory with frames to process."
    
    with gr.Blocks() as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )
        with gr.Column(elem_id="mask_target_column"):
            gr.Markdown("Select mask target")
            with gr.Row():
                target_buttons = [
                    gr.Button(target)
                    for target in MASK_TARGETS
                ]

        with gr.Row():
            img_dir_field = gr.Text(
                value=default_img_dir,
                label="Input image directory (full path)"
            )
            
        with gr.Row():
            with gr.Column():
                frame_index = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    interactive=False
                )
                reset_button = gr.Button("Reset")
                input_image = gr.Image(
                    label="Input Frame",
                    every=1,
                )
                with gr.Row():
                    pos_button = gr.Button("Toggle positive")
                    neg_button = gr.Button("Toggle negative")
                clear_button = gr.Button("Clear points")

            with gr.Column():
                mask_root_field = gr.Text(
                    value=default_mask_root_dir,
                    label="Mask root path",
                    interactive=True
                )
                mask_dir_field = gr.Text(
                    value="",
                    label="Selected mask save path",
                    interactive=True
                )
                output_img = gr.Image(label="Current selection")
                submit_button = gr.Button("Submit mask for tracking")
                final_video = gr.Video(label="Masked video")
                save_button = gr.Button("Save masks")
                exit_button = gr.Button("Exit", elem_id="exit_button", variant="stop")

        def update_image_dir(img_dir):
            if not os.path.isdir(img_dir):
                raise gr.Error(f"Directory {img_dir} does not exist.")
            
            num_imgs = prompts.set_img_dir(img_dir)
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1, interactive=True)
            image = prompts.set_input_image(0)
            message, image = prompts.get_sam_features()
            message = f"Loaded {num_imgs} images from {img_dir}. {message}"
            return slider, message, image

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            prompts.require_mask_target()
            if prompts.inference_state is None:
                raise gr.Error("SAM features are not ready yet. Reload the image directory or press Reset.")
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            index_mask = prompts.add_point(frame_idx, i, j)
            guru.debug(f"{index_mask.shape=}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, color_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out, "Point added. Submit the mask when the selection looks good."

        def select_mask_target(mask_root_dir, target):
            save_path, message = prompts.set_mask_target(mask_root_dir, target)
            prompts.clear_annotation_state()
            return save_path, None, None, message

        # when the img_dir_field changes
        img_dir_field.submit(
            update_image_dir,
            [img_dir_field],
            [frame_index, instruction, input_image],
        )

        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [frame_index, input_image], [output_img, instruction])

        reset_button.click(prompts.reset, outputs=[instruction, input_image])
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])

        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [mask_dir_field], outputs=[instruction]
        )

        def exit_program():
            def delayed_exit():
                time.sleep(0.3)
                os._exit(0)

            threading.Thread(target=delayed_exit, daemon=True).start()
            return "Closing the browser page and stopping the app."

        close_page_js = "() => { window.open('', '_self'); window.close(); }"

        for target, button in zip(
            MASK_TARGETS,
            target_buttons,
        ):
            button.click(
                select_mask_target,
                [mask_root_field, gr.State(target)],
                [mask_dir_field, output_img, final_video, instruction],
            )

        exit_button.click(exit_program, outputs=[instruction], js=close_page_js)

        def load_default_directory():
            if os.path.isdir(default_img_dir):
                return update_image_dir(default_img_dir)
            raise gr.Error(f"Default directory {default_img_dir} does not exist. Please select a valid directory.")

        demo.load(load_default_directory, outputs=[frame_index, instruction, input_image])

    return demo


if __name__ == "__main__":
    import argparse
    
    FILE_LOC = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23341)
    parser.add_argument("--checkpoint_dir", type=str, default=f"/{FILE_LOC}/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--model_cfg", type=str, default=f"/{FILE_LOC}/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--im_dir", type=str, required=True)
    parser.add_argument("--mask_root_dir", type=str, default="output_masks")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    demo = make_demo(
        args.checkpoint_dir,
        args.model_cfg,
        args.im_dir,
        args.mask_root_dir,
    )
    demo.launch(server_port=args.port)
