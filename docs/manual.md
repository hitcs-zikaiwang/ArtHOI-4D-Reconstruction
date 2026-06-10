# Preprocessing Manual


The preprocessing pipeline is orchestrated by `make_data.py`. It intentionally
calls several third-party tools as standalone scripts because their dependency
sets can be difficult to merge into a single environment.

## Before You Start

1. Prepare one RGB video of a hand manipulating an articulated object.
2. Install the main ArtHOI environment and the required third-party tools. See
   `docs/installation_guide.md` for the general setup.
3. Check the environment names and script paths near the top of `make_data.py`:

   - `AHOI_ENV`
   - `DIFFUERASER_ENV`
   - `WILOR_ENV`
   - `PARTFIELD_ENV`
   - paths such as `SAM2_FILEPATH`, `DIFFUERASER_FILEPATH`, and
     `WILOR_FILEPATH`

4. Make sure the required model weights are available before running each stage.
   Some tools download weights on first use, while others require manual
   placement.

Run preprocessing with:

```bash
python make_data.py \
  --video /path/to/input.mp4 \
  --seq-name my_sequence \
  --out ds/ \
  --part-cnt 2 \
  --cano-frame 0 \
  --target_reso 960 540
```

`make_data.py` pauses before each stage. Press enter to run the stage, type `s`
to skip it, or type `exit` to terminate the pipeline.

## 1. Prepare the Raw Data

The first stage creates the dataset directory structure, splits the input video
into frames, and writes the normalized ground-truth video to:

```text
<seq_path>/processed/gt.mp4
<seq_path>/build/image/
```

Use `--target_reso width height` if you want all later stages to use a fixed
resolution. The default in `make_data.py` is `960 540`.

## 2. Human Mask Labeling and Inpainting

### 2A. Label the Human Mask with SAM2

Install SAM2 according to the official guide. In many cases it can be installed
inside the main ArtHOI conda environment with dependencies ignored, but verify
critical package versions afterward, especially `torch` and `xformers`. If the
merged environment becomes unstable, install and run SAM2 in a separate conda
environment instead.

The pipeline launches the ArtHOI SAM2 GUI wrapper:

```text
third_party/sam2/app_hrse.py
```

Open the Gradio URL printed by the script, label the human mask, and click
`Exit` in the GUI when labeling is complete. The raw mask output is written
under:

```text
<seq_path>/processed/mask_raw/gt/
```

If the SAM2 checkpoint or config is missing, download or update the expected
files:

```text
sam2.1_hiera_base_plus.pt
sam2.1_hiera_b+.yaml
```

The exact default paths are defined in `third_party/sam2/app_hrse.py`.

You may also use any other segmentation model or manual annotation workflow.
The pipeline only requires that the final masks follow the expected directory
layout.

### 2B. Run Video Inpainting with DiffuEraser

DiffuEraser removes the hand from the video using the human mask. The pipeline
expects the inpainted result to be saved as:

```text
<seq_path>/processed/inpainted_gt.mp4
```

It is useful to patch the beginning of `run_diffueraser.py` so it accepts input
video and mask paths consistently:

```python
parser.add_argument('--input_video', '--iv', type=str, help='Path to the input video')
parser.add_argument('--input_mask', '--im', type=str, help='Path to the input mask')
parser.add_argument('--seed', type=int, default=0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

input_video_name = os.path.splitext(os.path.basename(args.input_video))[0]
output_path = os.path.join(args.save_path, f"inpainted_{input_video_name}.mp4")
```

If you do not patch DiffuEraser and it saves the video under a different file
name, update Stage 2B in `make_data.py` accordingly before continuing.

After inpainting, the pipeline resizes the result to:

```text
<seq_path>/processed/vi.mp4
```

and splits it into:

```text
<seq_path>/build/inpainting/image/
```

## 3. Part-Level Mask Labeling

Run SAM2 twice:

1. Label object-part masks on the original frames in `<seq_path>/build/image/`.
2. Label the corresponding object-part masks on the inpainted frames in
   `<seq_path>/build/inpainting/image/`.

The raw masks are written under:

```text
<seq_path>/processed/mask_raw/gt/
<seq_path>/processed/mask_raw/vi/
```

`utils.file_utils.post_process_sam2_masks()` then converts them into the layout
used by training:

```text
<seq_path>/build/mask/
<seq_path>/build/inpainting/mask/
```

Keep object part names consistent between the original and inpainted videos.
The later reconstruction stages assume that the same semantic part appears in
both mask folders.

## 4. Depth and Camera Intrinsics

### 4A. Run Video-Depth-Anything

Video-Depth-Anything estimates temporal depth for both the original and
inpainted videos. It can often be installed into the main ArtHOI environment
with dependencies ignored, but again verify critical package versions after
installation. If the merged environment is unstable, run it in a separate
environment.

The pipeline calls the shipped `run.py` entrypoint and saves depth predictions
under:

```text
<seq_path>/processed/vda/
```

If you launch the Video-Depth-Anything demo UI manually and want to keep it
local-only, edit its `app.py` and set:

```python
share = False
```

### 4B. Align Depth and Camera Intrinsics with UniDepthV2

UniDepthV2 aligns the VDA depth maps to metric scale and estimates camera
intrinsics. It can also often be installed into the main ArtHOI environment with
dependencies ignored. Pay special attention to `numpy`, `torch`, and `xformers`
compatibility. If needed, install UniDepthV2 in a separate conda environment.

The ArtHOI wrapper script is:

```text
third_party/unidepth/metric_align.py
```

The aligned outputs are first written under:

```text
<seq_path>/processed/unidepv2/gt/
<seq_path>/processed/unidepv2/vi/
```

The pipeline then moves the final metric-depth files and camera parameters into:

```text
<seq_path>/build/metric_depth.pkl
<seq_path>/build/cameras.npz
<seq_path>/build/inpainting/metric_depth.pkl
<seq_path>/build/inpainting/cameras.npz
```

## 5. Object Initialization

Two object initialization workflows are supported:

1. Generate a mesh locally with Hunyuan3D or another image-to-3D model.
2. Call a Hunyuan3D API service.

For a quick trial run, it is usually faster to skip this stage in the script and
manually generate the canonical object mesh with the Hunyuan3D web service.
Place the resulting `.glb` file in:

```text
<seq_path>/build/mesh/
```

The pipeline converts `.glb` files in this folder into vertex-colored `.obj`
files automatically.

If you use an international or out-of-China Tencent Hunyuan API endpoint, set:

```bash
export HUNYUAN_3D_API_URL=<your-api-url>
export HUNYUAN_3D_API_KEY=<your-api-key>
```

Then check `utils/hy3d/api_caller.py` and `make_data.py` to make sure the model
name and output path match your service.

## 6. PartField Precomputation and Part Separation

PartField is used to generate object features and separate the object mesh into
candidate parts.

The pipeline runs:

```text
third_party/PartField/partfield_inference.py
third_party/PartField/run_seperate.py
```

`partfield_inference.py` writes intermediate features under the PartField
repository:

```text
third_party/PartField/exp_results/partfield_features/<seq_name>/
```

`run_seperate.py` then writes part-separation results to:

```text
<seq_path>/processed/partseps/
```

Set `--part-cnt` to the number of articulated object parts you expect. The
current script passes `part_cnt * 2` clusters to PartField separation, which is
intended to over-segment before later selection and fitting.

After this stage, visually inspect the separated parts. Part separation quality
can vary across capture sources such as RealSense data, YouTube videos, and
RSDK/R3D-style captures, so this stage should not be treated as fully automatic.

## 7. Hand Reconstruction and MLLM Contact Preparation

### 7A. Run WiLoR

WiLoR may not be installable cleanly as a normal package in every environment.
If installation fails, copy `WiLoR_ArtHOI.py` into your local WiLoR environment
and run it there, or update `WILOR_FILEPATH` in `make_data.py` to point to the
working local script.

The repository contains the ArtHOI wrapper at:

```text
third_party/WiLoR/WiLoR_ArtHOI.py
```

or, in some layouts:

```text
third_party/WiLoR_ArtHOI.py
```

Make sure `WILOR_FILEPATH` points to the file that exists in your checkout.

If WiLoR cannot find MANO, modify:

```text
third_party/WiLoR/wilor/models/__init__.py
```

so it loads the correct `MANO_DIR`. In the original WiLoR file this may require
commenting out the hard-coded MANO path block around lines 30-35 and replacing
it with the path used by your environment.

### 7B. Register MANO

After WiLoR inference, the pipeline runs:

```text
utils/WiLoR_reg_mano.py
```

This registers the MANO hand mesh to the object sequence. The registration can
take a long time because it is optimization-based. The code keeps this step for
now because directly using MANO parameters from HaMeR or WiLoR may produce
inconsistent finger-axis ordering.

### 7C. Contact Reasoning

The MLLM contact stage depends on the processed masks, object parts, hand mesh,
and sequence config generated by the earlier stages. Before running contact
reasoning, verify that the following assets exist:

```text
<seq_path>/build/image/
<seq_path>/build/mask/
<seq_path>/build/inpainting/image/
<seq_path>/build/inpainting/mask/
<seq_path>/build/mesh/
<seq_path>/processed/partseps/
```

Then generate or review the sequence config created under:

```text
conf/<seq_name>_p<part_cnt>c<cano_frame>.yml
```

## Troubleshooting Checklist

- If a third-party script cannot import its dependencies, first check whether
  `make_data.py` is calling it from the intended conda environment.
- If a stage finishes but the next stage cannot find its input, compare the file
  name written by the third-party script with the path expected in `make_data.py`.
- If SAM2 launches but masks are missing, confirm that you clicked `Exit` in the
  Gradio UI after finishing annotation.
- If depth alignment fails, verify that VDA wrote both `gt_depths.npz` and
  `vi_depths.npz`.
- If object initialization fails through the API, use a manually generated
  Hunyuan3D `.glb` file for the first trial run.
- If PartField separation is poor, try a different canonical frame, increase the
  cluster count, or manually choose a better separated part set before training.
- If WiLoR fails to load MANO, check both the MANO file location and the hard
  coded path inside the local WiLoR checkout.
