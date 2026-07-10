# ArtHOI Manual


The preprocessing pipeline is orchestrated by `make_data.py`. It intentionally calls several third-party tools as standalone scripts because their dependency sets can be difficult to merge into a single environment.

## Before You Start

1. Prepare one RGB video of a hand manipulating an articulated object.
2. Check the environment names and script paths near the top of `make_data.py`:
   - `AHOI_ENV`
   - `DIFFUERASER_ENV`
   - `WILOR_ENV`
   - `PARTFIELD_ENV`
   - paths such as `SAM2_FILEPATH`, `DIFFUERASER_FILEPATH`, and
     `WILOR_FILEPATH`

3. Make sure the required model weights are available before running each stage. Some tools download weights on first use, while others require manual placement.

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

> **Note: If you have multiple available GPUs, it's recommended to set `CUDA_VISIBLE_DEVICES` to only one GPU, or else you may have to modify `partfield_inference.py` to prevent it from using all GPUs and cause an error.**

`make_data.py` pauses before each stage. Press enter to run the stage, type `s` to skip it, or type `exit` to terminate the pipeline.

## 1. Prepare the Raw Data

The first stage creates the dataset directory structure, splits the input video into frames, and writes the normalized ground-truth video to:

```text
<seq_path>/processed/gt.mp4
<seq_path>/build/image/
```

Use `--target_reso width height` if you want all later stages to use a fixed resolution. The default in `make_data.py` is `960 540`.

## 2. Human Mask Labeling and Inpainting

### 2A. Label the Human Mask with SAM2

Install SAM2 according to the official guide. In many cases it can be installed inside the main ArtHOI conda environment with dependencies ignored, but verify critical package versions afterward, especially `torch` and `xformers`. If the merged environment becomes unstable, install and run SAM2 in a separate conda environment instead.

The pipeline launches the ArtHOI SAM2 GUI wrapper:

```text
third_party/sam2/app_hrse.py
```

Open the Gradio URL printed by the script, label the human mask, and click `Exit` in the GUI when labeling is complete. The raw mask output is written
under:

```text
<seq_path>/processed/mask_raw/gt/
```

You may also use any other segmentation model or manual annotation workflow. The pipeline only requires that the final masks follow the expected directory layout.

### 2B. Run Video Inpainting with DiffuEraser

DiffuEraser removes the hand from the video using the human mask. The pipeline expects the inpainted result to be saved as:

```text
<seq_path>/processed/inpainted_gt.mp4
```

It is useful to patch the beginning of `run_diffueraser.py` so it accepts input video and mask paths consistently:

```python
parser.add_argument('--input_video', '--iv', type=str, help='Path to the input video')
parser.add_argument('--input_mask', '--im', type=str, help='Path to the input mask')
parser.add_argument('--seed', type=int, default=0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

input_video_name = os.path.splitext(os.path.basename(args.input_video))[0]
output_path = os.path.join(args.save_path, f"inpainted_{input_video_name}.mp4")
```

If you do not patch DiffuEraser and it saves the video under a different file name, update Stage 2B in `make_data.py` accordingly before continuing.

If you find inpainting quality to be poor, you can either try a different random seed or use a newer inpainting / object removal model such as [ROSE](https://github.com/Kunbyte-AI/ROSE/).

## 3. Part-Level Mask Labeling

Run SAM2 twice:

1. Label object-part masks on the original frames seperately for each part.
2. Label the corresponding object-part masks on the inpainted frames.

Make sure to keep **object part names** consistent between the original and inpainted videos.

## 4. Depth and Camera Intrinsics

If you launch the Video-Depth-Anything demo UI manually and want to keep it local-only, edit its `app.py` and set:

```python
share = False
```

## 5. Object Initialization

Two object initialization workflows are supported:

1. Generate a mesh locally with Hunyuan3D or another image-to-3D model.
2. Call a Hunyuan3D API service.

For a quick trial run, it is usually faster to skip this stage in the script and manually generate the canonical object mesh with the Hunyuan3D web service. Place the resulting `.glb` file in:

```text
<seq_path>/build/mesh/
```

The pipeline converts `.glb` files in this folder into vertex-colored `.obj` files automatically.

If you use an international Tencent Hunyuan API endpoint, set:

```bash
export HUNYUAN_3D_API_URL=<your-api-url>
export HUNYUAN_3D_API_KEY=<your-api-key>
```

Then check `utils/hy3d/api_caller.py` and `make_data.py` to make sure the model name and output path match your service.

## 6. PartField Precomputation and Part Separation

PartField is used to generate object features and separate the object mesh into candidate parts.

The pipeline runs:

```text
third_party/PartField/partfield_inference.py
third_party/PartField/run_seperate.py
```

`partfield_inference.py` writes intermediate features under the PartField repository:

```text
third_party/PartField/exp_results/partfield_features/<seq_name>/
```

`run_seperate.py` then writes part-separation results to:

```text
<seq_path>/processed/partseps/
```

`PF_cluster_cnt` in the config file (`./conf/<seq_name>.yml`) to the number of articulated object parts you expect. The current script passes `part_cnt * 2` clusters to PartField separation, which is intended to over-segment before later selection and fitting.

However, PartField separation quality can vary across capture sources, please visually inspect the separated parts, adjust the cluster count, and re-run the seperation stage if necessary. If PartField fails to separate the parts correctly, you can also use a 2D-mask based separation method by running `hrse/object/part_seperation.py` with the `--use_2d_mask` argument.

## 7. Hand Reconstruction and MLLM Contact Preparation

**Since WiLoR is not installable, you have to copy and paster `WiLoR_ArtHOI.py` to your local environment and run it there.**

Modify `third_party/WiLoR/wilor/models/__init__.py` to load the right `MANO_DIR`: comment `line 30-35`.

Known problem in MLLM contact reasoning: 

- Qwen-VL-Max may be unavailable from the original provider, you can switch to a newer version of Qwen (e.g., qwen3.7-plus) or use a third-party provider to keep this model choice.
- some models (e.g., qwen3-vl-flash) may **not be able to output the correct frame number in the answer**, which will cause contact.json fail to be generated. If you encounter this problem and don't want to use a cutting-edge model, please modify the code in `mllm_contact.py` to set `frame_num` according to the actual frame number by the input frame image name. For example, if the input frame image name is `sampleRGBD_123.jpg`, then set `frame_num = 123` accordingly.
- Changing model to a newer one (e.g., claude opus 4.6) may encounter a inference speed slowdown due to long thinking CoT, where you might consider adjust the reasoning effort and/or use batch processing.

## 8. Sequence Config

Review the generated sequence config created under:

```text
conf/<seq_name>_p<part_cnt>c<cano_frame>.yml
```

## 9. Optimizing

- Run MLLM contact reasoning:

```bash
# generate image sequences
python hrse/mllm/seq_vlmho_gen.py -i <dataset_path> -o <your_desired_path/seqk3k1/> -k 3 --interval 1 --text-position top-right
# run reasoning
python hrse/mllm_contact.py --mllm-path <your_desired_path/seqk3k1/> --seq-path <dataset_path> --out <dataset_path>/processed/
```

Meanwhile, you can run the object part:

- Run ASR object initialization optimization with:

```bash
# asr cano init, run under foundationpose env (or arthoi env if you merged them)
conda run -n fpose --live-stream \
python hrse/run_asr.py --seq-path <dataset_path> --conf <config_file> --out <your_desired_path/asr_init/> 
# then seperate parts
python hrse/object/part_seperation.py \
  --seq-path <dataset_path> --conf <config_file> \
  --asr <your_desired_path/asr_init/seq_name> \
  --pfsep <your_desired_path/partfield_seps/>
  --output-path <your_desired_path/obj_cano_init> \
  # --use_2d_mask  # optional, use 2D mask based separation instead of PartField separation
```

- Run object motion optimization with:
  
```bash
python hrse/object_4d.py --seq-path <dataset_path> --conf <config_file> \
  --cano-reg-ckpt <your_desired_path/obj_cano_init/seq_name>
# you can also specify --output-path
```

> Note: If the optimized motion is not aligned well, try to adjust the `per_part.num_iter` in the config file to a larger value (e.g. `850`).

- Finally, align HO with:

```bash
python hrse/ho_align.py --seq-path <dataset_path> --conf <config_file> --fit-ckpt <your_object_recon_path>
```