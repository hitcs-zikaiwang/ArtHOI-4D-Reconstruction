# MLLM Contact Reasoning

This directory contains the multimodal large language model (MLLM) utilities used by ArtHOI-4D-Reconstruction. The tools prepare RGB-D frame sequences, send the resulting visual prompts to a vision-language model, and convert the model responses into hand-object contact annotations for later hand-object alignment.

The MLLM stage is especially useful when contact cannot be recovered reliably from geometry alone. It asks a vision-language model to reason about the left and right hands independently, using both the RGB image and the colorized depth image. The resulting contact labels are consumed by the downstream HRSE/HO alignment pipeline.

## Pipeline overview

```text
<sequence>/build/image/*.jpg
<sequence>/build/metric_depth.pkl
             |
             |  seq_vlmho_gen.py or vlmho_gen.py
             v
<output>/<sequence>/sampleRGBD_*.jpg
             |
             |  hrse/mllm_contact.py
             |  Qwen or Gemini multimodal inference
             v
<sequence>/processed/ho_contact.json
             |
             v
       hand-object alignment
```

The generated `sampleRGBD_*.jpg` images contain RGB frames in the top row and colorized depth frames in the bottom row. Black separators and frame-name overlays make it easier for the model to associate each prediction with the correct frame.

## File structure

```text
mllm/
├── README.md             # This document
├── prompts.py            # Prompts for perspective and contact reasoning
├── qwen_api.py           # Qwen/DashScope multimodal API wrappers
├── gemini_api.py         # Gemini multimodal API wrappers
├── vlmho_gen.py          # Randomly samples and merges RGB-D frames
├── seq_vlmho_gen.py      # Sequentially samples and merges RGB-D frames
└── manual_label_ho.py    # Interactive contact and finger annotation tool
```

### `seq_vlmho_gen.py`

Creates a complete sequence of overlapping RGB-D composites. This is the recommended preprocessing script when every consecutive temporal window should be evaluated.

### `vlmho_gen.py`

Creates a configurable number of randomly selected RGB-D composites. It is useful for quick experiments or repeated sampling of a long sequence.

### `qwen_api.py` and `gemini_api.py`

Provide small wrappers with a common style of interface for single-image and multi-image requests. They encode local images, call the selected provider, and return the model text response. The current contact pipeline uses the Qwen wrappers; Gemini wrappers are available for experiments and alternative providers.

### `prompts.py`

Stores the prompts used for camera-perspective classification and frame-by-frame contact reasoning. The contact prompt asks the model to check visibility, RGB proximity, depth continuity, and the distinction between solid contact and mere proximity.

### `manual_label_ho.py`

Generates or edits contact labels without an MLLM. It supports interactive contact ranges and optional finger labels (`thumb`, `index`, and `middle`) for the left and right hands.

## Requirements and API keys

Install the dependencies required by the repository environment before running these scripts. In addition, install the provider SDK used by the selected backend:

```bash
# Qwen/DashScope backend
pip install openai dashscope

# Gemini backend (optional)
pip install google-genai
```

Set API credentials through environment variables. Do not put real keys in this directory or commit them to Git:

```bash
export QWEN_API_KEY="<your-dashscope-key>"
export GEMINI_API_KEY="<your-gemini-key>"  # only needed for Gemini
```

The scripts use provider-specific model defaults defined in `qwen_api.py` and `gemini_api.py`. Check the selected model and provider availability before a long run; model names and provider availability can change over time.

## Input data contract

For a sequence directory such as `data/seq/rs_headphone`, the preprocessing scripts expect:

```text
data/seq/rs_headphone/
├── build/
│   ├── image/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── metric_depth.pkl
└── ...
```

RGB files are sorted naturally by filename. The depth pickle must contain one depth map per RGB frame. The frame order must match; otherwise the RGB and depth rows will describe different frames and contact reasoning will be unreliable.

## Usage

Run commands from the repository root.

### 1. Generate sequential RGB-D windows

```bash
python hrse/mllm/seq_vlmho_gen.py \
  -i data/seq/rs_headphone \
  -o data/mllm/seqk3k1/ \
  -k 3 \
  --interval 1 \
  --text-position top-right
```

This creates files such as `sampleRGB_1.jpg` and `sampleRGBD_1.jpg` under the output sequence directory. `-k` controls the number of frames per window and `--interval` controls the frame stride.

### 2. Generate randomly sampled windows

```bash
python hrse/mllm/vlmho_gen.py \
  -i data/seq/rs_headphone \
  -o data/mllm/seqk3k1/ \
  -k 3 \
  -r 50 \
  --interval 1 \
  --text-position top-right
```

Use `--rand-k` when the frames inside each composite should also be sampled randomly. Inspect a few generated composites before launching a large inference job.

### 3. Run MLLM contact reasoning

```bash
python hrse/mllm_contact.py \
  --mllm-path data/mllm/seqk3k1/rs_headphone \
  --seq-name rs_headphone \
  --out data/seq/rs_headphone/processed/
```

The contact script reads RGB-D composites, performs perspective and contact reasoning, parses the structured model responses, and saves the merged result as `ho_contact.json` in the requested output directory. Multiple predictions for a frame are merged by majority vote; finger labels are merged using the most frequent combination.

The exact `--mllm-path` should point to the directory containing the `sampleRGBD_*.jpg` files produced by the preprocessing step. If the output layout differs, adjust the path accordingly.

### 4. Manual labeling alternative

```bash
python hrse/mllm/manual_label_ho.py \
  --seq-path data/seq/rs_headphone \
  --out /processed/ho_contact.json \
  --mode fingers
```

Use `--mode 1` for the original contact-range workflow. Use `--mode 2` or `--mode fingers` to mark contact ranges and annotate the fingers involved. Add `--no-interactive` only for smoke tests; it skips user input and does not produce meaningful human labels.

## Output format

The contact output is a JSON object with the following general structure:

```json
{
  "frames_cnt": 2,
  "appeared": ["left", "right"],
  "contacts": [
    {
      "frame": 0,
      "r_contact": true,
      "l_contact": false,
      "r_fingers": ["index"],
      "l_fingers": []
    }
  ]
}
```

Frame numbers should match the source image names. The downstream alignment code relies on this mapping, so verify a few annotations visually before optimization.

## Practical checks

- Confirm that the RGB image count and depth-map count match.
- Open several `sampleRGBD_*.jpg` files and verify that RGB and depth rows are aligned.
- Check that `QWEN_API_KEY` or `GEMINI_API_KEY` is set in the active shell.
- Keep API retries and batch sizes conservative when using a provider with rate limits.
- Review the generated `ho_contact.json`; MLLM predictions are not guaranteed to be correct, especially under occlusion or when the hand is close to but not touching the object.
- If a model returns incorrect frame numbers, derive the frame number from the composite filename and adjust the parser in `hrse/mllm_contact.py` before using the annotations downstream.

## Relationship to the rest of the pipeline

This folder prepares and labels hand-object interaction evidence; it does not reconstruct the object or optimize the final 4D motion by itself. A typical workflow is:

1. Build or preprocess the sequence and depth maps.
2. Generate RGB-D windows with `seq_vlmho_gen.py` or `vlmho_gen.py`.
3. Run `hrse/mllm_contact.py` or create labels manually.
4. Inspect `processed/ho_contact.json`.
5. Pass the contact labels to the downstream hand-object alignment and reconstruction stages.

