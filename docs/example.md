Here, we show an example using `rs_headphone` sequence from ArtHOI-RGBD to demonstrate the pipeline. The same steps can be applied to any other video sequence.

First, we pre-process the raw video file. Run `make_data.py` with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python make_data.py \
  --video data/raw_videos/rs_headphone.mp4 \
  --seq-name rs_headphone \
  --out data/seq/rs_headphone \
  --part-cnt 2 \
  --cano-frame 0 \
  --target_reso 960 540
```

Go through the script following the instructions printed in the terminal. The script will pause before each stage, and you can press enter to run the stage, type `s` to skip it, or type `exit` to terminate the pipeline.

After the preprocessing is done, you will get a data folder like this:

```text
├── build
│   ├── camera_param.npy
│   ├── image
│   ├── inpainting
│   │   ├── camera_param.npy
│   │   ├── image
│   │   ├── mask
│   │   └── metric_depth.pkl
│   ├── mask
│   │   ├── composite
│   │   ├── human
│   │   ├── obj
│   │   ├── part_0
│   │   ├── part_1
│   ├── mesh
│   │   ├── 00000.obj
│   │   └── raw
│   └── metric_depth.pkl
├── packed
│   ├── visuals_inpainting.npz
│   └── visuals.npz
└── processed
├── cano_obj
│   ├── 00000.png
├── gt.mp4
├── hamer
├── human_mask.mp4
├── inpainted_gt.mp4
├── mask_raw
│   ├── gt
│   └── vi
├── partseps
│   ├── 00000_part_0.ply
│   ├── 00000_part_0_vmap.npy
│   ├── 00000_part_1.ply
│   ├── 00000_part_1_vmap.npy
│   ├── 00000_part_2.ply
│   ├── 00000_part_2_vmap.npy
├── unidepv2
│   ├── gt
│   └── vi
├── vda
│   ├── gt_depths.npz
│   ├── gt_src.mp4
│   ├── gt_vis.mp4
│   ├── vi_depths.npz
│   ├── vi_src.mp4
│   └── vi_vis.mp4
├── vi.mp4
└── wilor_af
```

You can run the following command to start the ASR:

```bash
conda run -n fpose --live-stream \
python hrse/run_asr.py \
  --seq-path data/seq/rs_headphone \
  --conf conf/rs_headphone_p2c0.yml \
  --out data/asr_init/
```

Then run the partfield separation:

```bash
python hrse/object/part_seperation.py \
  --seq-path data/seq/rs_headphone \
  --conf conf/rs_headphone_p2c0.yml \
  --asr data/asr_init/rs_headphone \
  --pfsep data/seq/rs_headphone/processed/partseps/ \
  --output-path data/obj_cano_init/ \
```

On in-the-wild (youtube) videos, the results produced often are not perfect, and it is recommended to choose the canonical frame with almost all parts are the most visible state - hence ASR depends on a accurate 3D mesh reconstruction. If the ASR fails, you can try to choose a different canonical frame and re-run the relavent stages.

Finally, run the object motion optimization with:

```bash
python hrse/object_4d.py \
  --seq-path data/seq/rs_headphone \
  --conf conf/rs_headphone_p2c0.yml \
  --cano-reg-ckpt data/obj_cano_init/rs_headphone
```

Meanwhile, you can run MLLM contact reasoning:

```bash
# generate image sequences
python hrse/mllm/seq_vlmho_gen.py \
  -i data/seq/rs_headphone \
  -o data/mllm/seqk3k1/ -k 3 --interval 1 --text-position top-right
# reasoning
python hrse/mllm_contact.py \
  --mllm-path data/mllm/seqk3k1/ \
  --seq-path data/seq/rs_headphone \
  --out data/seq/rs_headphone/processed/
```

And finally H-O align:

```bash
python hrse/ho_align.py \
  --seq-path data/seq/rs_headphone \
  --conf conf/rs_headphone_p2c0.yml \
  --fit-ckpt data/obj_cano_init/rs_headphone
```

You can also ask LLM to generate a script to do this all automatically, but it is recommended to check the intermediate results after every step and adjust the config values if necessary.