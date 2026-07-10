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

After the preprocessing is done, you can run the following command to start the ASR:

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