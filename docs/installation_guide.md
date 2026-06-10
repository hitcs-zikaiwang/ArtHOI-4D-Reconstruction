# ArtHOI Installation Guide

This guide describes a clean installation path for the ArtHOI preprocessing and reconstruction code. It is also readable and actionable for Coding Agents such as Codex or Claude Code.

The reference environment used while preparing this guide is `arthoi`:
  
- Python `3.10.0`
- PyTorch `2.2.0+cu121`
- CUDA toolkit path used for extension builds: `/usr/local/cuda-12.1`

> It is recommended to use the exactly same torch and cuda version as suggested. But it is highly possible that versions close to these will also work (e.g. torch 2.7 & cuda 12.7), if the suggested CUDA version is not available for your system setup.

> To Coding Agent: You should not make any write, file changes or other actions before **reciting every critical warnings listed below and confirming with the user**. Always ask for confirmation before taking any action that modifies the system or files, and make sure to explain the potential risks and how to mitigate them.

## 1. Create the Conda Environment

```bash
mamba create -n arthoi python=3.10.0 
mamba activate arthoi
```

Install PyTorch with CUDA 12.1 first. It is not included in the `requirements.txt`.

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# install CUDA12.1 if it's not already available on the system, e.g. by following NVIDIA's official guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
```

Make sure CUDA build environment variables are set before installing CUDA extensions:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
```

## 2. Install Standard Python Requirements

From the repository root:

```bash
cd <path>/ArtHOI
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally **excludes** packages that should be installed from local source or pinned git commits, including `PyTorch3D`, `CoTracker`, `nvdiffrast`, `manopth`, `chumpy`, `SAM2` and others. You should install them following the instructions in the next section. You can ask a coding agent to write a script.

Then install ArtHOI:

`pip install -e . --no-build-isolation`

## 3. Install Local / Source Packages

Install these after PyTorch is available in the environment.

### PyTorch3D

Install Pytorch3D v0.7.8 or the latest compatible version following the official guide:

`https://github.com/facebookresearch/pytorch3d#installation`

basically you can just run `cd third_party && git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && git checkout 33824be` and then `pip install -e . --no-build-isolation` to install the package in editable mode.

### nvdiffrast

```bash
python -m pip install \
  "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
```

### manopth

```bash
python -m pip install \
  "git+https://github.com/hassony2/manopth.git@4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b"
```

### chumpy 

```bash
python -m pip install \
  "git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a" \
  --no-build-isolation
```

The observed working version is `0.71`.

### SAM2



### CoTracker 3

It is highly recommended to install CoTracker locally to visualize tracking trajectories. Clone the CoTracker repository from official source `https://github.com/facebookresearch/co-tracker#install-a-development-version` to the `third_party/co-tracker` directory. It is not needed to download weights manually since the code will automatically download the required weights on first use.

### Final Check

Run a `pip check` command after this step, it should report no broken requirements found.

### Other conveniently installable packages

You can basically merge `UniDepth`, `PartField`, `FoundationPose` and `Video-Depth-Anything` into the same `ArtHOI` environment without creating annoying separate environments. It can be done by installing them with `--no-deps` and adding missing dependencies to the environment manually. Moreover, ask a coding agent to do this is also a good option, but **please make sure not to break any existing packages while doing this** while you prompting them.

Such process is easily to broke `numpy` version, so please double-check and make sure to use `numpy<2` (recommended) or you're aware `numpy>=2` is compatible with all the other packages in the environment.

> It is known that `unidepth` requires `numpy>=2`, but it is basically ok to use `1.26`.

## 4. Download model weights

- SAM2 & CoTracker & DINO & UniDepthV2: weights are automatically downloaded on first use by the respective libraries.
- MANO: download the `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` files from the official MANO website and place them in `third_party/body_models/mano/` or the path specified by `MANO_MODEL_PATH` in the config.
- DiffuEraser:
- PartField:
- Video Depth Anything: 
- WiLoR (or HaMeR): 

Place the downloaded weights in the respective directories according to the official instructions.
