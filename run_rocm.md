# Running Reflex on ROCm (AMD GPU)

This document describes how to run the Reflex project on AMD GPUs using ROCm.

## Prerequisites

- AMD GPU with ROCm support (tested on AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, gfx1151)
- Docker installed
- ROCm PyTorch container: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`

## Quick Start (Pre-built Image)

### Build the Docker image once:

```bash
cd /home/phil/devel/AI/reflex
docker build -t reflex-rocm .
```

### Run training:

```bash
docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  -v /home/phil/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd):/workspace \
  reflex-rocm bash -c "python -m reflex.train"
```

### Run demo (interactive mode):

```bash
docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  -v /home/phil/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd):/workspace \
  reflex-rocm bash

# Inside container:
python -m reflex.demo -i
```

## Model Cache

The Qwen2.5-Coder-1.5B-Instruct backbone (~3GB) is downloaded automatically on first run and cached in `/home/phil/.cache/huggingface`. To pre-download it manually:

```bash
mkdir -p /home/phil/.cache/huggingface/hub/models--Qwen--Qwen2_5-Coder-1_5B-Instruct
cd /home/phil/.cache/huggingface/hub/models--Qwen--Qwen2_5-Coder-1_5B-Instruct
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct .
```

## Training Output

Training runs for 80,000 steps and saves weights to `weights.pth` when accuracy reaches 99.99%. Progress is printed every 250 steps:

```
step     0  ε=1.00  loss=11.1534  tf_acc=19.4%  inf_acc=16.9%
step   250  ε=1.00  loss=0.1934  tf_acc=96.4%  inf_acc=95.2%
...
```

## Demo Commands

- `python -m reflex.demo` - Run preset test cases
- `python -m reflex.demo -i` - Interactive mode (type instructions)

### Example Instructions:

- Sprites: `draw a smiley`, `draw a heart`, `draw a circle`
- Digits: `draw digit 7 at position 15 10`, `draw digits A and B`
- Math: `3 + 5`, `compute 4 plus 6 and draw result`

## Monitoring GPU Usage

Check ROCm GPU utilization:

```bash
rocm-smi
# or from Docker:
docker run --rm --device /dev/kfd rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0 rocm-smi
```

## Troubleshooting

### GPU not detected:
Ensure `/dev/kfd` and `/dev/dri` devices are accessible to Docker.

### Training stuck after model load:
Check for zombie training processes:
```bash
ps aux | grep reflex.train
pkill -f "reflex.train"  # Kill if needed
```

### Out of memory:
Reduce batch size in `train.py` (line ~350): `batch_size = min(16, n)`
