# FIDNet

FIDNet is a frequency-guided diffusion model for image restoration. The repository contains the training and testing entry scripts together with the model and dataset code used by the project.

## Network Architecture

![FIDNet architecture](./fidnet_architecture.png)

## Environment

The dependency list below is written from the current codebase rather than copied from another project.

- Python 3.8 or newer
- PyTorch
- torchvision
- accelerate
- einops
- ema-pytorch
- pillow
- numpy
- opencv-python
- tqdm
- Augmentor
- lmdb

You can create an environment with `conda` or `venv`, then install the packages with `pip`.

Example:

```bash
pip install torch torchvision accelerate einops ema-pytorch pillow numpy opencv-python tqdm Augmentor lmdb
```

## Dataset Preparation

Before running the scripts, update the dataset paths in [train.py](/G:/FIDNet/train.py) and [test.py](/G:/FIDNet/test.py) to match your local data layout.

The current code uses paired paths for conditional restoration:

- training ground-truth images
- training degraded/input images
- validation or test output directory / paired test paths

Please make sure the paths assigned to `folder` in both scripts point to your own dataset folders.

## Training

Training only needs:

```bash
python train.py
```

Optional:

```bash
python train.py 10
```

The optional argument sets `sampling_timesteps`.

Notes:

- `train.py` currently fixes `CUDA_VISIBLE_DEVICES=0`
- the default training configuration uses `image_size=256`
- checkpoints and sampled results are saved under `./results`
- the script currently resumes from `trainer.load(60)`, so adjust or remove that line if you want to start from scratch

## Testing

Testing only needs:

```bash
python test.py
```

Notes:

- `test.py` also fixes `CUDA_VISIBLE_DEVICES=0`
- the script currently loads checkpoint `trainer.load(100)`
- test outputs are written to the folder set by `trainer.set_results_folder(...)`
- update the test dataset paths in `test.py` before running

## Repository Structure

```text
FIDNet/
|-- datasets1/
|-- src/
|-- train.py
|-- test.py
|-- fidnet_architecture.png
`-- README.md
```

## Acknowledgements

- The diffusion implementation is built on code derived from [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- The README layout style was organized with reference to [lhfgghc/LLVE-AMNet](https://github.com/lhfgghc/LLVE-AMNet), while the environment and usage instructions here were written according to this repository
