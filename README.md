# diffusion_WSI
A research pipeline for preprocessing histopathology whole-slide images (WSIs), training a denoising diffusion U-Net, generating synthetic histopathology images from PyTorch Lightning checkpoints, and evaluating generated images with Fréchet Inception Distance (FID).

The repository was developed around TCGA-BRCA slides, but the main preprocessing and diffusion components can be reused with other WSI datasets supported by TIAToolbox.

Overview

The repository covers five stages:

WSI preprocessing

Read slide paths from a CSV file.

Detect tissue using Otsu thresholding and morphological operations.

Identify, merge, or split connected tissue components.

Export either high-resolution tiles or slide-level square representations.

Save corresponding binary tissue masks.

Diffusion training

Train an attention-based or attention-free 2D U-Net to predict diffusion noise.

Use linear or cosine diffusion schedules.

Run hyperparameter sweeps through Weights & Biases.

Save CSV logs and PyTorch Lightning checkpoints.

Image generation

Load one checkpoint or all checkpoints in a directory.

Generate images with DDPM or DDIM sampling.

Optionally save preview grids.

Performance profiling

Measure sampling time, RAM usage, and GPU memory consumption.

Save step-level and optional microbatch-level profiling results.

FID evaluation

Compare real and generated image distributions.

Evaluate one model or several checkpoint-specific output directories.

Estimate a real-versus-real split baseline.

Plot FID against sample size or training checkpoint step.

Repository structure

.
├── code/
│   ├── preprocessing/
│   │   ├── create_tiles_tia.py
│   │   ├── create_thumbnails_tia.py
│   │   ├── create_thumbnails_tia_fixedCanvas.py
│   │   ├── create_thumbnails.py
│   │   ├── check_components_size_distribution.py
│   │   ├── thumbnail_test.py
│   │   └── qc_exploration.ipynb
│   ├── train_UNet/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── callbacks.py
│   │   └── train_UNet_sweep.py
│   └── generation/
│       ├── generate_from_ckpt.py
│       ├── time_and_memory_test.py
│       └── compare_fid.py
├── logs/
├── sbatch_WSIpreprocessing.sh
└── README.md

Requirements

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for training and generation.

Install PyTorch and torchvision using the build appropriate for your CUDA environment, then install the remaining core dependencies:

python -m venv .venv
source .venv/bin/activate

# Install torch and torchvision for your platform first.

pip install \
  lightning \
  torchmetrics \
  wandb \
  numpy \
  pandas \
  pillow \
  scipy \
  scikit-image \
  matplotlib \
  tiatoolbox \
  psutil \
  pytorch-fid

Some WSI formats, including SVS, may require OpenSlide and its native system libraries. Refer to the TIAToolbox installation instructions for platform-specific setup.

code/train_UNet/external_func.py contains legacy utilities that are not used by the main diffusion pipeline. Importing that module additionally requires packages such as scikit-learn, joblib, SimpleITK, nibabel, and tabpfn.

Weights & Biases is used by the training script:

wandb login

Data preparation

Input CSV

The preprocessing scripts expect a CSV file with a column named path. Each row should contain the path to a WSI:

path
/data/wsi/patient_001/slide_001.svs
/data/wsi/patient_002/slide_002.svs

Absolute paths are the safest option, particularly when preprocessing is run through Slurm or inside a container.

Recommended option: high-resolution tissue tiles

create_tiles_tia.py is the most portable preprocessing entry point because its parameters are exposed through the command line.

From the repository root:

python code/preprocessing/create_tiles_tia.py \
  --csv-path /path/to/slides.csv \
  --output-root-img /path/to/preprocessed/images \
  --output-root-mask /path/to/preprocessed/masks \
  --input-root-for-structure /path/to/wsi/root \
  --target-size 256 \
  --thumbnail-mpp 8.0 \
  --tile-mpp 0.5 \
  --min-tissue-frac 0.50 \
  --pad 20 \
  --min-area-ratio 0.01 \
  --merge-radius-px 20 \
  --use-8-connectivity \
  --split-area-frac 0.25 \
  --density-single-thresh 0.5 \
  --small-component-policy union

The script first detects tissue components on a low-resolution thumbnail and then reads square tiles directly from the WSI at the requested --tile-mpp. Tiles with a tissue fraction below --min-tissue-frac are discarded.

Example output filenames:

slide_001_part01_x000_y000_256x256_tilempp0p5_tissue50.png
slide_001_part01_x000_y000_256x256_tilempp0p5_tissue50_mask.png

When --input-root-for-structure is provided, the relative directory structure below that root is preserved in the image and mask output folders.

Slide-level square representations

The repository also provides several alternatives for producing one or more square images per slide:

create_thumbnails_tia_fixedCanvas.py: places selected tissue crops on a common square canvas before resizing. This preserves a more consistent scale across slides.

create_thumbnails_tia.py: crops selected tissue components and letterboxes each crop to the requested square output size.

create_thumbnails.py: uses WSI pyramid levels to create 512 × 512 component representations.

check_components_size_distribution.py: analyses selected component dimensions and exports CSV summaries and histograms, which can help choose a fixed canvas size.

These scripts currently define input paths, output paths, and preprocessing parameters in their if __name__ == "__main__" blocks. Edit those values before running them:

python code/preprocessing/create_thumbnails_tia_fixedCanvas.py

Tissue-component logic

The preprocessing pipeline uses the following general procedure:

Generate a thumbnail at the requested MPP.

Convert the thumbnail to grayscale.

Apply Otsu thresholding to separate tissue from background.

Fill holes and optionally apply morphological closing to merge nearby fragments.

Remove components below a relative area threshold.

Either:

split multiple large components,

retain the largest component, or

use the union of retained components.

Export the selected image regions and binary masks.

Slides without usable MPP metadata may produce warnings or fail during MPP-based extraction. Review the preprocessing logs and verify image scale before training.

Training

Training data layout

The data module recursively loads .png, .jpg, and .jpeg files below DATA_ROOT:

/path/to/preprocessed/images/
├── slide_001_part01.png
├── slide_001_part02.png
└── nested_folder/
    └── slide_002_part01.png

Important: point DATA_ROOT to the RGB image directory only. Do not point it to a common parent containing the mask directory, because mask images would also be treated as training samples.

Images are converted to RGB, resized to the configured square size, augmented during training, and normalized from [0, 1] to [-1, 1].

The current data module is train-only. Validation and test loaders are intentionally empty.

Experiment configuration

train_UNet_sweep.py expects an experiment name and loads:

<config_dir>/<experiment_name>_config.json

The configuration directory is currently hard-coded near the top of code/train_UNet/train_UNet_sweep.py:

config_dir = "/storage/DSH/projects/iaso/diffusion_wholewsi/logs/config_files"

Change this path to a location available in your environment.

Example exp1_config.json:

{
  "path": {
    "DATA_ROOT": "/path/to/preprocessed/images",
    "LOGS": "/path/to/training/logs"
  },
  "data": {
    "IMAGE_SIZE": 256,
    "CHANNELS": 3,
    "AUGMENT_TRAIN": true
  },
  "model": {
    "DIM": 64,
    "DIM_MULTS": [1, 2, 4, 8],
    "COND_DROP_PROB": 0.5,
    "RESNET_BLOCK_GROUPS": 8,
    "LEARNED_SINUSOIDAL_COND": false,
    "RANDOM_FOURIER": false,
    "LEARNED_SINUSOIDAL_DIM": 16,
    "USE_ATTENTION": true,
    "NUM_CLASSES": 1
  },
  "diffusion": {
    "NUM_TIMESTEPS": 1000,
    "SCHEDULE": "linear",
    "BETA_START": 0.0001,
    "BETA_END": 0.02,
    "COSINE_S": 0.008,
    "SAMPLE_STEPS": 50
  },
  "training": {
    "LEARNING_RATE": [0.0001],
    "BATCH_SIZE": [16],
    "WEIGHT_DECAY": [0.0],
    "ACCUMULATE_GRAD_BATCHES": [8],
    "NUM_EPOCHS": 100,
    "MONITOR_METRIC": "train_loss_epoch",
    "MONITOR_MODE": "min",
    "FAST_DEV_RUN": false,
    "SWEEP_METHOD": "grid",
    "SWEEP_CAP": 20,
    "SPLIT_SEED": 1234
  },
  "computation": {
    "NUM_WORKERS": 8,
    "ACCELERATOR": "gpu",
    "DEVICES": 1,
    "PRECISION": "16-mixed"
  },
  "logging": {
    "WANDB_PROJECT": "diffusion_unet"
  }
}

Configuration values represented as lists are used to construct the W&B sweep. A single value in each list produces one hyperparameter combination.

Start a training sweep

The training script uses local imports, so run it from code/train_UNet:

cd code/train_UNet
python train_UNet_sweep.py exp1

Training outputs are written below the configured LOGS directory:

LOGS/
├── csv_logs/
└── model_logs/
    └── <run_name>/
        └── checkpoints/
            ├── <best_checkpoint>.ckpt
            └── <step_checkpoint>.ckpt

The run name records the experiment, learning rate, batch size, weight decay, U-Net dimension, number of diffusion steps, and schedule.

Current training behavior

The model predicts the Gaussian noise added to each clean image and is optimized with mean squared error.

Linear and cosine beta schedules are supported.

The U-Net can be trained with attention (USE_ATTENTION=true) or without attention.

Class-conditioning components are implemented, but the current training script passes class_map=None, so the default workflow is unconditional.

Sample grids are configured to be generated every 1,000 epochs using full DDPM sampling.

Checkpoints are additionally saved every 10,000 training steps.

accumulate_grad_batches is currently fixed to 8 in the Lightning trainer. The ACCUMULATE_GRAD_BATCHES sweep value contributes to the sweep size but is not passed to the trainer; edit the script if different accumulation values are required.

Generate synthetic images

The generation modules import packages from code/. From the repository root, add that directory to PYTHONPATH.

One checkpoint

PYTHONPATH=code python -m generation.generate_from_ckpt \
  --exp exp1 \
  --ckpt /path/to/model.ckpt \
  --outdir results/generated \
  --num_samples 64 \
  --batch_size 8 \
  --image_size 256 \
  --channels 3 \
  --num_timesteps 1000 \
  --schedule linear \
  --beta_start 0.0001 \
  --beta_end 0.02 \
  --sampler ddim \
  --sample_steps 50 \
  --eta 0.0 \
  --attention true \
  --clip_denoised \
  --save_grid

All checkpoints in a directory

PYTHONPATH=code python -m generation.generate_from_ckpt \
  --exp exp1 \
  --ckpt_dir /path/to/checkpoints \
  --outdir results/generated \
  --num_samples 500 \
  --batch_size 8 \
  --image_size 256 \
  --num_timesteps 1000 \
  --schedule linear \
  --sampler ddim \
  --sample_steps 50 \
  --attention true

Generated images are stored by experiment and checkpoint:

results/generated/
└── exp1/
    └── <checkpoint_name>/
        ├── exp1_<checkpoint>_<step>_sample_000000.png
        ├── exp1_<checkpoint>_<step>_sample_000001.png
        └── ...

DDIM supports reduced-step sampling through --sample_steps; --eta 0.0 gives deterministic DDIM for a fixed initial noise tensor. DDPM always uses the full configured diffusion trajectory.

Checkpoint compatibility: --image_size, --channels, the diffusion schedule parameters, and --attention must match the settings used to train the checkpoint. The checkpoint stores the network hyperparameters, but the reverse diffusion schedule is reconstructed from the command-line arguments.

On Windows PowerShell, set the module path with:

$env:PYTHONPATH = "code"
python -m generation.generate_from_ckpt `
  --exp exp1 `
  --ckpt C:\path\to\model.ckpt `
  --outdir results\generated

Profile sampling time and memory

time_and_memory_test.py generates samples while recording wall-clock time, CPU RAM, and CUDA memory statistics.

PYTHONPATH=code python -m generation.time_and_memory_test \
  --exp exp1 \
  --ckpt /path/to/model.ckpt \
  --outdir results/profile \
  --num_samples 64 \
  --batch_size 8 \
  --image_size 256 \
  --num_timesteps 1000 \
  --schedule linear \
  --sampler ddim \
  --sample_steps 50 \
  --attention true \
  --use_autocast \
  --autocast_dtype float16 \
  --profile_every_batch \
  --save_grid

The profiler saves:

generated PNG images;

a step-level CSV;

an optional microbatch-level CSV;

a JSON summary containing timing and memory statistics;

an optional preview grid.

The profiler keeps the latent bank in CPU memory and streams microbatches through the GPU. Estimate the required host memory before requesting very large numbers of high-resolution samples.

Evaluate generated images with FID

compare_fid.py uses pytorch-fid Inception features. It supports a single generated-image directory or a parent directory containing one subdirectory per checkpoint.

Single FID estimate

PYTHONPATH=code python -m generation.compare_fid \
  --real_dir /path/to/real/images \
  --fake_dir results/generated/exp1/<checkpoint_name> \
  --outdir results/fid \
  --batch_size 16 \
  --num_workers 4 \
  --baseline_repeats 5 \
  --plot

FID as a function of sample size

PYTHONPATH=code python -m generation.compare_fid \
  --real_dir /path/to/real/images \
  --fake_dir results/generated/exp1 \
  --outdir results/fid \
  --sample_sizes 100,250,500,1000 \
  --num_repeats 5 \
  --baseline_repeats 5 \
  --batch_size 16 \
  --plot

The evaluator can:

calculate FID for real and generated images;

repeat subsampling at specified sample sizes;

compute a real-versus-real split baseline;

detect checkpoint steps from directory or checkpoint names;

save JSON results and diagnostic plots;

create group plots across checkpoints.

Use the same preprocessing, color representation, image size, and sampling strategy for real and generated images whenever possible. FID values are meaningful only when the two image collections are prepared consistently.

Slurm execution

sbatch_WSIpreprocessing.sh is an example FBK/Slurm job for WSI preprocessing. It contains site-specific values for:

node selection;

GPU, CPU, memory, time, and QoS requests;

Enroot container image;

mounted storage paths;

repository path;

log destinations;

preprocessing script selection.

Edit all of these values before submitting on another cluster:

sbatch sbatch_WSIpreprocessing.sh

The current preprocessing workload is CPU- and memory-intensive; a GPU is generally unnecessary unless required by the selected execution environment.

Reproducibility notes

Global Lightning seeding is set to 42 with worker seeding enabled.

Training is configured as deterministic in the Lightning trainer.

Data files are discovered recursively and sorted before shuffling.

The current workflow does not create a train/validation/test split.

W&B sweep metadata and CSV logs are stored separately.

Generation scripts infer training-step labels from common Lightning checkpoint naming patterns.

DDIM and profiling commands expose an explicit random seed where applicable.

Deterministic execution can still depend on the GPU, CUDA, PyTorch version, and operations used by the selected architecture.

Known environment-specific assumptions

Before using the repository outside its original environment, review the following:

Several preprocessing entry points contain absolute /storage/DSH/... paths.

train_UNet_sweep.py contains an absolute configuration directory.

The Slurm script references an FBK node, storage layout, and Enroot image.

No requirements.txt, Conda environment file, or container recipe is currently included.

No automated tests are currently included.

No license file is currently included.

Suggested end-to-end workflow

WSI files
   │
   ├── slides.csv
   │
   ▼
create_tiles_tia.py
   │
   ├── RGB image tiles ───────────────┐
   └── tissue masks                   │
                                      ▼
                              train_UNet_sweep.py
                                      │
                                      ├── W&B and CSV logs
                                      └── Lightning checkpoints
                                                   │
                                                   ▼
                                      generate_from_ckpt.py
                                                   │
                                                   ├── generated PNGs
                                                   ├── time_and_memory_test.py
                                                   └── compare_fid.py
