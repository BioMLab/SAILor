# SAILor

SAILor is a multimodal fusion system for lncRNA subcellular localization prediction. It integrates multiple sources of information, including sequence, secondary structure, CGR, and RPI, and performs final classification through supervised warmup plus PPO fine-tuning.

## Introduction

The core idea of this project is to encode each modality separately and then fuse them through policy-guided bottleneck fusion. For paper readers, it can be understood as a multimodal modeling framework for lncRNA subcellular localization.

## Environment Setup

We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate amploc
```

If you prefer pip, you can create a virtual environment first and then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your machine uses a specific CUDA version, you may need to adjust the PyTorch and PyG dependencies to match your local environment.

## Data Preparation

If you already have the raw data and the required pretrained resources, you can skip this step. If you need to prepare the derived data from scratch, run the following commands in order.

```bash
# 1. Generate secondary-structure files and produce the .dbn intermediates
python scripts/data_prep/preprocess_structures.py --input_csv data/rna_data.csv --output_dir data/processed_structures_linearfold

# 2. Aggregate the .dbn files into a CSV. Training reads the CSV, not the raw .dbn files
python scripts/data_prep/generate_structure_csv.py

# 3. Compute CGR features. This script only uses torchvision's ResNet50 ImageNet weights and does not depend on DNABERT-2, RNA-FM, or Mamba
python scripts/data_prep/preprocess_cgr_features.py --csv_path data/rna_data.csv --output_dir data/cgr_features
```

### Pretrained Resources

#### 1. Mamba

Mamba is a Python package dependency and is installed through `requirements.txt`. It does not require downloading a separate file into `pretrained/`.

If `pip install -r requirements.txt` fails while building Mamba on your machine, install it manually with no build isolation and then rerun the requirements installation:

```bash
pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install "mamba-ssm>=1.2.0" --no-build-isolation
pip install -r requirements.txt
```

If you prefer to install Mamba from source:

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install . --no-build-isolation
```

#### 2. DNABERT-2-117M

We recommend downloading it directly from Hugging Face into the local directory instead of using `scripts/pretrained/download_helper.py` or `scripts/pretrained/final_downloader.py`:

```bash
huggingface-cli download zhihan1996/DNABERT-2-117M \
  --local-dir pretrained/DNABERT-2-117M \
  --local-dir-use-symlinks False
```

#### 3. RNA-FM

The RNA-FM Python package is installed through `requirements.txt`. Download only the model weight file into the local pretrained directory:

```bash
huggingface-cli download cuhkaih/rnafm RNA-FM_pretrained.pth \
  --local-dir pretrained/RNA-FM \
  --local-dir-use-symlinks False
```

If you want to warm up the torchvision ResNet50 weights in the local cache first, you can run this before the first CGR preprocessing step:

```bash
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"
```

- The `.dbn` files are intermediate products. The actual data used for training is the `id` and `dbn_string` stored in `data/structures.csv`.
- `pretrained/` is no longer shipped with the repository, so readers need to download DNABERT-2-117M and RNA-FM using the commands above.
- `mamba/` is also no longer provided in the repository. The Mamba package dependency is managed by `requirements.txt`; use the manual commands above only if the normal installation fails on your machine.
- The CGR preprocessing script depends only on torchvision's ResNet50 pretrained weights, not on DNABERT-2-117M, RNA-FM, or Mamba.

You can adjust the input and output paths in these commands to match your local directory layout.

## Run Commands

```bash
python main_run.py --config configs/main_config.yaml
```

If you want to specify an output directory:

```bash
python main_run.py --config configs/main_config.yaml --output_dir outputs/my_run
```

If you want to pin a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python main_run.py --config configs/main_config.yaml
```
