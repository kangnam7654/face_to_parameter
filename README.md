# Face to Parameter

[한국어](README_ko.md)

A PyTorch proof-of-concept for mapping face images to a controllable parameter vector: a generator (Imitator) renders an image from a parameter vector, and a Translator is meant to predict that vector from a face image.

## What actually works in this repository

- **Works:** the one public pretrained checkpoint shipped here, `weigths/imitator.pt`, is a **legacy 37-D generator**. With `infer_legacy.py` it turns a 37-D parameter vector into a 256x256 RGB image on CPU.
- **Not included:** no pretrained Translator is shipped. Because of that, the full *face image → parameters → character* pipeline cannot be reproduced from this repository. This is not a game-engine integration, and nothing here connects to an engine.
- `weigths/style_transfer.pt` is **not** Translator weights. Whether it is compatible with `models/animegan.py` has not been verified here.
- No dataset, example faces or generated images are included. Rights for public portrait imagery or training data are not established, so none are distributed.

### Run the legacy generator (synthetic input)

```bash
python infer_legacy.py \
  --checkpoint weigths/imitator.pt \
  --seed 0 \
  --output /tmp/legacy_imitator_seed0.png
```

- **Input:** no face image is read. The input is a **synthetic** 37-D vector, sampled uniformly from `[0, 1)` with a seeded CPU `torch.Generator`. The same seed always gives the same vector. The meaning and intended range of the 37 parameters are not documented in this repository, so the vector is only a probe.
- **Output:** exactly one 256x256 RGB PNG, written to the path passed in `--output`. That flag is required. The path must end in `.png` and its parent directory must already exist. The script won't overwrite an existing file unless you pass `--overwrite`. Write the file outside the checkout, and do not commit generated images.
- **Safety:** the checkpoint is loaded with `torch.load(..., map_location="cpu", weights_only=True)` and checked against the exact 12-key legacy schema (key names, shapes, dtypes, finite values). It is then loaded with `strict=True`. An incompatible file is reported as an error and is never partially loaded. Before saving, the script checks the output's shape, that every value is finite, and that values fall within the Tanh range `[-1, 1]`. The script does not download anything, collect telemetry or call any external service.
- **Seed:** an integer in `[0, 2**32 - 1]`.

### Two generator architectures (they are separate)

| | Legacy (`models/legacy_imitator.py`) | Current (`models/imitator.py`) |
|---|---|---|
| Input | 37-D vector | 960-D label vector |
| Output | 256x256 | 512x512 |
| Structure | ConvTranspose 4x4 (BatchNorm, LeakyReLU) → 5 × bilinear-upsample Conv blocks (InstanceNorm, LeakyReLU) → upsample Conv + Tanh; channels 512, 512, 512, 256, 128, 64, 3 | MLP conditioner + FiLM-modulated upsampling blocks, trained against `ProjectionDiscriminator` with EMA |
| Pretrained weights here | `weigths/imitator.pt` | none |
| Used by | `infer_legacy.py`, `train.py` (Stage 2) | `train_imitator.py` (Stage 1) |

The legacy checkpoint does **not** load into the current `Imitator`, and this repository does not pretend that it does. The current 960-D/512px generator is a separate training architecture. To use it, you have to train it yourself with your own data.

## Training architecture (requires your own data)

The training code is organised in two stages. No trained weights are provided for the *current* 960-D generator or the Translator, and training either requires a dataset that you supply. The separate legacy generator checkpoint above is the only runnable trained model here.

1. **Stage 1 - Imitator Training** (`train_imitator.py`): a conditional GAN (the current `Imitator` generator + `ProjectionDiscriminator`) is trained to generate 512x512 images from a 960-dimensional label vector, using FiLM conditioning and EMA.
2. **Stage 2 - Translator Training** (`train.py`): a `Translator` (InceptionResnetV1 backbone from `facenet-pytorch`) predicts 37-D parameters from face images. It is trained through a frozen 37-D generator and a style-transfer model with a loopback loss. `train.py` uses the **legacy** 37-D generator and loads `--weight_imitator` with the same strict `weights_only=True` CPU loader. This repository has not demonstrated that Stage 2 training works: it needs a face dataset, and no trained Translator exists here.

**Intended flow (not reproducible here without a trained Translator)**: Face Image → Translator → Parameter Vector → Imitator → Generated Image

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- CPU is enough for `infer_legacy.py`; a CUDA 12.1 compatible GPU is recommended for training

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/kangnam7654/face_to_parameter.git
    cd face_to_parameter
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    For the legacy CPU inference command and the tests, the parent verified this small environment on Python 3.11 (no training dependencies needed):
    ```bash
    python -m pip install 'torch==2.5.1' 'numpy==1.26.4' 'pillow==10.4.0'
    ```
    For the separate training code on a supported CUDA/Linux setup, install the project's full pinned requirements using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

Prepare your dataset using the utility scripts:

```bash
# Step 1: Detect and align faces
python utils/face_align.py --root_dir /path/to/images --out_dir /path/to/aligned --resolution 512

# Step 2: Generate embedding labels
python utils/make_parameter.py --image_dir /path/to/aligned --save_dir /path/to/labels
```

Then create a Parquet file mapping images to labels using `utils/make_pair_parquet.py`.

### 2. Training the Imitator (Stage 1)

Train the generator and discriminator in a GAN setup:

```bash
python train_imitator.py \
  --csv_or_parquet /path/to/pairs.parquet \
  --iteration 1000000 \
  --lr 1e-3 \
  --batch_size 16 \
  --device cuda:0
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv_or_parquet` | (required) | Path to dataset Parquet/CSV file |
| `--checkpoint_path` | `None` | Path to checkpoint to resume training |
| `--iteration` | `1000000` | Number of training steps |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `16` | Batch size |
| `--device` | `mps` | Device (`cuda:0`, `mps`, `cpu`) |

### 3. Training the Translator (Stage 2)

This stage trains the Translator through the legacy 37-D generator. You need your own face dataset under `--root_dir`. Building `Translator` makes `facenet-pytorch` fetch its pretrained InceptionResnetV1 weights. Running this command has not been shown to produce a working Translator. The tracked `style_transfer.pt` has not been verified against `models/animegan.py`.

```bash
python train.py \
  --root_dir ./data \
  --lr 1e-4 \
  --resolution 256 \
  --max_steps 10000000 \
  --batch_size 2 \
  --w_idt 1 \
  --w_loop 1 \
  --weight_imitator ./weigths/imitator.pt \
  --weight_style_transfer ./weigths/style_transfer.pt
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | `./data` | Root directory for dataset |
| `--lr` | `1e-4` | Learning rate |
| `--resolution` | `256` | Input image resolution |
| `--max_steps` | `10000000` | Maximum training steps |
| `--batch_size` | `2` | Batch size |
| `--w_idt` | `1` | Identity loss weight |
| `--w_loop` | `1` | Loopback loss weight |
| `--weight_imitator` | `None` | Legacy 37-D Imitator plain state dict (`.pt`), strictly validated |
| `--weight_style_transfer` | `None` | Pre-trained style transfer weights (`.pt`) |

### 4. Running Tests

```bash
python -m unittest discover tests
```

`tests/test_legacy_imitator.py` covers strict checkpoint compatibility, output shape, determinism and invalid-input errors. If `weigths/imitator.pt` is present, these tests also load it.

## Directory Structure

```
.
├── datamodules/          # PyTorch Dataset classes
│   └── simple_datamodule.py
├── models/               # Neural network architectures
│   ├── imitator.py       # Current 960-D/512px generator (Imitator) + ProjectionDiscriminator
│   ├── legacy_imitator.py # Legacy 37-D/256px generator + strict checkpoint loader
│   ├── translator.py     # Face encoder (InceptionResnetV1 backbone)
│   └── animegan.py       # AnimeGAN style transfer generator
├── pipelines/            # PyTorch Lightning training modules
│   ├── imitator_pipeline.py  # GAN training loop with EMA
│   └── pipeline.py           # Translator training with loopback loss
├── tests/                # Unit tests
│   ├── test_models.py
│   └── test_legacy_imitator.py
├── weigths/              # imitator.pt (legacy 37-D), style_transfer.pt
├── utils/                # Data preparation utilities
│   ├── face_align.py         # MTCNN face detection and alignment
│   ├── make_parameter.py     # MobileNet-V3 embedding generation
│   ├── make_pair_parquet.py  # Create image-label pair datasets
│   ├── concat_tensor_images.py
│   └── move_samples.py
├── infer_legacy.py       # Legacy checkpoint CLI (synthetic 37-D input -> one PNG)
├── train_imitator.py     # Stage 1 training script
├── train.py              # Stage 2 training script
├── pyproject.toml        # Project metadata and dependencies
└── requirements.txt      # Frozen dependency versions
```
