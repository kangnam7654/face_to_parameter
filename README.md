# Face to Parameter

[한국어](README_ko.md)

A deep learning project that translates human face images into controllable latent parameter vectors. The system first generates realistic face images from a latent vector (Imitator), then learns to translate real face images back into that latent vector space (Translator).

Built with PyTorch and PyTorch Lightning.

## Architecture

The project follows a two-stage training process:

1. **Stage 1 - Imitator Training**: A conditional GAN (`Imitator` generator + `ProjectionDiscriminator`) learns to generate 512x512 face images from a 960-dimensional latent vector. Uses FiLM conditioning and EMA for stable training.
2. **Stage 2 - Translator Training**: A `Translator` model (pre-trained InceptionResnetV1 backbone) predicts latent parameters from real face images. Uses the frozen Imitator and a style transfer model for loopback loss.

**Core flow**: Face Image → Translator → Latent Vector → Imitator → Generated Face

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- CUDA 12.1 compatible GPU

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
    Using `uv` (recommended):
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

Train the Translator model using pre-trained Imitator and style transfer weights:

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
| `--weight_imitator` | `None` | Pre-trained Imitator weights (`.pt`) |
| `--weight_style_transfer` | `None` | Pre-trained style transfer weights (`.pt`) |

### 4. Running Tests

```bash
python -m unittest discover tests
```

## Directory Structure

```
.
├── datamodules/          # PyTorch Dataset classes
│   └── simple_datamodule.py
├── models/               # Neural network architectures
│   ├── imitator.py       # Generator (Imitator) + ProjectionDiscriminator
│   ├── translator.py     # Face encoder (InceptionResnetV1 backbone)
│   └── animegan.py       # AnimeGAN style transfer generator
├── pipelines/            # PyTorch Lightning training modules
│   ├── imitator_pipeline.py  # GAN training loop with EMA
│   └── pipeline.py           # Translator training with loopback loss
├── tests/                # Unit tests
│   └── test_models.py
├── utils/                # Data preparation utilities
│   ├── face_align.py         # MTCNN face detection and alignment
│   ├── make_parameter.py     # MobileNet-V3 embedding generation
│   ├── make_pair_parquet.py  # Create image-label pair datasets
│   ├── concat_tensor_images.py
│   └── move_samples.py
├── train_imitator.py     # Stage 1 training script
├── train.py              # Stage 2 training script
├── pyproject.toml        # Project metadata and dependencies
└── requirements.txt      # Frozen dependency versions
```
