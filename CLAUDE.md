# CLAUDE.md

## Project Overview

**Face to Parameter** is a deep learning project that translates human face images into controllable latent parameter vectors. It uses a two-stage generative pipeline built with PyTorch and PyTorch Lightning:

1. **Stage 1 - Imitator Training**: A conditional GAN (Generator + ProjectionDiscriminator) learns to generate 512x512 face images from a latent vector (960-dim).
2. **Stage 2 - Translator Training**: A Translator model (InceptionResnetV1 backbone) predicts latent parameters from real face images, using the frozen Imitator and a style transfer model for loopback loss.

Core flow: **Face Image → Translator → Latent Vector → Imitator → Generated Face**

## Directory Structure

```
face_to_parameter/
├── datamodules/                  # PyTorch Dataset classes
│   └── simple_datamodule.py      # CSV/Parquet-based image-label dataset
├── models/                       # Neural network architectures
│   ├── imitator.py               # Generator (Imitator) + ProjectionDiscriminator
│   ├── translator.py             # Face encoder with InceptionResnetV1 backbone
│   └── animegan.py               # AnimeGAN-style transfer generator
├── pipelines/                    # PyTorch Lightning training modules
│   ├── imitator_pipeline.py      # GAN training loop with EMA
│   └── pipeline.py               # Translator training with style transfer + loopback
├── tests/                        # Unit tests
│   └── test_models.py            # Model shape verification tests
├── utils/                        # Data preparation and utility scripts
│   ├── face_align.py             # MTCNN face detection and alignment
│   ├── make_parameter.py         # MobileNet-V3 embedding generation
│   ├── make_pair_parquet.py      # Create image-label pair datasets
│   ├── concat_tensor_images.py   # Image grid visualization
│   └── move_samples.py           # Data sampling and resizing
├── weigths/                      # Pre-trained model weights (gitignored)
├── train_imitator.py             # CLI: Stage 1 Imitator training
├── train.py                      # CLI: Stage 2 Translator training
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Frozen dependency versions
└── README.md                     # User-facing documentation
```

## Build and Environment

- **Python**: 3.11+
- **Package manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **GPU**: CUDA 12.1 (nvidia libraries in requirements.txt)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| torch (>=2.2.2) | Core deep learning framework |
| lightning (>=2.5.4) | Training framework (PyTorch Lightning) |
| facenet-pytorch (>=2.6.0) | Pre-trained InceptionResnetV1 |
| opencv-python (>=4.11.0.86) | Image loading and processing |
| polars (>=1.32.3) | DataFrame operations for dataset files |
| wandb (>=0.21.3) | Experiment tracking |
| tqdm (>=4.67.1) | Progress bars |

## Running Tests

```bash
python -m unittest discover tests
```

Tests use Python's built-in `unittest` framework. Currently covers model forward-pass shape validation.

## Training Commands

### Stage 1 - Imitator (GAN)

```bash
python train_imitator.py \
  --csv_or_parquet /path/to/pairs.parquet \
  --iteration 1000000 \
  --lr 1e-3 \
  --batch_size 16 \
  --device cuda:0
```

Optional: `--checkpoint_path` to resume from a checkpoint.

### Stage 2 - Translator

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

### Data Preparation Pipeline

1. **Face alignment**: `python utils/face_align.py --root_dir <images> --out_dir <aligned> --resolution 512`
2. **Embedding generation**: `python utils/make_parameter.py --image_dir <images> --save_dir <embeddings>`
3. **Dataset pairing**: Use `utils/make_pair_parquet.py` to create Parquet files mapping images to `.npy` label files.

## Code Conventions

### Style

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Type hints**: Minimal usage; PEP 604 union syntax (`X | Y`) where present
- **Imports**: Standard library, then third-party, then local modules
- **Image range**: Normalized to `[-1, 1]` (mean=0.5, std=0.5)
- **Image format**: OpenCV BGR loaded, converted to RGB for model input
- **Default resolution**: 512x512

### Architecture Patterns

- Models inherit from `torch.nn.Module`
- Training pipelines inherit from `pytorch_lightning.LightningModule`
- GAN training uses manual optimization (`self.automatic_optimization = False` in `imitator_pipeline.py`)
- EMA decay (0.999) applied to generator weights during Imitator training
- Spectral normalization used on discriminator layers
- FiLM (Feature-wise Linear Modulation) for conditional generation
- Pre-trained backbone (InceptionResnetV1) is frozen during Translator training

### Experiment Tracking

- Weights & Biases (`wandb`) used for logging during Imitator training
- PyTorch Lightning's built-in logging for Translator training
- Training images saved periodically to `log_images/` or `--save_dir`

## Key Model Dimensions

| Model | Input | Output |
|-------|-------|--------|
| Imitator | (batch, 960) latent vector | (batch, 3, 512, 512) image |
| ProjectionDiscriminator | (batch, 3, 512, 512) image + (batch, 960) condition | scalar logit |
| Translator | (batch, 3, 160, 160) face image | (batch, 37) parameter vector |
| Style Transfer (Generator) | (batch, 3, H, W) image | (batch, 3, H, W) stylized image |

Note: The latent dimension varies by use case (960 for Imitator standalone training, 37 for Translator stage).

## Files to Never Commit

Per `.gitignore`, these are excluded from version control:
- `.venv/` - Virtual environments
- `__pycache__/`, `*.pyc` - Python cache
- `lightning_logs/`, `logs/`, `wandb/`, `log_images/` - Training artifacts
- `data/`, `*.parquet` - Dataset files
- `checkpoints/`, `weigths/`, `*.pt` - Model weights
- `playground/`, `playground.py` - Scratch files
- `*.sh` - Shell scripts
