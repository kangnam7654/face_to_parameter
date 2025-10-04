# Face to Parameter

This repository contains a deep learning project designed to translate human face images into a set of controllable parameters. The core idea is to create a system that can first generate a realistic face image based on a latent vector (Imitator) and then translate a given face image into that latent vector space (Translator).

This project is structured using PyTorch and PyTorch Lightning for robust and scalable training pipelines.

## Architecture

The project follows a two-stage training process:

1.  **Imitator & Style Transfer Training**: The first stage involves training an `Imitator` model. This model is a Generator in a GAN setup, which learns to generate face images from a latent parameter vector. It is trained adversarially against a `ProjectionDiscriminator`.
2.  **Translator Training**: The second stage trains a `Translator` model. This model takes a real face image as input and predicts the corresponding latent parameter vector that can be used by the `Imitator`. It uses a pre-trained InceptionResnetV1 for feature extraction.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/face_to_parameter.git
    cd face_to_parameter
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Using `uv` (recommended):
    ```bash
    uv pip install -r requirements.txt
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be generated from `pyproject.toml` in the next steps.)*

## Usage

### 1. Data Preparation

This project requires a dataset of face images and corresponding parameter labels. The `SimpleDatamodule` expects a CSV or Parquet file that maps image file paths to their label file paths.

You will need to prepare your data accordingly, for example, by creating a `pairs.parquet` file.

### 2. Training the Imitator

The `train_imitator.py` script is used to train the generator and discriminator models.

```bash
.venv/bin/python train_imitator.py --csv_or_parquet /path/to/your/pairs.parquet
```

**Optional Arguments:**

- `--checkpoint_path`: Path to a checkpoint file to resume training. If not provided, the model starts from scratch.
- `--lr`: Learning rate (default: `1e-3`).
- `--batch_size`: Batch size (default: `16`).

### 3. Running Tests

To ensure the models and components are working correctly, you can run the unit tests.

```bash
.venv/bin/python -m unittest discover tests
```

## Directory Structure

```
.
├── datamodules/      # PyTorch Lightning DataModules
├── models/           # Model definitions (Imitator, Translator)
├── pipelines/        # PyTorch Lightning training pipelines
├── tests/            # Unit and integration tests
├── utils/            # Utility scripts for data processing
├── train_imitator.py # Training script for the Imitator model
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```
