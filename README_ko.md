# Face to Parameter

[English](README.md)

사람의 얼굴 이미지를 제어 가능한 잠재 파라미터 벡터로 변환하는 딥러닝 프로젝트입니다. 잠재 벡터로부터 사실적인 얼굴 이미지를 생성(Imitator)하고, 실제 얼굴 이미지를 다시 해당 잠재 벡터 공간으로 변환(Translator)하는 시스템입니다.

PyTorch와 PyTorch Lightning을 기반으로 구축되었습니다.

## 아키텍처

2단계 학습 과정을 따릅니다:

1. **1단계 - Imitator 학습**: 조건부 GAN (`Imitator` 생성기 + `ProjectionDiscriminator`)이 960차원 잠재 벡터로부터 512x512 얼굴 이미지를 생성하도록 학습합니다. 안정적인 학습을 위해 FiLM 컨디셔닝과 EMA를 사용합니다.
2. **2단계 - Translator 학습**: `Translator` 모델(사전 학습된 InceptionResnetV1 백본)이 실제 얼굴 이미지로부터 잠재 파라미터를 예측합니다. 동결된 Imitator와 스타일 변환 모델을 사용하여 루프백 손실을 계산합니다.

**핵심 흐름**: 얼굴 이미지 → Translator → 잠재 벡터 → Imitator → 생성된 얼굴

## 시작하기

### 사전 요구 사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip
- CUDA 12.1 호환 GPU

### 설치

1. **저장소 클론:**
    ```bash
    git clone https://github.com/kangnam7654/face_to_parameter.git
    cd face_to_parameter
    ```

2. **가상 환경 생성:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **의존성 설치:**
    `uv` 사용 (권장):
    ```bash
    uv pip install -r requirements.txt
    ```
    또는 `pip` 사용:
    ```bash
    pip install -r requirements.txt
    ```

## 사용법

### 1. 데이터 준비

유틸리티 스크립트를 사용하여 데이터셋을 준비합니다:

```bash
# 1단계: 얼굴 감지 및 정렬
python utils/face_align.py --root_dir /path/to/images --out_dir /path/to/aligned --resolution 512

# 2단계: 임베딩 라벨 생성
python utils/make_parameter.py --image_dir /path/to/aligned --save_dir /path/to/labels
```

이후 `utils/make_pair_parquet.py`를 사용하여 이미지-라벨 매핑 Parquet 파일을 생성합니다.

### 2. Imitator 학습 (1단계)

GAN 구조로 생성기와 판별기를 학습합니다:

```bash
python train_imitator.py \
  --csv_or_parquet /path/to/pairs.parquet \
  --iteration 1000000 \
  --lr 1e-3 \
  --batch_size 16 \
  --device cuda:0
```

**인자:**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--csv_or_parquet` | (필수) | 데이터셋 Parquet/CSV 파일 경로 |
| `--checkpoint_path` | `None` | 학습 재개를 위한 체크포인트 경로 |
| `--iteration` | `1000000` | 학습 스텝 수 |
| `--lr` | `1e-3` | 학습률 |
| `--batch_size` | `16` | 배치 크기 |
| `--device` | `mps` | 디바이스 (`cuda:0`, `mps`, `cpu`) |

### 3. Translator 학습 (2단계)

사전 학습된 Imitator 및 스타일 변환 가중치를 사용하여 Translator 모델을 학습합니다:

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

**인자:**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--root_dir` | `./data` | 데이터셋 루트 디렉토리 |
| `--lr` | `1e-4` | 학습률 |
| `--resolution` | `256` | 입력 이미지 해상도 |
| `--max_steps` | `10000000` | 최대 학습 스텝 |
| `--batch_size` | `2` | 배치 크기 |
| `--w_idt` | `1` | Identity 손실 가중치 |
| `--w_loop` | `1` | 루프백 손실 가중치 |
| `--weight_imitator` | `None` | 사전 학습된 Imitator 가중치 (`.pt`) |
| `--weight_style_transfer` | `None` | 사전 학습된 스타일 변환 가중치 (`.pt`) |

### 4. 테스트 실행

```bash
python -m unittest discover tests
```

## 디렉토리 구조

```
.
├── datamodules/          # PyTorch Dataset 클래스
│   └── simple_datamodule.py
├── models/               # 신경망 아키텍처
│   ├── imitator.py       # 생성기 (Imitator) + ProjectionDiscriminator
│   ├── translator.py     # 얼굴 인코더 (InceptionResnetV1 백본)
│   └── animegan.py       # AnimeGAN 스타일 변환 생성기
├── pipelines/            # PyTorch Lightning 학습 모듈
│   ├── imitator_pipeline.py  # EMA 적용 GAN 학습 루프
│   └── pipeline.py           # 루프백 손실을 활용한 Translator 학습
├── tests/                # 단위 테스트
│   └── test_models.py
├── utils/                # 데이터 준비 유틸리티
│   ├── face_align.py         # MTCNN 얼굴 감지 및 정렬
│   ├── make_parameter.py     # MobileNet-V3 임베딩 생성
│   ├── make_pair_parquet.py  # 이미지-라벨 쌍 데이터셋 생성
│   ├── concat_tensor_images.py
│   └── move_samples.py
├── train_imitator.py     # 1단계 학습 스크립트
├── train.py              # 2단계 학습 스크립트
├── pyproject.toml        # 프로젝트 메타데이터 및 의존성
└── requirements.txt      # 고정된 의존성 버전
```
