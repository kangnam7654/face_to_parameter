# Face to Parameter

[English](README.md)

얼굴 이미지를 제어 가능한 파라미터 벡터로 매핑하기 위한 PyTorch 개념 증명(PoC) 프로젝트입니다. 생성기(Imitator)는 파라미터 벡터로부터 이미지를 만들고, Translator는 얼굴 이미지로부터 그 벡터를 예측하는 것을 목표로 합니다.

## 이 저장소에서 실제로 동작하는 것

- **동작함:** 이 저장소에 포함된 유일한 공개 사전 학습 체크포인트 `weigths/imitator.pt`는 **레거시 37차원 생성기**입니다. `infer_legacy.py`를 사용하면 CPU에서 37차원 파라미터 벡터를 256x256 RGB 이미지로 변환합니다.
- **포함되지 않음:** 사전 학습된 Translator는 제공되지 않습니다. 따라서 *얼굴 이미지 → 파라미터 → 캐릭터* 전체 파이프라인은 이 저장소만으로 재현할 수 없습니다. 게임 엔진 연동이 아니며, 엔진과 연결되는 부분도 없습니다.
- `weigths/style_transfer.pt`는 Translator 가중치가 **아닙니다**. `models/animegan.py`와 호환되는지는 이 저장소에서 검증되지 않았습니다.
- 데이터셋, 예시 얼굴 이미지, 생성된 이미지는 포함되어 있지 않습니다. 공개 인물 이미지나 학습 데이터의 권리가 확인되지 않았으므로 배포하지 않습니다.

### 레거시 생성기 실행 (합성 입력)

```bash
python infer_legacy.py \
  --checkpoint weigths/imitator.pt \
  --seed 0 \
  --output /tmp/legacy_imitator_seed0.png
```

- **입력:** 얼굴 이미지는 읽지 않습니다. 입력은 **합성** 37차원 벡터이며, 시드가 고정된 CPU `torch.Generator`로 `[0, 1)` 구간에서 균등 샘플링합니다. 같은 시드는 항상 같은 벡터를 만듭니다. 37개 파라미터의 의미와 의도된 범위는 이 저장소에 문서화되어 있지 않으므로, 이 벡터는 동작 확인용일 뿐입니다.
- **출력:** 256x256 RGB PNG 한 장을 `--output`으로 지정한 경로에 저장합니다. 이 옵션은 필수입니다. 경로는 `.png`로 끝나야 하고 상위 디렉토리가 이미 존재해야 합니다. `--overwrite`를 주지 않으면 기존 파일을 덮어쓰지 않습니다. 파일은 저장소 밖에 저장하고, 생성된 이미지는 커밋하지 마세요.
- **안전성:** 체크포인트는 `torch.load(..., map_location="cpu", weights_only=True)`로 로드한 뒤 정확한 12개 키의 레거시 스키마(키 이름, shape, dtype, 유한값)와 대조하여 검사합니다. 그다음 `strict=True`로 로드합니다. 호환되지 않는 파일은 오류로 보고하며 절대 부분 로드하지 않습니다. 저장 전에 출력의 shape, 모든 값이 유한한지, 값이 Tanh 범위 `[-1, 1]` 안에 있는지 검사합니다. 스크립트는 다운로드, 텔레메트리 수집, 외부 서비스 호출을 하지 않습니다.
- **시드:** `[0, 2**32 - 1]` 범위의 정수.

### 두 가지 생성기 아키텍처 (서로 별개)

| | 레거시 (`models/legacy_imitator.py`) | 현재 (`models/imitator.py`) |
|---|---|---|
| 입력 | 37차원 벡터 | 960차원 라벨 벡터 |
| 출력 | 256x256 | 512x512 |
| 구조 | ConvTranspose 4x4 (BatchNorm, LeakyReLU) → bilinear 업샘플 Conv 블록 5개 (InstanceNorm, LeakyReLU) → 업샘플 Conv + Tanh; 채널 512, 512, 512, 256, 128, 64, 3 | MLP 컨디셔너 + FiLM 변조 업샘플 블록, `ProjectionDiscriminator`와 EMA로 학습 |
| 포함된 사전 학습 가중치 | `weigths/imitator.pt` | 없음 |
| 사용처 | `infer_legacy.py`, `train.py` (2단계) | `train_imitator.py` (1단계) |

레거시 체크포인트는 현재 `Imitator`에 로드되지 **않으며**, 이 저장소는 로드되는 것처럼 꾸미지 않습니다. 현재 960차원/512px 생성기는 별개의 학습용 아키텍처입니다. 사용하려면 직접 준비한 데이터로 학습해야 합니다.

## 학습 아키텍처 (직접 준비한 데이터 필요)

학습 코드는 2단계로 구성되어 있습니다. *현재* 960차원 생성기와 Translator의 학습된 가중치는 없고, 두 학습 과정 모두 직접 준비한 데이터셋이 필요합니다. 위의 별도 레거시 생성기 체크포인트만 여기서 실행할 수 있는 학습된 모델입니다.

1. **1단계 - Imitator 학습** (`train_imitator.py`): 조건부 GAN(현재 `Imitator` 생성기 + `ProjectionDiscriminator`)을 FiLM 컨디셔닝과 EMA를 사용해 960차원 라벨 벡터로부터 512x512 이미지를 생성하도록 학습합니다.
2. **2단계 - Translator 학습** (`train.py`): `Translator`(`facenet-pytorch`의 InceptionResnetV1 백본)가 얼굴 이미지로부터 37차원 파라미터를 예측합니다. 동결된 37차원 생성기와 스타일 변환 모델을 거쳐 루프백 손실로 학습합니다. `train.py`는 **레거시** 37차원 생성기를 사용하며, `--weight_imitator`를 동일한 엄격한 `weights_only=True` CPU 로더로 로드합니다. 2단계 학습이 동작한다는 것은 이 저장소에서 입증되지 않았습니다. 얼굴 데이터셋이 필요하고, 학습된 Translator는 여기에 없습니다.

**의도된 흐름 (학습된 Translator 없이는 여기서 재현 불가)**: 얼굴 이미지 → Translator → 파라미터 벡터 → Imitator → 생성된 이미지

## 시작하기

### 사전 요구 사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip
- `infer_legacy.py`는 CPU로 충분합니다. 학습에는 CUDA 12.1 호환 GPU를 권장합니다

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
    레거시 CPU 추론과 테스트만 실행할 때는 Python 3.11에서 부모가 검증한 다음 최소 환경으로 충분합니다(학습 의존성은 불필요):
    ```bash
    python -m pip install 'torch==2.5.1' 'numpy==1.26.4' 'pillow==10.4.0'
    ```
    별도 학습 코드는 지원되는 CUDA/Linux 환경에서 프로젝트 전체 요구 사항을 `uv`로 설치하세요:
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

이 단계는 레거시 37차원 생성기를 거쳐 Translator를 학습합니다. `--root_dir` 아래에 직접 준비한 얼굴 데이터셋이 필요합니다. `Translator`를 생성하면 `facenet-pytorch`가 사전 학습된 InceptionResnetV1 가중치를 내려받습니다. 이 명령으로 동작하는 Translator가 만들어진다는 것은 입증되지 않았습니다. 포함된 `style_transfer.pt`는 `models/animegan.py`와의 호환성이 검증되지 않았습니다.

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
| `--weight_imitator` | `None` | 레거시 37차원 Imitator 순수 state dict (`.pt`), 엄격 검증 |
| `--weight_style_transfer` | `None` | 사전 학습된 스타일 변환 가중치 (`.pt`) |

### 4. 테스트 실행

```bash
python -m unittest discover tests
```

`tests/test_legacy_imitator.py`는 엄격한 체크포인트 호환성, 출력 shape, 결정성, 잘못된 입력에 대한 오류 처리를 검사합니다. `weigths/imitator.pt`가 있으면 이 테스트에서 해당 파일도 로드합니다.

## 디렉토리 구조

```
.
├── datamodules/          # PyTorch Dataset 클래스
│   └── simple_datamodule.py
├── models/               # 신경망 아키텍처
│   ├── imitator.py       # 현재 960차원/512px 생성기 (Imitator) + ProjectionDiscriminator
│   ├── legacy_imitator.py # 레거시 37차원/256px 생성기 + 엄격한 체크포인트 로더
│   ├── translator.py     # 얼굴 인코더 (InceptionResnetV1 백본)
│   └── animegan.py       # AnimeGAN 스타일 변환 생성기
├── pipelines/            # PyTorch Lightning 학습 모듈
│   ├── imitator_pipeline.py  # EMA 적용 GAN 학습 루프
│   └── pipeline.py           # 루프백 손실을 활용한 Translator 학습
├── tests/                # 단위 테스트
│   ├── test_models.py
│   └── test_legacy_imitator.py
├── weigths/              # imitator.pt (레거시 37차원), style_transfer.pt
├── utils/                # 데이터 준비 유틸리티
│   ├── face_align.py         # MTCNN 얼굴 감지 및 정렬
│   ├── make_parameter.py     # MobileNet-V3 임베딩 생성
│   ├── make_pair_parquet.py  # 이미지-라벨 쌍 데이터셋 생성
│   ├── concat_tensor_images.py
│   └── move_samples.py
├── infer_legacy.py       # 레거시 체크포인트 CLI (합성 37차원 입력 -> PNG 1장)
├── train_imitator.py     # 1단계 학습 스크립트
├── train.py              # 2단계 학습 스크립트
├── pyproject.toml        # 프로젝트 메타데이터 및 의존성
└── requirements.txt      # 고정된 의존성 버전
```
