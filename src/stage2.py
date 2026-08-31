# Stage 2 - Fine-tuning / adaptation
#
# GitHub-ready configuration:
#   PROJECT_ROOT       Repository root (optional)
#   MODEL_KEY          convnext | swin | eva02 | effnet | caformer | maxvit | coatnet
#   STAGE2_DATA_ROOT   Data root; defaults to <repo>/data
#   STAGE2_TRAIN_DIR   Training ImageFolder root; defaults to <data_root>/Newtrain
#   STAGE1_ROOT        Directory containing Stage-1 checkpoints
#   STAGE2_OUT_DIR     Output directory
#   STAGE1_*_WEIGHTS   Per-backbone Stage-1 checkpoint override
#
# Install dependencies before running:
#   pip install -r requirements.txt

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"src", "scripts", "training"}
    else SCRIPT_DIR
)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()

# -------------------- user settings --------------------
MODEL_KEY = os.getenv("MODEL_KEY", "eva02").strip().lower()
DATA_ROOT = Path(
    os.getenv("STAGE2_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
STAGE1_ROOT = Path(
    os.getenv("STAGE1_ROOT", str(PROJECT_ROOT / "weights" / "stage1"))
).expanduser()
OUT_DIR = Path(
    os.getenv("STAGE2_OUT_DIR", str(PROJECT_ROOT / "outputs" / f"stage2_{MODEL_KEY}"))
).expanduser()

# Enable the conservative cuDNN fallback only for EfficientNetV2.
# This directly addresses: FIND was unable to find an engine...
FORCE_CUDNN_V7_FOR_EFFNET = os.getenv(
    "FORCE_CUDNN_V7_FOR_EFFNET", "1"
).strip().lower() not in {"0", "false", "no", "off"}

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if MODEL_KEY == "effnet" and FORCE_CUDNN_V7_FOR_EFFNET:
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = Path(
    os.getenv("STAGE2_TRAIN_DIR", str(DATA_ROOT / "Newtrain"))
).expanduser()


def stage1_weight(env_name, default_filename):
    value = os.getenv(env_name, "").strip()
    return Path(value).expanduser() if value else STAGE1_ROOT / default_filename


STAGE1_WEIGHTS = {
    "convnext": stage1_weight("STAGE1_CONVNEXT_WEIGHTS", "model_v1.1_convnext_gtsrb.pth"),
    "swin":     stage1_weight("STAGE1_SWIN_WEIGHTS", "model_v2.1_swin_gtsrb.pth"),
    "eva02":    stage1_weight("STAGE1_EVA02_WEIGHTS", "model_v3.1_eva02_gtsrb.pth"),
    "effnet":   stage1_weight("STAGE1_EFFNET_WEIGHTS", "model_v4.1_effnet_gtsrb.pth"),
    "caformer": stage1_weight("STAGE1_CAFORMER_WEIGHTS", "model_v5.1_caformer_stage1_best.pth"),
    "maxvit":   stage1_weight("STAGE1_MAXVIT_WEIGHTS", "model_v6.1_maxvit_stage1_best.pth"),
    "coatnet":  stage1_weight("STAGE1_COATNET_WEIGHTS", "model_v7.1_coatnet_stage1_best.pth"),
}

MODEL_CFG = {
    "convnext": {
        "name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "crop": 256, "batch": 20, "accum": 1,
        "lr": 2.5e-5, "head_lr_mult": 8.0,
        "drop_path": 0.15, "channels_last": True,
        "grad_checkpointing": False,
    },
    "swin": {
        "name": "swinv2_small_window16_256.ms_in1k",
        "crop": 256, "batch": 20, "accum": 1,
        "lr": 3.0e-5, "head_lr_mult": 8.0,
        "drop_path": 0.15, "channels_last": False,
        "grad_checkpointing": False,
    },
    "eva02": {
        "name": "eva02_base_patch14_224.mim_in22k",
        "crop": 224, "batch": 12, "accum": 2,
        "lr": 1.8e-5, "head_lr_mult": 10.0,
        "drop_path": 0.15, "channels_last": False,
        "grad_checkpointing": True,
    },
    "effnet": {
        "name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "crop": 256, "batch": 28, "accum": 1,
        "lr": 3.5e-5, "head_lr_mult": 8.0,
        "drop_path": 0.15,
        # Keep contiguous NCHW for EfficientNetV2 depthwise convolution.
        "channels_last": False,
        "grad_checkpointing": False,
    },
    "caformer": {
        "name": "caformer_s18.sail_in1k",
        "crop": 224, "batch": 24, "accum": 1,
        "lr": 3.0e-5, "head_lr_mult": 8.0,
        "drop_path": 0.12, "channels_last": False,
        "grad_checkpointing": False,
    },
    "maxvit": {
        "name": "maxvit_tiny_rw_224.sw_in1k",
        "crop": 224, "batch": 16, "accum": 2,
        "lr": 2.0e-5, "head_lr_mult": 10.0,
        "drop_path": 0.12, "channels_last": False,
        "grad_checkpointing": True,
    },
    "coatnet": {
        "name": "coatnet_0_rw_224.sw_in1k",
        "crop": 224, "batch": 10, "accum": 2,
        "lr": 2.2e-5, "head_lr_mult": 10.0,
        "drop_path": 0.12, "channels_last": False,
        "grad_checkpointing": True,
    },
}

if MODEL_KEY not in MODEL_CFG:
    raise ValueError(f"Unsupported MODEL_KEY={MODEL_KEY}; choose from {list(MODEL_CFG)}")
MCFG = MODEL_CFG[MODEL_KEY]

# -------------------- training settings --------------------
SEED = 3407
VAL_RATIO = 0.20
EPOCHS = 30
WARMUP_EPOCHS = 2
PATIENCE = 8
NUM_WORKERS = 4

WEIGHT_DECAY = 0.04
HEAD_WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
EFFECTIVE_NUMBER_BETA = 0.999

MIXUP_PROB = 0.25
MIXUP_ALPHA = 0.20
GRAD_CLIP = 1.0

USE_EMA = True
EMA_DECAY = 0.9998
EMA_START_EPOCH = 2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}

# -------------------- imports after env setup --------------------
import copy
import gc
import inspect
import json
import math
import random
import re
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.data import resolve_data_config

try:
    from timm.data import resolve_model_data_config
except ImportError:
    resolve_model_data_config = None

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None


# ============================================================
# 1. Device and reproducibility
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

# EfficientNetV2 uses FP16 instead of BF16 for better cuDNN compatibility.
if not USE_CUDA:
    AMP_DTYPE = torch.float32
elif MODEL_KEY == "effnet":
    AMP_DTYPE = torch.float16
elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = torch.float16

SCALER_ENABLED = USE_CUDA and AMP_DTYPE == torch.float16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if USE_CUDA:
    # Avoid cuDNN FIND search for EfficientNetV2.
    torch.backends.cudnn.benchmark = MODEL_KEY != "effnet"
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def autocast_context():
    if not USE_CUDA:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE)


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=SCALER_ENABLED)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=SCALER_ENABLED)


# ============================================================
# 2. Dataset scan and fixed stratified split
# ============================================================
def smart_class_key(name):
    """Sort 0_x, 1_x, ..., 42_x by their leading integer."""
    text = str(name).strip()
    match = re.match(r"^(\d+)(?:_|$)", text)
    if match:
        return 0, int(match.group(1)), text.lower()
    return 1, text.lower()


def scan_imagefolder(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {root}")

    class_names = sorted(
        [p.name for p in root.iterdir() if p.is_dir()],
        key=smart_class_key,
    )
    if not class_names:
        raise RuntimeError(f"No class folders found under {root}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    paths, labels = [], []

    for class_name in class_names:
        for path in sorted((root / class_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                paths.append(str(path))
                labels.append(class_to_idx[class_name])

    if not paths:
        raise RuntimeError(f"No images found under {root}")

    return paths, np.asarray(labels, dtype=np.int64), class_names, class_to_idx


ALL_PATHS, ALL_LABELS, CLASS_NAMES, CLASS_TO_IDX = scan_imagefolder(TRAIN_DIR)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COUNTS = np.bincount(ALL_LABELS, minlength=NUM_CLASSES)

if CLASS_COUNTS.min() < 2:
    bad = np.flatnonzero(CLASS_COUNTS < 2).tolist()
    raise RuntimeError(f"Every class needs at least two images; bad class indices: {bad}")

all_indices = np.arange(len(ALL_PATHS))
TRAIN_IDX, VAL_IDX = train_test_split(
    all_indices,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=ALL_LABELS,
)
TRAIN_IDX = np.asarray(sorted(TRAIN_IDX), dtype=np.int64)
VAL_IDX = np.asarray(sorted(VAL_IDX), dtype=np.int64)

TRAIN_PATHS = [ALL_PATHS[i] for i in TRAIN_IDX]
VAL_PATHS = [ALL_PATHS[i] for i in VAL_IDX]
TRAIN_LABELS = ALL_LABELS[TRAIN_IDX]
VAL_LABELS = ALL_LABELS[VAL_IDX]

print("=" * 80)
print(f"MODEL_KEY: {MODEL_KEY}")
print(f"TIMM model: {MCFG['name']}")
print(f"Device: {DEVICE}")
print(f"AMP dtype: {AMP_DTYPE}")
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark if USE_CUDA else False}")
print(f"cuDNN v8 disabled: {os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0')}")
print(f"Classes: {NUM_CLASSES}")
print(f"Train images: {len(TRAIN_PATHS)}")
print(f"Validation images: {len(VAL_PATHS)}")
print("=" * 80)
print("Class order:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {idx:02d}: {name} ({CLASS_COUNTS[idx]})")

with open(OUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {"class_names": CLASS_NAMES, "class_to_idx": CLASS_TO_IDX},
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 3. Traffic-sign-friendly augmentation
# ============================================================
class RandomPhotometricDegrade:
    def __call__(self, image):
        out = image.copy()

        if np.random.random() < 0.45:
            alpha = np.random.uniform(0.82, 1.18)
            beta = np.random.uniform(-22.0, 18.0)
            out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if np.random.random() < 0.25:
            gamma = np.random.uniform(0.78, 1.55)
            lut = np.clip((np.arange(256) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
            out = cv2.LUT(out, lut)

        if np.random.random() < 0.22:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-4, 4)) % 180
            hsv[..., 1] *= np.random.uniform(0.88, 1.12)
            hsv[..., 2] *= np.random.uniform(0.90, 1.10)
            hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Low-light simulation.
        if np.random.random() < 0.32:
            x = out.astype(np.float32) / 255.0
            x = np.power(np.clip(x, 0.0, 1.0), np.random.uniform(1.05, 1.85))
            x = x * np.random.uniform(0.50, 0.90) + np.random.uniform(-0.055, 0.01)

            if np.random.random() < 0.45:
                h, w = x.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                cx = np.random.uniform(0.15, 0.85) * max(w - 1, 1)
                cy = np.random.uniform(0.15, 0.85) * max(h - 1, 1)
                nx = (xx - cx) / max(w, 1)
                ny = (yy - cy) / max(h, 1)
                radius = np.sqrt(nx * nx + ny * ny)
                radius /= max(float(radius.max()), 1e-6)
                mask = 1.0 - np.random.uniform(0.10, 0.32) * np.power(
                    radius, np.random.uniform(1.2, 2.4)
                )
                x *= mask[..., None]

            x += np.random.normal(0.0, np.random.uniform(0.003, 0.022), x.shape).astype(np.float32)
            out = np.clip(x * 255.0, 0, 255).astype(np.uint8)

        if np.random.random() < 0.16:
            k = int(np.random.choice([3, 5]))
            if np.random.random() < 0.60:
                out = cv2.GaussianBlur(out, (k, k), sigmaX=np.random.uniform(0.2, 1.3))
            else:
                kernel = np.zeros((k, k), dtype=np.float32)
                if np.random.random() < 0.5:
                    kernel[k // 2, :] = 1.0 / k
                else:
                    kernel[:, k // 2] = 1.0 / k
                out = cv2.filter2D(out, -1, kernel)

        if np.random.random() < 0.14:
            noise = np.random.normal(0.0, np.random.uniform(3.0, 12.0), out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if np.random.random() < 0.12:
            h, w = out.shape[:2]
            scale = np.random.uniform(0.55, 0.85)
            rw, rh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
            out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_AREA)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        if np.random.random() < 0.12:
            quality = int(np.random.randint(45, 91))
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        if np.random.random() < 0.12:
            h, w = out.shape[:2]
            bh = max(1, int(h * np.random.uniform(0.04, 0.12)))
            bw = max(1, int(w * np.random.uniform(0.04, 0.12)))
            y1 = np.random.randint(0, max(1, h - bh + 1))
            x1 = np.random.randint(0, max(1, w - bw + 1))
            fill = np.median(out.reshape(-1, 3), axis=0).astype(np.uint8)
            out[y1:y1 + bh, x1:x1 + bw] = fill

        return out


def make_compose(transforms):
    kwargs = {}
    try:
        if "strict" in inspect.signature(A.Compose).parameters:
            kwargs["strict"] = True
    except (TypeError, ValueError):
        pass
    return A.Compose(transforms, **kwargs)


def make_random_resized_crop(crop):
    params = inspect.signature(A.RandomResizedCrop).parameters
    kwargs = {
        "scale": (0.86, 1.00),
        "ratio": (0.90, 1.11),
        "interpolation": cv2.INTER_CUBIC,
        "p": 1.0,
    }
    if "size" in params:
        kwargs["size"] = (crop, crop)
    else:
        kwargs["height"] = crop
        kwargs["width"] = crop
    return A.RandomResizedCrop(**kwargs)


def build_train_transform(mean, std, crop):
    # No horizontal/vertical flip: direction signs can change meaning.
    return make_compose([
        make_random_resized_crop(crop),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.38),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_eval_transform(mean, std, crop):
    resize = int(round(crop * 1.10))
    return make_compose([
        A.SmallestMaxSize(max_size=resize, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(height=crop, width=crop),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class SignDataset(Dataset):
    def __init__(self, paths, labels, transform, train=False):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self.train = bool(train)
        self.photo_aug = RandomPhotometricDegrade()

    def __len__(self):
        return len(self.paths)

    def set_worker_seed(self, seed):
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(int(seed))

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            image = self.photo_aug(image)
        tensor = self.transform(image=image)["image"]
        return tensor, int(self.labels[index])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed(worker_seed)


# ============================================================
# 4. Load Stage-1 backbone and rebuild classifier
# ============================================================
def load_checkpoint_cpu(path):
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        if safe_load_file is None:
            raise ImportError("Install safetensors first")
        return safe_load_file(str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("model_state", "state_dict_ema", "model_ema", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_common_prefix(state_dict, prefix):
    keys = list(state_dict)
    if not keys:
        return state_dict
    ratio = sum(key.startswith(prefix) for key in keys) / len(keys)
    if ratio < 0.80:
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def create_model_without_classifier():
    kwargs = {
        "pretrained": False,
        "num_classes": 0,
        "drop_path_rate": MCFG["drop_path"],
    }
    try:
        return timm.create_model(MCFG["name"], **kwargs)
    except TypeError:
        kwargs.pop("drop_path_rate", None)
        return timm.create_model(MCFG["name"], **kwargs)


def build_model_from_stage1():
    weight_path = Path(STAGE1_WEIGHTS[MODEL_KEY])
    if not weight_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {weight_path}")

    model = create_model_without_classifier()
    checkpoint = unwrap_state_dict(load_checkpoint_cpu(weight_path))

    for prefix in ("module.", "_orig_mod.", "model."):
        checkpoint = strip_common_prefix(checkpoint, prefix)

    model_state = model.state_dict()
    loadable = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
        and key in model_state
        and model_state[key].shape == value.shape
    }

    loaded_numel = sum(v.numel() for v in loadable.values())
    total_numel = sum(v.numel() for v in model_state.values())
    coverage = loaded_numel / max(total_numel, 1)

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(f"Stage-1 backbone coverage: {coverage * 100:.2f}%")
    print(f"Missing keys before rebuilding head: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if coverage < 0.80:
        raise RuntimeError(
            f"Backbone coverage is only {coverage * 100:.2f}%; "
            f"check whether the checkpoint matches {MCFG['name']}"
        )

    if not hasattr(model, "reset_classifier"):
        raise RuntimeError("Selected timm model has no reset_classifier")
    model.reset_classifier(NUM_CLASSES)

    if MCFG.get("grad_checkpointing", False) and hasattr(model, "set_grad_checkpointing"):
        try:
            model.set_grad_checkpointing(enable=True)
        except TypeError:
            model.set_grad_checkpointing(True)
        print("Gradient checkpointing enabled")

    return model


def get_model_data_config(model):
    if resolve_model_data_config is not None:
        try:
            return resolve_model_data_config(model)
        except Exception:
            pass
    return resolve_data_config({}, model=model)


@torch.no_grad()
def validate_model_forward(model):
    model.eval()
    dummy = torch.randn(2, 3, MCFG["crop"], MCFG["crop"], device=DEVICE)
    if MCFG["channels_last"]:
        dummy = dummy.contiguous(memory_format=torch.channels_last)

    try:
        with autocast_context():
            logits = model(dummy)
        if USE_CUDA:
            torch.cuda.synchronize()
        expected = (2, NUM_CLASSES)
        if tuple(logits.shape) != expected:
            raise RuntimeError(f"Unexpected output shape {tuple(logits.shape)}, expected {expected}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Forward validation produced NaN/Inf")
        print(
            "Forward validation passed | "
            f"input={tuple(dummy.shape)} | output={tuple(logits.shape)} | "
            f"AMP={AMP_DTYPE} | channels_last={MCFG['channels_last']}"
        )
    finally:
        del dummy
        if "logits" in locals():
            del logits
        if USE_CUDA:
            torch.cuda.empty_cache()
    model.train()


# ============================================================
# 5. Macro-F1-oriented loss, MixUp and EMA
# ============================================================
def make_effective_number_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective = 1.0 - np.power(EFFECTIVE_NUMBER_BETA, counts)
    weights = (1.0 - EFFECTIVE_NUMBER_BETA) / np.maximum(effective, 1e-12)
    weights /= weights.mean()
    weights = np.clip(weights, 0.50, 3.00)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class WeightedSmoothSoftCE(nn.Module):
    def __init__(self, num_classes, class_weights, smoothing=0.05):
        super().__init__()
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)
        self.register_buffer("class_weights", class_weights.float())

    def make_target(self, labels):
        target = F.one_hot(labels, num_classes=self.num_classes).float()
        if self.smoothing > 0:
            target = target * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        return target

    def forward(self, logits, labels_a, labels_b, lam):
        target = lam * self.make_target(labels_a) + (1.0 - lam) * self.make_target(labels_b)
        log_prob = F.log_softmax(logits, dim=1)
        per_sample = -(target * log_prob).sum(dim=1)
        sample_weight = (target * self.class_weights.unsqueeze(0)).sum(dim=1)
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def mixup_batch(images, labels):
    if images.size(0) < 2 or np.random.random() >= MIXUP_PROB:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.num_updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        source = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = source[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


# ============================================================
# 6. Optimizer and scheduler
# ============================================================
def get_classifier_parameter_ids(model):
    try:
        classifier = model.get_classifier()
    except Exception:
        classifier = None
    if isinstance(classifier, nn.Module):
        return {id(p) for p in classifier.parameters()}
    return set()


def make_optimizer(model):
    classifier_ids = get_classifier_parameter_ids(model)
    groups = {k: [] for k in ("body_decay", "body_no_decay", "head_decay", "head_no_decay")}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in classifier_ids
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        key = ("head_" if is_head else "body_") + ("no_decay" if no_decay else "decay")
        groups[key].append(p)

    base_lr = MCFG["lr"]
    head_lr = base_lr * MCFG["head_lr_mult"]
    param_groups = []

    def add(params, lr, wd, name):
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": wd, "group_name": name})

    add(groups["body_decay"], base_lr, WEIGHT_DECAY, "body_decay")
    add(groups["body_no_decay"], base_lr, 0.0, "body_no_decay")
    add(groups["head_decay"], head_lr, HEAD_WEIGHT_DECAY, "head_decay")
    add(groups["head_no_decay"], head_lr, 0.0, "head_no_decay")

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def make_scheduler(optimizer, updates_per_epoch):
    total_steps = max(1, EPOCHS * updates_per_epoch)
    warmup_steps = max(1, WARMUP_EPOCHS * updates_per_epoch)

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1e-3 + (1.0 - 1e-3) * cosine

    return LambdaLR(optimizer, lr_lambda=[factor for _ in optimizer.param_groups])


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# 7. Train and evaluate
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    predictions, targets = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        if MCFG["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context():
            logits = model(images)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())

    if not predictions:
        raise RuntimeError("Evaluation loader is empty")

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum = int(MCFG["accum"])
    total_loss, total_samples = 0.0, 0

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"{MODEL_KEY} epoch {epoch + 1:02d}/{EPOCHS}",
    )

    for step, (images, labels) in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if MCFG["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)

        images, labels_a, labels_b, lam = mixup_batch(images, labels)

        group_start = (step // accum) * accum
        group_size = min(accum, len(loader) - group_start)

        with autocast_context():
            logits = model(images)
            raw_loss = criterion(logits, labels_a, labels_b, lam)
            loss = raw_loss / group_size

        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite loss: {raw_loss.item()}")

        scaler.scale(loss).backward()

        should_update = (step + 1) % accum == 0 or (step + 1) == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        batch_size = images.size(0)
        total_loss += raw_loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / max(total_samples, 1)


# ============================================================
# 8. Main
# ============================================================
def main():
    model = build_model_from_stage1()
    data_cfg = get_model_data_config(model)
    mean, std = list(data_cfg["mean"]), list(data_cfg["std"])

    train_transform = build_train_transform(mean, std, MCFG["crop"])
    eval_transform = build_eval_transform(mean, std, MCFG["crop"])

    train_dataset = SignDataset(TRAIN_PATHS, TRAIN_LABELS, train_transform, train=True)
    val_dataset = SignDataset(VAL_PATHS, VAL_LABELS, eval_transform, train=False)

    generator = torch.Generator().manual_seed(SEED)
    loader_common = {
        "num_workers": NUM_WORKERS,
        "pin_memory": USE_CUDA,
        "worker_init_fn": seed_worker,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=MCFG["batch"],
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(8, MCFG["batch"]),
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    model = model.to(DEVICE)
    if MCFG["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    validate_model_forward(model)

    class_weights = make_effective_number_weights(TRAIN_LABELS, NUM_CLASSES).to(DEVICE)
    print(f"Class-weight range: {class_weights.min().item():.3f} - {class_weights.max().item():.3f}")

    criterion = WeightedSmoothSoftCE(
        NUM_CLASSES,
        class_weights=class_weights,
        smoothing=LABEL_SMOOTHING,
    ).to(DEVICE)
    optimizer = make_optimizer(model)
    updates_per_epoch = math.ceil(len(train_loader) / int(MCFG["accum"]))
    scheduler = make_scheduler(optimizer, updates_per_epoch)
    scaler = make_grad_scaler()
    ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_path = OUT_DIR / f"{MODEL_KEY}_best.pth"

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, ema, epoch
        )

        raw_metrics = evaluate(model, val_loader)
        candidates = [{"name": "RAW", "model": model, "metrics": raw_metrics}]

        if ema is not None and epoch >= EMA_START_EPOCH:
            candidates.append({"name": "EMA", "model": ema.module, "metrics": evaluate(ema.module, val_loader)})

        selected = max(
            candidates,
            key=lambda item: (item["metrics"]["macro_f1"], item["metrics"]["accuracy"]),
        )

        print(f"\nEpoch {epoch + 1:02d} | train loss={train_loss:.4f}")
        for c in candidates:
            print(
                f"  {c['name']} | Val Macro-F1={c['metrics']['macro_f1'] * 100:.2f}% "
                f"| Val Accuracy={c['metrics']['accuracy'] * 100:.2f}%"
            )
        print(f"  Selected: {selected['name']}")

        selected_f1 = selected["metrics"]["macro_f1"]
        selected_acc = selected["metrics"]["accuracy"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "selected": selected["name"],
            "val_macro_f1": float(selected_f1),
            "val_accuracy": float(selected_acc),
            "body_lr": float(optimizer.param_groups[0]["lr"]),
        })
        pd.DataFrame(history).to_csv(OUT_DIR / f"{MODEL_KEY}_history.csv", index=False)
        with open(OUT_DIR / f"{MODEL_KEY}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        improved = (
            selected_f1 > best_f1 + 1e-6
            or (abs(selected_f1 - best_f1) <= 1e-6 and selected_acc > best_acc)
        )

        if improved:
            best_f1 = selected_f1
            best_acc = selected_acc
            best_epoch = epoch + 1
            no_improve = 0

            torch.save({
                "model_state": cpu_state_dict(selected["model"]),
                "model_key": MODEL_KEY,
                "timm_name": MCFG["name"],
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "crop": MCFG["crop"],
                "mean": mean,
                "std": std,
                "channels_last": MCFG["channels_last"],
                "drop_path": MCFG["drop_path"],
                "val_macro_f1": best_f1,
                "val_accuracy": best_acc,
                "best_epoch": best_epoch,
                "selected_weights": selected["name"],
                "seed": SEED,
            }, best_path)

            print(f"  New best saved | Macro-F1={best_f1 * 100:.2f}%")
        else:
            no_improve += 1
            print(f"  No improvement: {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print("Early stopping")
                break

    summary = {
        "model_key": MODEL_KEY,
        "timm_name": MCFG["name"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_accuracy": best_acc,
        "checkpoint": str(best_path),
    }
    with open(OUT_DIR / f"{MODEL_KEY}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Training finished: {MODEL_KEY}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_f1 * 100:.2f}%")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"Checkpoint: {best_path}")
    print("=" * 80)

    del model, ema, optimizer, scheduler, scaler, criterion
    gc.collect()
    if USE_CUDA:
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
