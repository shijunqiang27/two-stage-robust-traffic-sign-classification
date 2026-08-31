
# Stage 1 - Robust backbone training
#
# GitHub-ready configuration:
#   PROJECT_ROOT       Repository root (optional)
#   MODEL_KEY          convnext | swin | eva02 | effnet | caformer | maxvit | coatnet
#   GTSRB_ROOT         GTSRB root containing Train/, Test/, and Test.csv
#   GTSRB_TRAIN_DIR    Override the training directory
#   GTSRB_TEST_DIR     Override the test directory
#   GTSRB_TEST_CSV     Override the official test CSV
#   STAGE1_OUT_DIR     Training output directory
#   PRETRAINED_ROOT    Directory containing optional local pretrained weights
#   *_WEIGHTS          Per-backbone local pretrained-weight override
#   USE_HF_MIRROR=1    Opt in to https://hf-mirror.com
#
# Install dependencies before running:
#   pip install -r requirements.txt

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.getenv("USE_HF_MIRROR", "0").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import copy
import inspect
import json
import math
import random
import warnings
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

import timm
from timm.data import resolve_data_config

import albumentations as A
from albumentations.pytorch import ToTensorV2

from safetensors.torch import load_file as safe_load_file


# ================================================================
# 1. Main configuration
# ================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"src", "scripts", "training"}
    else SCRIPT_DIR
)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()

DATASET = os.getenv("DATASET", "gtsrb").strip().lower()
MODEL_KEY = os.getenv("MODEL_KEY", "coatnet").strip().lower()
# Supported: convnext | swin | eva02 | effnet | caformer | maxvit | coatnet

OUT_DIR = os.getenv(
    "STAGE1_OUT_DIR",
    str(PROJECT_ROOT / "outputs" / f"stage1_darkrobust_{MODEL_KEY}_{DATASET}"),
)
os.makedirs(OUT_DIR, exist_ok=True)

PRETRAINED_ROOT = Path(
    os.getenv("PRETRAINED_ROOT", str(PROJECT_ROOT / "weights" / "pretrained"))
).expanduser()


def optional_local_weight(env_name, filename):
    """Use an explicit env path, otherwise use a conventional local file if present."""
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        return str(Path(explicit).expanduser())

    candidate = PRETRAINED_ROOT / filename
    return str(candidate) if candidate.is_file() else ""


LOCAL_WEIGHTS = {
    "convnext": optional_local_weight("CONVNEXT_WEIGHTS", "convnextv2_base.safetensors"),
    "swin":     optional_local_weight("SWIN_WEIGHTS", "swinv2_small.safetensors"),
    "eva02":    optional_local_weight("EVA02_WEIGHTS", "eva02_base.safetensors"),
    "effnet":   optional_local_weight("EFFNET_WEIGHTS", "efficientnetv2_s.safetensors"),
    "caformer": optional_local_weight("CAFORMER_WEIGHTS", "caformer_s18.safetensors"),
    "maxvit":   optional_local_weight("MAXVIT_WEIGHTS", "maxvit_tiny_rw_224.safetensors"),
    "coatnet":  optional_local_weight("COATNET_WEIGHTS", "coatnet_0_rw_224.safetensors"),
}

ALLOW_HF_DOWNLOAD = os.getenv("ALLOW_HF_DOWNLOAD", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
REQUIRE_PRETRAINED = os.getenv("REQUIRE_PRETRAINED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

GTSRB_ROOT = Path(
    os.getenv("GTSRB_ROOT", str(PROJECT_ROOT / "data" / "GTSRB"))
).expanduser()

DATASET_PROFILES = {
    "gtsrb": {
        "loader": "gtsrb",
        "train_dir": os.getenv("GTSRB_TRAIN_DIR", str(GTSRB_ROOT / "Train")),
        "test_dir": os.getenv("GTSRB_TEST_DIR", str(GTSRB_ROOT / "Test")),
        "test_csv": os.getenv("GTSRB_TEST_CSV", str(GTSRB_ROOT / "Test.csv")),
        "numeric_align": True,
        "use_test_roi": False,
        "group": "gtsrb_track",
    },
}
PROF = DATASET_PROFILES[DATASET]

MODEL_CFG = {
    "convnext": {
        "name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "crop": 256, "resize": 288,
        "batch": 24, "accum": 1,
        "lr": 3e-5, "head_lr_mult": 4.0,
        "drop_path": 0.20,
        "channels_last": True,
        "grad_checkpointing": False,
    },
    "swin": {
        "name": "swinv2_small_window16_256.ms_in1k",
        "crop": 256, "resize": 280,
        "batch": 24, "accum": 1,
        "lr": 4e-5, "head_lr_mult": 4.0,
        "drop_path": 0.20,
        "channels_last": False,
        "grad_checkpointing": False,
    },
    "eva02": {
        "name": "eva02_base_patch14_224.mim_in22k",
        "crop": 224, "resize": 256,
        "batch": 16, "accum": 1,
        "lr": 2e-5, "head_lr_mult": 5.0,
        "drop_path": 0.20,
        "channels_last": False,
        "grad_checkpointing": False,
    },
    "effnet": {
        "name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "crop": 256, "resize": 288,
        "batch": 32, "accum": 1,
        "lr": 4e-5, "head_lr_mult": 4.0,
        "drop_path": 0.20,
        "channels_last": True,
        "grad_checkpointing": False,
    },
    "caformer": {
        "name": "caformer_s18.sail_in1k",
        "crop": 224, "resize": 256,
        "batch": 16, "accum": 1,
        "lr": 3e-5, "head_lr_mult": 5.0,
        "drop_path": 0.15,
        "channels_last": False,
        "grad_checkpointing": False,
    },
    "maxvit": {
        "name": "maxvit_tiny_rw_224.sw_in1k",
        "crop": 224, "resize": 256,
        "batch": 16, "accum": 1,
        "lr": 2e-5, "head_lr_mult": 5.0,
        "drop_path": 0.15,
        "channels_last": False,
        "grad_checkpointing": True,
    },
    "coatnet": {
        "name": "coatnet_0_rw_224.sw_in1k",
        "crop": 224, "resize": 256,
        "batch": 10, "accum": 2,
        "lr": 3e-5, "head_lr_mult": 5.0,
        "drop_path": 0.15,
        "channels_last": False,
        "grad_checkpointing": True,
    },
}
MCFG = MODEL_CFG[MODEL_KEY]


class CFG:
    epochs = 35
    warmup_epochs = 3
    min_lr_ratio = 1e-3

    weight_decay = 0.05
    head_weight_decay = 0.01
    grad_clip = 1.0

    label_smoothing = 0.05
    la_tau = 1.0

    use_mixup = True
    mixup_prob = 0.25
    mixup_alpha = 0.20

    use_ema = True
    ema_decay = 0.9998
    ema_warmup_epochs = 3

    val_ratio = 0.15
    patience = 10

    num_workers = 4
    seed = 42

    monitor_dark_val = True
    dark_score_weight = 0.25

    fixed_dark_alpha = 0.925
    fixed_dark_beta = -0.275

    profile_sizes = True
    validate_augmentation_pipeline = True
    final_test_eval = True
    final_test_dark_eval = True


SIZE_AWARE_AUG = True
SMALL_MAX = 64
LARGE_MIN = 96
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")


# ================================================================
# 2. Environment and random seeds
# ================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

if (
    USE_CUDA
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = torch.float16

SCALER_ENABLED = USE_CUDA and AMP_DTYPE == torch.float16

try:
    scaler = torch.amp.GradScaler("cuda", enabled=SCALER_ENABLED)
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=SCALER_ENABLED)


def amp_autocast():
    return torch.amp.autocast(
        device_type="cuda",
        dtype=AMP_DTYPE,
        enabled=USE_CUDA,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)


def _set_transform_seed(transform, seed):
    if isinstance(transform, dict):
        for value in transform.values():
            _set_transform_seed(value, seed)
        return

    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(int(seed))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "set_worker_seed"):
            dataset.set_worker_seed(worker_seed)


set_seed(CFG.seed)

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ================================================================
# 3. Data loading
# ================================================================
def _as_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def gather_imagefolder(root, numeric_align):
    root = str(root)

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data directory not found: {root}")

    subdirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    if not subdirs:
        raise RuntimeError(f"No class subdirectories under {root}")

    all_numeric = all(_as_int(d) is not None for d in subdirs)

    if numeric_align and all_numeric:
        class_names = sorted(subdirs, key=lambda x: int(x))
        classid_to_idx = {
            int(class_name): idx
            for idx, class_name in enumerate(class_names)
        }
        print("✓ Numeric class folders sorted as integers and aligned to ClassId")
    else:
        class_names = sorted(subdirs)
        classid_to_idx = None

    name_to_idx = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    paths, labels = [], []

    for class_name in class_names:
        class_dir = os.path.join(root, class_name)

        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(class_dir, filename))
                labels.append(name_to_idx[class_name])

    if not paths:
        raise RuntimeError(f"No images found under {root}")

    return (
        paths,
        np.asarray(labels, dtype=np.int64),
        class_names,
        classid_to_idx,
    )


def locate_test_csv(csv_path, test_dir):
    if csv_path and os.path.isfile(csv_path):
        return csv_path

    base = os.path.dirname(test_dir.rstrip("/"))

    candidates = []
    if os.path.isdir(base):
        candidates = sorted(
            os.path.join(base, filename)
            for filename in os.listdir(base)
            if filename.lower().endswith(".csv")
        )

    if not candidates:
        raise FileNotFoundError(
            f"Test CSV not found: {csv_path}; no CSV under {base}"
        )

    print(f"⚠ Specified CSV not found; using {candidates[0]}")
    return candidates[0]


def load_gtsrb_test(prof, classid_to_idx):
    csv_path = locate_test_csv(
        prof["test_csv"],
        prof["test_dir"],
    )
    test_dir = prof["test_dir"]

    frame = pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
    )
    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    if "Path" in frame.columns:
        path_column = "Path"
    elif "Filename" in frame.columns:
        path_column = "Filename"
    else:
        raise RuntimeError(
            f"CSV has no Path/Filename column: {list(frame.columns)}"
        )

    if "ClassId" not in frame.columns:
        raise RuntimeError(
            f"CSV has no ClassId column: {list(frame.columns)}"
        )

    roi_columns = [
        "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2",
    ]
    has_roi = all(
        column in frame.columns
        for column in roi_columns
    )

    paths, labels, rois = [], [], []
    missing_files = 0
    missing_classes = 0

    for _, row in frame.iterrows():
        class_id = _as_int(row["ClassId"])

        if class_id is None:
            missing_classes += 1
            continue

        if classid_to_idx:
            if class_id not in classid_to_idx:
                missing_classes += 1
                continue
            label = classid_to_idx[class_id]
        else:
            label = class_id

        filename = os.path.basename(
            str(row[path_column])
        )
        file_path = os.path.join(
            test_dir,
            filename,
        )

        if not os.path.isfile(file_path):
            missing_files += 1
            continue

        paths.append(file_path)
        labels.append(label)

        if has_roi:
            rois.append((
                int(row["Roi.X1"]),
                int(row["Roi.Y1"]),
                int(row["Roi.X2"]),
                int(row["Roi.Y2"]),
            ))
        else:
            rois.append(None)

    if missing_files:
        print(f"⚠ Skipped {missing_files} test images with missing files")
    if missing_classes:
        print(f"⚠ Skipped {missing_classes} test images with unmatched classes")

    return {
        "paths": paths,
        "labels": np.asarray(labels, dtype=np.int64),
        "rois": rois if has_roi else None,
    }


def load_data(prof):
    if prof["loader"] == "imagefolder":
        paths, labels, class_names, classid_to_idx = gather_imagefolder(
            prof["root"],
            prof["numeric_align"],
        )
        return {
            "paths": paths,
            "labels": labels,
            "class_names": class_names,
            "classid_to_idx": classid_to_idx,
            "test": None,
        }

    if prof["loader"] == "gtsrb":
        paths, labels, class_names, classid_to_idx = gather_imagefolder(
            prof["train_dir"],
            prof["numeric_align"],
        )
        test = load_gtsrb_test(
            prof,
            classid_to_idx,
        )

        return {
            "paths": paths,
            "labels": labels,
            "class_names": class_names,
            "classid_to_idx": classid_to_idx,
            "test": test,
        }

    raise ValueError(
        f"Unknown loader: {prof['loader']}"
    )


def profile_min_sizes(paths):
    try:
        from PIL import Image
    except ImportError:
        return

    values = []

    for path in tqdm(
        paths,
        desc="Profiling original sizes",
        leave=False,
    ):
        try:
            with Image.open(path) as image:
                width, height = image.size
            values.append(min(width, height))
        except Exception:
            pass

    if not values:
        return

    array = np.asarray(values)
    quantiles = np.percentile(
        array,
        [5, 25, 50, 75, 95],
    ).astype(int)

    print(
        "Original min-side distribution: "
        f"min={int(array.min())} "
        f"p5={quantiles[0]} "
        f"p25={quantiles[1]} "
        f"p50={quantiles[2]} "
        f"p75={quantiles[3]} "
        f"p95={quantiles[4]} "
        f"max={int(array.max())}"
    )
    print(
        f"  small (<{SMALL_MAX}px): "
        f"{(array < SMALL_MAX).mean() * 100:.1f}% | "
        "mid: "
        f"{((array >= SMALL_MAX) & (array < LARGE_MIN)).mean() * 100:.1f}% | "
        f"large (>={LARGE_MIN}px): "
        f"{(array >= LARGE_MIN).mean() * 100:.1f}%"
    )


DATA = load_data(PROF)
PATHS = DATA["paths"]
LABELS = DATA["labels"]
CLASS_NAMES = DATA["class_names"]
CLASSID_TO_IDX = DATA["classid_to_idx"]
TEST = DATA["test"]
NUM_CLASSES = len(CLASS_NAMES)

class_counts_all = np.bincount(
    LABELS,
    minlength=NUM_CLASSES,
)

if class_counts_all.min() < 2:
    bad_classes = np.where(
        class_counts_all < 2
    )[0].tolist()
    raise RuntimeError(
        "Classes with fewer than two images: "
        f"{bad_classes}"
    )

print(
    f"[{DATASET}] train={len(PATHS)}, classes={NUM_CLASSES}"
    + (
        f", official test={len(TEST['paths'])}"
        if TEST is not None
        else ", no independent test"
    )
)
print(
    "Class counts:",
    dict(zip(
        CLASS_NAMES,
        class_counts_all.tolist(),
    )),
)

if CFG.profile_sizes:
    profile_min_sizes(PATHS)


# ================================================================
# 4. Low-light simulation
# ================================================================
LOWLIGHT_POLICY = {
    "small": {
        "p": 0.50,
        "severity_probs": (0.58, 0.35, 0.07),
        "local_shadow_p": 0.25,
    },
    "mid": {
        "p": 0.60,
        "severity_probs": (0.46, 0.42, 0.12),
        "local_shadow_p": 0.35,
    },
    "large": {
        "p": 0.65,
        "severity_probs": (0.36, 0.44, 0.20),
        "local_shadow_p": 0.45,
    },
}


class LowLightSimulator:
    SEVERITY = {
        "mild": {
            "alpha": (0.86, 1.00),
            "beta": (-0.18, -0.08),
            "gamma": (1.05, 1.45),
            "gain": (0.62, 0.88),
            "bias": (-0.08, -0.02),
            "noise": (0.004, 0.014),
            "shadow": (0.08, 0.22),
            "black_clip": (0.00, 0.018),
            "linear_prob": 0.55,
        },
        "mid": {
            "alpha": (0.70, 0.94),
            "beta": (-0.32, -0.16),
            "gamma": (1.25, 1.95),
            "gain": (0.42, 0.72),
            "bias": (-0.12, -0.04),
            "noise": (0.010, 0.030),
            "shadow": (0.15, 0.35),
            "black_clip": (0.01, 0.045),
            "linear_prob": 0.62,
        },
        "heavy": {
            "alpha": (0.58, 0.90),
            "beta": (-0.48, -0.30),
            "gamma": (1.60, 2.60),
            "gain": (0.25, 0.52),
            "bias": (-0.16, -0.06),
            "noise": (0.020, 0.055),
            "shadow": (0.22, 0.48),
            "black_clip": (0.025, 0.075),
            "linear_prob": 0.72,
        },
    }

    def __call__(self, image, level):
        policy = LOWLIGHT_POLICY[level]

        if np.random.random() >= policy["p"]:
            return image

        severity_name = np.random.choice(
            ["mild", "mid", "heavy"],
            p=policy["severity_probs"],
        )
        config = self.SEVERITY[severity_name]

        x = image.astype(np.float32) / 255.0

        if np.random.random() < config["linear_prob"]:
            alpha = np.random.uniform(
                *config["alpha"]
            )
            beta = np.random.uniform(
                *config["beta"]
            )
            x = x * alpha + beta
        else:
            gamma = np.random.uniform(
                *config["gamma"]
            )
            gain = np.random.uniform(
                *config["gain"]
            )
            bias = np.random.uniform(
                *config["bias"]
            )
            x = (
                np.power(
                    np.clip(x, 0.0, 1.0),
                    gamma,
                )
                * gain
                + bias
            )

        if np.random.random() < policy["local_shadow_p"]:
            height, width = x.shape[:2]
            yy, xx = np.mgrid[
                0:height,
                0:width,
            ].astype(np.float32)

            cx = np.random.uniform(
                0.20,
                0.80,
            ) * max(width - 1, 1)
            cy = np.random.uniform(
                0.20,
                0.80,
            ) * max(height - 1, 1)

            nx = (xx - cx) / max(width, 1)
            ny = (yy - cy) / max(height, 1)
            radius = np.sqrt(nx * nx + ny * ny)
            radius = radius / max(
                float(radius.max()),
                1e-6,
            )

            strength = np.random.uniform(
                *config["shadow"]
            )
            power = np.random.uniform(
                1.1,
                2.4,
            )
            illumination = (
                1.0
                - strength
                * np.power(radius, power)
            )
            x = x * illumination[..., None]

        if np.random.random() < 0.35:
            channel_gain = np.random.uniform(
                0.90,
                1.08,
                size=(1, 1, 3),
            )
            x = x * channel_gain

        black_clip = np.random.uniform(
            *config["black_clip"]
        )
        if black_clip > 0:
            x = (
                (x - black_clip)
                / max(1.0 - black_clip, 1e-6)
            )

        x = np.clip(x, 0.0, 1.0)

        sigma = np.random.uniform(
            *config["noise"]
        )
        if sigma > 0:
            noise_scale = sigma * (
                0.45
                + np.sqrt(
                    np.clip(x, 0.0, 1.0)
                )
            )
            x = (
                x
                + np.random.normal(
                    0.0,
                    1.0,
                    size=x.shape,
                )
                * noise_scale
            )

        return np.clip(
            x * 255.0,
            0,
            255,
        ).astype(np.uint8)


class FixedDarkPreprocess:
    def __init__(
        self,
        alpha=0.925,
        beta=-0.275,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, image):
        x = image.astype(np.float32) / 255.0
        x = x * self.alpha + self.beta
        return np.clip(
            x * 255.0,
            0,
            255,
        ).astype(np.uint8)


LOWLIGHT_SIMULATOR = LowLightSimulator()
FIXED_DARK_PREPROCESS = FixedDarkPreprocess(
    alpha=CFG.fixed_dark_alpha,
    beta=CFG.fixed_dark_beta,
)


# ================================================================
# 5. Strict Albumentations compatibility
# ================================================================
_INVALID_ARG_WARNING = (
    r"Argument\(s\).*are not valid for transform.*"
)


def supports_argument(callable_object, argument_name):
    try:
        signature = inspect.signature(
            callable_object
        )
    except (TypeError, ValueError):
        return False

    return argument_name in signature.parameters


def require_one_argument(
    callable_object,
    candidate_names,
):
    for name in candidate_names:
        if supports_argument(
            callable_object,
            name,
        ):
            return name

    raise RuntimeError(
        f"{getattr(callable_object, '__name__', callable_object)} "
        f"supports none of {candidate_names}. "
        f"Installed Albumentations={getattr(A, '__version__', 'unknown')}"
    )


def make_compose(
    transforms,
    *,
    seed=None,
    strict=True,
):
    kwargs = {}

    if (
        strict
        and supports_argument(A.Compose, "strict")
    ):
        kwargs["strict"] = True

    if (
        seed is not None
        and supports_argument(A.Compose, "seed")
    ):
        kwargs["seed"] = int(seed)

    return A.Compose(
        transforms,
        **kwargs,
    )


def make_random_resized_crop(
    crop,
    scale,
    ratio,
):
    if supports_argument(
        A.RandomResizedCrop,
        "size",
    ):
        return A.RandomResizedCrop(
            size=(crop, crop),
            scale=scale,
            ratio=ratio,
            interpolation=cv2.INTER_CUBIC,
            p=1.0,
        )

    return A.RandomResizedCrop(
        height=crop,
        width=crop,
        scale=scale,
        ratio=ratio,
        interpolation=cv2.INTER_CUBIC,
        p=1.0,
    )


def make_brightness_contrast(
    brightness,
    contrast,
    p,
):
    brightness_name = require_one_argument(
        A.RandomBrightnessContrast,
        ("brightness_limit", "brightness_range"),
    )
    contrast_name = require_one_argument(
        A.RandomBrightnessContrast,
        ("contrast_limit", "contrast_range"),
    )

    kwargs = {
        brightness_name: brightness,
        contrast_name: contrast,
        "p": p,
    }
    return A.RandomBrightnessContrast(
        **kwargs
    )


def make_random_gamma(
    gamma_range,
    p,
):
    gamma_name = require_one_argument(
        A.RandomGamma,
        ("gamma_limit", "gamma_range"),
    )

    return A.RandomGamma(
        **{
            gamma_name: gamma_range,
            "p": p,
        }
    )


def _odd_kernel_bounds(limit):
    low = 3
    high = max(3, int(limit))

    if high % 2 == 0:
        high += 1

    return low, high


def make_motion_blur(limit):
    low, high = _odd_kernel_bounds(limit)

    blur_name = require_one_argument(
        A.MotionBlur,
        ("blur_limit", "blur_range"),
    )

    return A.MotionBlur(
        **{
            blur_name: (low, high),
            "p": 1.0,
        }
    )


def make_gaussian_blur(limit):
    low, high = _odd_kernel_bounds(limit)

    blur_name = require_one_argument(
        A.GaussianBlur,
        ("blur_limit", "blur_range"),
    )
    sigma_name = require_one_argument(
        A.GaussianBlur,
        ("sigma_limit", "sigma_range"),
    )

    return A.GaussianBlur(
        **{
            blur_name: (low, high),
            sigma_name: (0.1, 2.0),
            "p": 1.0,
        }
    )


def make_image_compression(
    qmin,
    qmax,
):
    if supports_argument(
        A.ImageCompression,
        "quality_range",
    ):
        return A.ImageCompression(
            quality_range=(qmin, qmax),
            p=1.0,
        )

    if (
        supports_argument(
            A.ImageCompression,
            "quality_lower",
        )
        and supports_argument(
            A.ImageCompression,
            "quality_upper",
        )
    ):
        return A.ImageCompression(
            quality_lower=qmin,
            quality_upper=qmax,
            p=1.0,
        )

    raise RuntimeError(
        "Unsupported ImageCompression API"
    )


def make_gauss_noise():
    if supports_argument(
        A.GaussNoise,
        "std_range",
    ):
        kwargs = {
            "std_range": (0.02, 0.08),
            "p": 1.0,
        }

        if supports_argument(
            A.GaussNoise,
            "mean_range",
        ):
            kwargs["mean_range"] = (0.0, 0.0)

        return A.GaussNoise(**kwargs)

    if supports_argument(
        A.GaussNoise,
        "var_limit",
    ):
        # Match std_range=(0.02, 0.08) on uint8 images:
        # sigma=5.1..20.4, variance≈26..416.
        return A.GaussNoise(
            var_limit=(
                (0.02 * 255.0) ** 2,
                (0.08 * 255.0) ** 2,
            ),
            mean=0,
            p=1.0,
        )

    raise RuntimeError(
        "Unsupported GaussNoise API"
    )


def make_downscale():
    if supports_argument(
        A.Downscale,
        "scale_range",
    ):
        return A.Downscale(
            scale_range=(0.55, 0.85),
            p=1.0,
        )

    return A.Downscale(
        scale_min=0.55,
        scale_max=0.85,
        p=1.0,
    )


def make_random_fog():
    if supports_argument(
        A.RandomFog,
        "fog_coef_range",
    ):
        return A.RandomFog(
            fog_coef_range=(0.10, 0.30),
            p=1.0,
        )

    if (
        supports_argument(
            A.RandomFog,
            "fog_coef_lower",
        )
        and supports_argument(
            A.RandomFog,
            "fog_coef_upper",
        )
    ):
        return A.RandomFog(
            fog_coef_lower=0.10,
            fog_coef_upper=0.30,
            p=1.0,
        )

    return A.RandomFog(p=1.0)


def make_random_rain():
    kwargs = {"p": 1.0}

    if supports_argument(
        A.RandomRain,
        "blur_value",
    ):
        kwargs["blur_value"] = 3

    return A.RandomRain(**kwargs)


def make_rotate(limit, p):
    angle_name = require_one_argument(
        A.Rotate,
        ("limit", "angle_range"),
    )

    kwargs = {
        angle_name: (-limit, limit),
        "border_mode": cv2.BORDER_REFLECT_101,
        "p": p,
    }

    if supports_argument(
        A.Rotate,
        "fill",
    ):
        kwargs["fill"] = 0
    elif supports_argument(
        A.Rotate,
        "value",
    ):
        kwargs["value"] = 0

    return A.Rotate(**kwargs)


def make_coarse_dropout(
    crop,
    holes,
    frac_range,
    p,
):
    if supports_argument(
        A.CoarseDropout,
        "num_holes_range",
    ):
        return A.CoarseDropout(
            num_holes_range=holes,
            hole_height_range=frac_range,
            hole_width_range=frac_range,
            p=p,
        )

    min_holes, max_holes = holes
    min_fraction, max_fraction = frac_range

    min_size = max(
        1,
        int(round(crop * min_fraction)),
    )
    max_size = max(
        min_size,
        int(round(crop * max_fraction)),
    )

    return A.CoarseDropout(
        min_holes=min_holes,
        max_holes=max_holes,
        min_height=min_size,
        max_height=max_size,
        min_width=min_size,
        max_width=max_size,
        p=p,
    )


DEGRADE_LEVELS = {
    "large": {
        "rrc_scale": (0.82, 1.00),
        "degrade_p": 0.32,
        "blur_limit": 7,
        "use_downscale": True,
        "jpg_quality": (32, 75),
        "drop_p": 0.22,
        "drop_holes": (1, 3),
        "drop_frac": (0.035, 0.09),
    },
    "mid": {
        "rrc_scale": (0.90, 1.00),
        "degrade_p": 0.24,
        "blur_limit": 5,
        "use_downscale": False,
        "jpg_quality": (45, 82),
        "drop_p": 0.16,
        "drop_holes": (1, 2),
        "drop_frac": (0.025, 0.07),
    },
    "small": {
        "rrc_scale": (0.95, 1.00),
        "degrade_p": 0.16,
        "blur_limit": 3,
        "use_downscale": False,
        "jpg_quality": (58, 90),
        "drop_p": 0.08,
        "drop_holes": (1, 1),
        "drop_frac": (0.02, 0.05),
    },
}


def build_train_aug(
    mean,
    std,
    crop,
    level,
):
    config = DEGRADE_LEVELS[level]

    degradation_ops = [
        make_motion_blur(
            config["blur_limit"]
        ),
        make_gaussian_blur(
            config["blur_limit"]
        ),
        make_image_compression(
            *config["jpg_quality"]
        ),
        make_gauss_noise(),
        A.ISONoise(p=1.0),
        make_random_fog(),
        make_random_rain(),
    ]

    if config["use_downscale"]:
        degradation_ops.append(
            make_downscale()
        )

    transforms = [
        make_random_resized_crop(
            crop=crop,
            scale=config["rrc_scale"],
            ratio=(0.90, 1.11),
        ),
        make_brightness_contrast(
            brightness=(-0.10, 0.12),
            contrast=(-0.12, 0.15),
            p=0.32,
        ),
        make_random_gamma(
            (88, 116),
            p=0.16,
        ),
        A.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.10,
            hue=0.02,
            p=0.25,
        ),
        A.OneOf(
            degradation_ops,
            p=config["degrade_p"],
        ),
        make_coarse_dropout(
            crop=crop,
            holes=config["drop_holes"],
            frac_range=config["drop_frac"],
            p=config["drop_p"],
        ),
        make_rotate(
            limit=12,
            p=0.45,
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ]

    return make_compose(
        transforms,
        strict=True,
    )


def build_train_augs(
    mean,
    std,
    crop,
):
    # Invalid transform arguments must stop execution instead of
    # silently reverting to defaults.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            category=UserWarning,
            message=_INVALID_ARG_WARNING,
        )

        return {
            level: build_train_aug(
                mean,
                std,
                crop,
                level,
            )
            for level in DEGRADE_LEVELS
        }


def build_eval_aug(
    mean,
    std,
    crop,
    resize,
):
    transforms = [
        A.SmallestMaxSize(
            max_size=resize,
            interpolation=cv2.INTER_CUBIC,
        ),
        A.CenterCrop(
            height=crop,
            width=crop,
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ]

    return make_compose(
        transforms,
        strict=True,
    )


def print_augmentation_api_report():
    print("=" * 72)
    print(
        "Albumentations version:",
        getattr(A, "__version__", "unknown"),
    )

    transforms = [
        A.MotionBlur,
        A.GaussianBlur,
        A.RandomBrightnessContrast,
        A.RandomGamma,
        A.Rotate,
    ]

    for transform in transforms:
        try:
            signature = inspect.signature(
                transform
            )
        except Exception:
            signature = "<unavailable>"

        print(
            f"{transform.__name__}: {signature}"
        )

    print("=" * 72)


def validate_augmentation_pipeline(
    train_augs,
    eval_aug,
    crop,
):
    dummy = np.random.default_rng(
        CFG.seed
    ).integers(
        0,
        256,
        size=(128, 160, 3),
        dtype=np.uint8,
    )

    for level, transform in train_augs.items():
        for _ in range(3):
            output = transform(
                image=dummy
            )["image"]

            if tuple(output.shape) != (
                3,
                crop,
                crop,
            ):
                raise RuntimeError(
                    f"{level} augmentation output shape "
                    f"is {tuple(output.shape)}"
                )

            if not torch.isfinite(output).all():
                raise RuntimeError(
                    f"{level} augmentation produced NaN/Inf"
                )

    output = eval_aug(
        image=dummy
    )["image"]

    if tuple(output.shape) != (
        3,
        crop,
        crop,
    ):
        raise RuntimeError(
            f"Eval augmentation output shape "
            f"is {tuple(output.shape)}"
        )

    print(
        "✓ Augmentation pipeline validation passed; "
        "no invalid Albumentations arguments were accepted."
    )


class SignDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        aug,
        rois=None,
        use_roi=False,
        train_mode=False,
        size_aware=False,
        lowlight_simulator=None,
        preprocess=None,
        small_max=64,
        large_min=96,
    ):
        self.paths = list(paths)
        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )
        self.aug = aug
        self.rois = rois
        self.use_roi = (
            use_roi
            and rois is not None
        )
        self.train_mode = train_mode
        self.size_aware = (
            size_aware
            and isinstance(aug, dict)
        )
        self.lowlight_simulator = (
            lowlight_simulator
        )
        self.preprocess = preprocess
        self.small_max = int(small_max)
        self.large_min = int(large_min)

    def __len__(self):
        return len(self.paths)

    def set_worker_seed(self, seed):
        _set_transform_seed(
            self.aug,
            int(seed),
        )

    def _level(self, height, width):
        if not self.size_aware:
            return "large"

        min_side = min(height, width)

        if min_side < self.small_max:
            return "small"

        if min_side >= self.large_min:
            return "large"

        return "mid"

    def __getitem__(self, index):
        image = cv2.imread(
            self.paths[index],
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(
                f"Failed to read image: "
                f"{self.paths[index]}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if (
            self.use_roi
            and self.rois[index] is not None
        ):
            x1, y1, x2, y2 = self.rois[index]
            height, width = image.shape[:2]

            x1 = max(
                0,
                min(int(x1), width - 1),
            )
            x2 = max(
                x1 + 1,
                min(int(x2) + 1, width),
            )
            y1 = max(
                0,
                min(int(y1), height - 1),
            )
            y2 = max(
                y1 + 1,
                min(int(y2) + 1, height),
            )

            image = image[
                y1:y2,
                x1:x2,
            ]

        height, width = image.shape[:2]
        level = self._level(
            height,
            width,
        )

        if (
            self.train_mode
            and self.lowlight_simulator is not None
        ):
            image = self.lowlight_simulator(
                image,
                level,
            )

        if self.preprocess is not None:
            image = self.preprocess(image)

        transform = (
            self.aug[level]
            if isinstance(self.aug, dict)
            else self.aug
        )
        tensor = transform(
            image=image
        )["image"]

        return (
            tensor,
            int(self.labels[index]),
        )


# ================================================================
# 6. Train/validation split
# ================================================================
def make_group_fn(strategy):
    if strategy == "gtsrb_track":
        def group_fn(path, label):
            stem = Path(path).stem
            parts = stem.split("_")

            if len(parts) >= 3:
                return (
                    f"{label}:"
                    f"{parts[0]}_{parts[1]}"
                )

            return f"{label}:{stem}"

        return group_fn

    return None


def stratified_split():
    sample_count = len(PATHS)

    validation_count = max(
        NUM_CLASSES,
        int(round(
            sample_count * CFG.val_ratio
        )),
    )
    validation_count = min(
        validation_count,
        sample_count - NUM_CLASSES,
    )

    if validation_count < NUM_CLASSES:
        raise RuntimeError(
            "Not enough samples to cover all classes "
            "in both train and validation sets"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_count,
        random_state=CFG.seed,
    )

    train_indices, val_indices = next(
        splitter.split(
            PATHS,
            LABELS,
        )
    )

    return (
        np.asarray(train_indices),
        np.asarray(val_indices),
    )


def grouped_per_class_split(group_fn):
    groups = np.asarray([
        group_fn(path, label)
        for path, label in zip(
            PATHS,
            LABELS,
        )
    ])

    class_to_groups = defaultdict(set)
    group_to_indices = defaultdict(list)

    for index, (label, group) in enumerate(
        zip(
            LABELS.tolist(),
            groups.tolist(),
        )
    ):
        class_to_groups[int(label)].add(group)
        group_to_indices[group].append(index)

    minimum_groups = min(
        len(group_set)
        for group_set in class_to_groups.values()
    )

    if minimum_groups < 2:
        print(
            f"⚠ A class has only {minimum_groups} entity group(s); "
            "falling back to stratified split"
        )
        return stratified_split()

    rng = np.random.default_rng(
        CFG.seed
    )
    train_groups = set()
    val_groups = set()

    for label in range(NUM_CLASSES):
        group_list = sorted(
            class_to_groups[label]
        )
        rng.shuffle(group_list)

        val_group_count = max(
            1,
            int(round(
                len(group_list)
                * CFG.val_ratio
            )),
        )
        val_group_count = min(
            val_group_count,
            len(group_list) - 1,
        )

        val_groups.update(
            group_list[:val_group_count]
        )
        train_groups.update(
            group_list[val_group_count:]
        )

    train_indices = []
    val_indices = []

    for group in sorted(train_groups):
        train_indices.extend(
            group_to_indices[group]
        )

    for group in sorted(val_groups):
        val_indices.extend(
            group_to_indices[group]
        )

    return (
        np.asarray(
            sorted(train_indices),
            dtype=np.int64,
        ),
        np.asarray(
            sorted(val_indices),
            dtype=np.int64,
        ),
    )


def split_train_val():
    group_fn = make_group_fn(
        PROF["group"]
    )

    if group_fn is None:
        train_indices, val_indices = (
            stratified_split()
        )
        split_name = "stratified split"
    else:
        train_indices, val_indices = (
            grouped_per_class_split(
                group_fn
            )
        )
        split_name = (
            "per-class entity-group split"
        )

    train_labels = LABELS[train_indices]
    val_labels = LABELS[val_indices]

    all_classes = set(range(NUM_CLASSES))
    train_classes = set(
        train_labels.tolist()
    )
    val_classes = set(
        val_labels.tolist()
    )

    if (
        train_classes != all_classes
        or val_classes != all_classes
    ):
        print(
            "⚠ Group split did not cover all classes; "
            "falling back to stratified split"
        )
        train_indices, val_indices = (
            stratified_split()
        )
        train_labels = LABELS[train_indices]
        val_labels = LABELS[val_indices]
        split_name = (
            "stratified split (fallback)"
        )

    assert set(
        train_labels.tolist()
    ) == all_classes
    assert set(
        val_labels.tolist()
    ) == all_classes
    assert not set(
        train_indices.tolist()
    ).intersection(
        val_indices.tolist()
    )

    print(
        f"✓ {split_name}: "
        f"train={len(train_indices)}, "
        f"val={len(val_indices)}, "
        f"val classes="
        f"{len(set(val_labels.tolist()))}/"
        f"{NUM_CLASSES}"
    )

    return (
        train_indices,
        val_indices,
    )


# ================================================================
# 7. Loss, MixUp and EMA
# ================================================================
class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(
        self,
        num_classes,
        prior,
        tau=1.0,
        smoothing=0.05,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2"
            )

        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)

        adjusted_prior = prior.clamp_min(
            1e-12
        )
        self.register_buffer(
            "logit_adjustment",
            float(tau)
            * torch.log(adjusted_prior),
        )

    def _smoothed_target(
        self,
        reference,
        labels,
    ):
        with torch.no_grad():
            target = torch.full_like(
                reference,
                self.smoothing
                / (self.num_classes - 1),
            )
            target.scatter_(
                1,
                labels.unsqueeze(1),
                1.0 - self.smoothing,
            )

        return target

    def forward(
        self,
        logits,
        labels_a,
        labels_b,
        lam,
    ):
        adjusted_logits = (
            logits
            + self.logit_adjustment
        )
        log_probability = F.log_softmax(
            adjusted_logits,
            dim=-1,
        )

        target_a = self._smoothed_target(
            log_probability,
            labels_a,
        )
        target_b = self._smoothed_target(
            log_probability,
            labels_b,
        )
        target = (
            lam * target_a
            + (1.0 - lam) * target_b
        )

        return -(
            target * log_probability
        ).sum(
            dim=-1
        ).mean()


def mixup_batch(images, labels):
    if (
        not CFG.use_mixup
        or np.random.random()
        > CFG.mixup_prob
        or images.size(0) < 2
    ):
        return (
            images,
            labels,
            labels,
            1.0,
        )

    lam = float(
        np.random.beta(
            CFG.mixup_alpha,
            CFG.mixup_alpha,
        )
    )
    permutation = torch.randperm(
        images.size(0),
        device=images.device,
    )

    mixed_images = (
        lam * images
        + (1.0 - lam)
        * images[permutation]
    )

    return (
        mixed_images,
        labels,
        labels[permutation],
        lam,
    )


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(
            model
        ).eval()
        self.decay = float(decay)
        self.num_updates = 0

        for parameter in (
            self.module.parameters()
        ):
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1

        decay = min(
            self.decay,
            (
                1.0 + self.num_updates
            )
            / (
                10.0 + self.num_updates
            ),
        )

        model_state = model.state_dict()
        ema_state = (
            self.module.state_dict()
        )

        for key, ema_value in (
            ema_state.items()
        ):
            model_value = (
                model_state[key].detach()
            )

            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(
                    model_value,
                    alpha=1.0 - decay,
                )
            else:
                ema_value.copy_(model_value)


# ================================================================
# 8. Model and pretrained weights
# ================================================================
def create_timm_model(
    pretrained=False,
    num_classes=None,
):
    if num_classes is None:
        num_classes = NUM_CLASSES

    kwargs = {
        "pretrained": pretrained,
        "num_classes": int(num_classes),
        "drop_path_rate": MCFG["drop_path"],
    }

    try:
        return timm.create_model(
            MCFG["name"],
            **kwargs,
        )
    except TypeError:
        kwargs.pop(
            "drop_path_rate",
            None,
        )
        return timm.create_model(
            MCFG["name"],
            **kwargs,
        )


def unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a dictionary"
        )

    for key in (
        "state_dict_ema",
        "model_ema",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    return checkpoint


def strip_prefix_if_common(
    state_dict,
    prefix,
):
    keys = list(state_dict.keys())

    if not keys:
        return state_dict

    ratio = sum(
        key.startswith(prefix)
        for key in keys
    ) / len(keys)

    if ratio < 0.80:
        return state_dict

    return {
        (
            key[len(prefix):]
            if key.startswith(prefix)
            else key
        ): value
        for key, value in state_dict.items()
    }


def load_local_pretrained(
    model,
    weight_path,
):
    print(
        f"Loading local pretrained weights: "
        f"{weight_path}"
    )

    if str(weight_path).lower().endswith(
        ".safetensors"
    ):
        checkpoint = safe_load_file(
            weight_path
        )
    else:
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
        )

    checkpoint = unwrap_checkpoint(
        checkpoint
    )

    for prefix in (
        "module.",
        "_orig_mod.",
        "model.",
    ):
        checkpoint = strip_prefix_if_common(
            checkpoint,
            prefix,
        )

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for key, value in checkpoint.items():
        if not isinstance(
            value,
            torch.Tensor,
        ):
            continue

        if (
            key in model_state
            and model_state[key].shape
            == value.shape
        ):
            loadable[key] = value
        else:
            skipped.append(key)

    missing, unexpected = (
        model.load_state_dict(
            loadable,
            strict=False,
        )
    )

    loaded_numel = sum(
        tensor.numel()
        for tensor in loadable.values()
    )
    total_numel = sum(
        tensor.numel()
        for tensor in model_state.values()
    )
    coverage = (
        loaded_numel
        / max(total_numel, 1)
    )

    print(
        f"  ✓ Loaded tensors: {len(loadable)}"
    )
    print(
        f"  Skipped tensors: {len(skipped)}"
    )
    print(
        "  Parameter/buffer element coverage: "
        f"{coverage * 100:.2f}%"
    )
    print(
        f"  Missing keys: {len(missing)}"
    )
    print(
        f"  Unexpected keys: {len(unexpected)}"
    )

    if coverage < 0.65:
        raise RuntimeError(
            "Pretrained coverage is only "
            f"{coverage * 100:.2f}%. "
            f"Model {MCFG['name']} may not match "
            "the supplied weights."
        )


def load_online_pretrained():
    print(
        "Trying timm online pretrained weights: "
        f"{MCFG['name']}"
    )

    try:
        model = create_timm_model(
            pretrained=True,
            num_classes=NUM_CLASSES,
        )
        print(
            "  ✓ Pretrained weights loaded; "
            "classifier adapted"
        )
        return model

    except Exception as first_error:
        try:
            base_model = create_timm_model(
                pretrained=True,
                num_classes=1000,
            )

            if not hasattr(
                base_model,
                "reset_classifier",
            ):
                raise RuntimeError(
                    "Model has no reset_classifier"
                )

            base_model.reset_classifier(
                NUM_CLASSES
            )
            print(
                "  ✓ Loaded original classifier "
                "then rebuilt target classifier"
            )
            return base_model

        except Exception as second_error:
            raise RuntimeError(
                "Online pretrained loading failed.\n"
                f"First error: {first_error}\n"
                "Classifier-reset error: "
                f"{second_error}"
            ) from second_error


def build_model():
    local_path = LOCAL_WEIGHTS.get(
        MODEL_KEY,
        "",
    )

    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                "Configured local weight file "
                f"does not exist: {local_path}"
            )

        model = create_timm_model(
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
        load_local_pretrained(
            model,
            local_path,
        )

    elif ALLOW_HF_DOWNLOAD:
        try:
            model = load_online_pretrained()
        except Exception as error:
            if REQUIRE_PRETRAINED:
                raise RuntimeError(
                    f"Could not obtain pretrained "
                    f"weights for {MCFG['name']}.\n"
                    "Upload model.safetensors and "
                    "fill LOCAL_WEIGHTS.\n"
                    f"Original error: {error}"
                ) from error

            print(
                "⚠ Pretrained loading failed; "
                f"using random initialization: {error}"
            )
            model = create_timm_model(
                pretrained=False,
                num_classes=NUM_CLASSES,
            )

    else:
        if REQUIRE_PRETRAINED:
            raise RuntimeError(
                f"{MODEL_KEY} has no local weights "
                "and online download is disabled"
            )

        print("⚠ Using random initialization")
        model = create_timm_model(
            pretrained=False,
            num_classes=NUM_CLASSES,
        )

    if MCFG.get(
        "grad_checkpointing",
        False,
    ):
        if hasattr(
            model,
            "set_grad_checkpointing",
        ):
            try:
                model.set_grad_checkpointing(
                    enable=True
                )
            except TypeError:
                model.set_grad_checkpointing(
                    True
                )
            print(
                "✓ Gradient checkpointing enabled"
            )
        else:
            print(
                "⚠ Model has no "
                "set_grad_checkpointing; skipped"
            )

    model = model.to(DEVICE)

    if MCFG["channels_last"]:
        model = model.to(
            memory_format=torch.channels_last
        )

    return model


# ================================================================
# 9. Optimizer and scheduler
# ================================================================
def get_classifier_param_ids(model):
    try:
        classifier = model.get_classifier()
    except Exception:
        classifier = None

    if isinstance(
        classifier,
        nn.Module,
    ):
        return {
            id(parameter)
            for parameter
            in classifier.parameters()
        }

    return set()


def make_optimizer(model):
    classifier_ids = (
        get_classifier_param_ids(model)
    )

    groups = {
        "body_decay": [],
        "body_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        is_head = (
            id(parameter)
            in classifier_ids
        )
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )

        if is_head and no_decay:
            groups[
                "head_no_decay"
            ].append(parameter)
        elif is_head:
            groups[
                "head_decay"
            ].append(parameter)
        elif no_decay:
            groups[
                "body_no_decay"
            ].append(parameter)
        else:
            groups[
                "body_decay"
            ].append(parameter)

    parameter_groups = []

    def add_group(
        parameters,
        lr,
        weight_decay,
        group_name,
    ):
        if parameters:
            parameter_groups.append({
                "params": parameters,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": group_name,
            })

    base_lr = MCFG["lr"]
    head_lr = (
        base_lr
        * MCFG["head_lr_mult"]
    )

    add_group(
        groups["body_decay"],
        base_lr,
        CFG.weight_decay,
        "body_decay",
    )
    add_group(
        groups["body_no_decay"],
        base_lr,
        0.0,
        "body_no_decay",
    )
    add_group(
        groups["head_decay"],
        head_lr,
        CFG.head_weight_decay,
        "head_decay",
    )
    add_group(
        groups["head_no_decay"],
        head_lr,
        0.0,
        "head_no_decay",
    )

    return optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def make_scheduler(
    optimizer,
    updates_per_epoch,
):
    total_steps = max(
        1,
        CFG.epochs
        * updates_per_epoch,
    )
    warmup_steps = max(
        1,
        CFG.warmup_epochs
        * updates_per_epoch,
    )

    def lr_factor(step):
        if step < warmup_steps:
            return (
                (step + 1)
                / warmup_steps
            )

        progress = (
            (step - warmup_steps)
            / max(
                1,
                total_steps - warmup_steps,
            )
        )
        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

        return (
            CFG.min_lr_ratio
            + (
                1.0
                - CFG.min_lr_ratio
            )
            * cosine
        )

    return LambdaLR(
        optimizer,
        lr_lambda=[
            lr_factor
            for _ in optimizer.param_groups
        ],
    )


# ================================================================
# 10. Saving
# ================================================================
def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }


def head_prefix(model):
    config = (
        getattr(
            model,
            "pretrained_cfg",
            None,
        )
        or getattr(
            model,
            "default_cfg",
            {},
        )
        or {}
    )

    return config.get(
        "classifier",
        "head",
    )


def save_backbone(model, tag):
    model_path = os.path.join(
        OUT_DIR,
        f"{MODEL_KEY}_stage1_{tag}.pth",
    )

    torch.save(
        cpu_state_dict(model),
        model_path,
    )

    meta = {
        "dataset": DATASET,
        "model_key": MODEL_KEY,
        "timm_name": MCFG["name"],
        "class_names": CLASS_NAMES,
        "classid_to_idx": (
            CLASSID_TO_IDX
            if CLASSID_TO_IDX is not None
            else {}
        ),
        "head_prefix": head_prefix(model),
        "crop": MCFG["crop"],
        "resize": MCFG["resize"],
        "channels_last": MCFG[
            "channels_last"
        ],
        "albumentations_version": getattr(
            A,
            "__version__",
            "unknown",
        ),
        "selection": {
            "monitor_dark_val": (
                CFG.monitor_dark_val
            ),
            "dark_score_weight": (
                CFG.dark_score_weight
            ),
            "fixed_dark_alpha": (
                CFG.fixed_dark_alpha
            ),
            "fixed_dark_beta": (
                CFG.fixed_dark_beta
            ),
        },
        "lowlight_policy": (
            LOWLIGHT_POLICY
        ),
        "degrade_levels": (
            DEGRADE_LEVELS
        ),
        "small_max": SMALL_MAX,
        "large_min": LARGE_MIN,
    }

    meta_path = os.path.join(
        OUT_DIR,
        f"{MODEL_KEY}_stage1_meta.json",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            meta,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  Saved: {model_path}")


def save_history(history):
    history_path = os.path.join(
        OUT_DIR,
        f"{MODEL_KEY}_stage1_history.json",
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ================================================================
# 11. Training and evaluation
# ================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        if MCFG["channels_last"]:
            images = images.contiguous(
                memory_format=torch.channels_last
            )

        with amp_autocast():
            logits = model(images)

        predictions.append(
            logits.argmax(
                dim=1
            ).cpu().numpy()
        )
        targets.append(
            labels.numpy()
        )

    if not predictions:
        raise RuntimeError(
            "Evaluation DataLoader is empty"
        )

    predictions = np.concatenate(
        predictions
    )
    targets = np.concatenate(
        targets
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    return (
        float(accuracy),
        float(macro_f1),
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    ema,
    epoch,
):
    model.train()
    optimizer.zero_grad(
        set_to_none=True
    )

    accumulation_steps = int(
        MCFG.get("accum", 1)
    )
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=(
            f"[{DATASET}/{MODEL_KEY}] "
            f"epoch {epoch + 1}"
        ),
    )

    for step, (images, labels) in progress:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if MCFG["channels_last"]:
            images = images.contiguous(
                memory_format=torch.channels_last
            )

        (
            images,
            labels_a,
            labels_b,
            lam,
        ) = mixup_batch(
            images,
            labels,
        )

        group_start = (
            step // accumulation_steps
        ) * accumulation_steps
        current_group_size = min(
            accumulation_steps,
            len(loader) - group_start,
        )

        with amp_autocast():
            logits = model(images)
            raw_loss = criterion(
                logits,
                labels_a,
                labels_b,
                lam,
            )
            loss = (
                raw_loss
                / current_group_size
            )

        if not torch.isfinite(raw_loss):
            raise FloatingPointError(
                "Non-finite loss detected: "
                f"{raw_loss.item()}"
            )

        scaler.scale(
            loss
        ).backward()

        should_update = (
            (step + 1)
            % accumulation_steps
            == 0
            or (step + 1)
            == len(loader)
        )

        if should_update:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CFG.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(
                set_to_none=True
            )
            scheduler.step()

            if ema is not None:
                ema.update(model)

        batch_size = images.size(0)
        total_loss += (
            raw_loss.item()
            * batch_size
        )
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{raw_loss.item():.4f}",
            lr=(
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ),
        )

    return (
        total_loss
        / max(total_samples, 1)
    )


# ================================================================
# 12. Main
# ================================================================
def main():
    print("=" * 72)
    print(f"DATASET={DATASET}")
    print(f"MODEL_KEY={MODEL_KEY}")
    print(f"TIMM_MODEL={MCFG['name']}")
    print(f"DEVICE={DEVICE}")
    print(f"AMP_DTYPE={AMP_DTYPE}")
    print(f"OUT_DIR={OUT_DIR}")
    print("=" * 72)

    print_augmentation_api_report()

    temp_model = create_timm_model(
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    data_config = resolve_data_config(
        {},
        model=temp_model,
    )
    mean = list(data_config["mean"])
    std = list(data_config["std"])
    del temp_model

    train_augs = build_train_augs(
        mean=mean,
        std=std,
        crop=MCFG["crop"],
    )
    eval_aug = build_eval_aug(
        mean=mean,
        std=std,
        crop=MCFG["crop"],
        resize=MCFG["resize"],
    )

    if CFG.validate_augmentation_pipeline:
        validate_augmentation_pipeline(
            train_augs,
            eval_aug,
            MCFG["crop"],
        )

    train_indices, val_indices = (
        split_train_val()
    )

    train_paths = [
        PATHS[index]
        for index in train_indices
    ]
    train_labels = LABELS[
        train_indices
    ]
    val_paths = [
        PATHS[index]
        for index in val_indices
    ]
    val_labels = LABELS[
        val_indices
    ]

    train_dataset = SignDataset(
        train_paths,
        train_labels,
        train_augs,
        train_mode=True,
        size_aware=SIZE_AWARE_AUG,
        lowlight_simulator=(
            LOWLIGHT_SIMULATOR
        ),
        small_max=SMALL_MAX,
        large_min=LARGE_MIN,
    )
    clean_val_dataset = SignDataset(
        val_paths,
        val_labels,
        eval_aug,
        train_mode=False,
    )
    dark_val_dataset = SignDataset(
        val_paths,
        val_labels,
        eval_aug,
        train_mode=False,
        preprocess=(
            FIXED_DARK_PREPROCESS
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        CFG.seed
    )

    loader_common = {
        "num_workers": CFG.num_workers,
        "pin_memory": USE_CUDA,
        "worker_init_fn": seed_worker,
        "persistent_workers": (
            CFG.num_workers > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=MCFG["batch"],
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_common,
    )
    clean_val_loader = DataLoader(
        clean_val_dataset,
        batch_size=MCFG["batch"],
        shuffle=False,
        drop_last=False,
        **loader_common,
    )
    dark_val_loader = DataLoader(
        dark_val_dataset,
        batch_size=MCFG["batch"],
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    test_loader = None
    dark_test_loader = None

    if (
        TEST is not None
        and CFG.final_test_eval
    ):
        test_dataset = SignDataset(
            TEST["paths"],
            TEST["labels"],
            eval_aug,
            rois=TEST.get("rois"),
            use_roi=PROF.get(
                "use_test_roi",
                False,
            ),
            train_mode=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=MCFG["batch"],
            shuffle=False,
            drop_last=False,
            **loader_common,
        )

        if CFG.final_test_dark_eval:
            dark_test_dataset = SignDataset(
                TEST["paths"],
                TEST["labels"],
                eval_aug,
                rois=TEST.get("rois"),
                use_roi=PROF.get(
                    "use_test_roi",
                    False,
                ),
                train_mode=False,
                preprocess=(
                    FIXED_DARK_PREPROCESS
                ),
            )
            dark_test_loader = DataLoader(
                dark_test_dataset,
                batch_size=MCFG["batch"],
                shuffle=False,
                drop_last=False,
                **loader_common,
            )

    train_counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    if np.any(train_counts == 0):
        raise RuntimeError(
            "Train split contains a zero-sample class"
        )

    prior = torch.tensor(
        train_counts / train_counts.sum(),
        dtype=torch.float32,
        device=DEVICE,
    )

    model = build_model()

    criterion = LogitAdjustedCrossEntropy(
        num_classes=NUM_CLASSES,
        prior=prior,
        tau=CFG.la_tau,
        smoothing=CFG.label_smoothing,
    ).to(DEVICE)

    optimizer = make_optimizer(model)

    accumulation_steps = int(
        MCFG.get("accum", 1)
    )
    updates_per_epoch = math.ceil(
        len(train_loader)
        / accumulation_steps
    )
    scheduler = make_scheduler(
        optimizer,
        updates_per_epoch,
    )

    ema = (
        ModelEMA(
            model,
            CFG.ema_decay,
        )
        if CFG.use_ema
        else None
    )

    best_score = -1.0
    best_clean_f1 = -1.0
    best_dark_f1 = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            epoch=epoch,
        )

        (
            raw_clean_acc,
            raw_clean_f1,
        ) = evaluate(
            model,
            clean_val_loader,
        )

        raw_dark_acc = None
        raw_dark_f1 = None

        if CFG.monitor_dark_val:
            (
                raw_dark_acc,
                raw_dark_f1,
            ) = evaluate(
                model,
                dark_val_loader,
            )

        candidates = [{
            "name": "RAW",
            "net": model,
            "clean_acc": raw_clean_acc,
            "clean_f1": raw_clean_f1,
            "dark_acc": raw_dark_acc,
            "dark_f1": raw_dark_f1,
        }]

        if (
            ema is not None
            and epoch
            >= CFG.ema_warmup_epochs
        ):
            (
                ema_clean_acc,
                ema_clean_f1,
            ) = evaluate(
                ema.module,
                clean_val_loader,
            )

            ema_dark_acc = None
            ema_dark_f1 = None

            if CFG.monitor_dark_val:
                (
                    ema_dark_acc,
                    ema_dark_f1,
                ) = evaluate(
                    ema.module,
                    dark_val_loader,
                )

            candidates.append({
                "name": "EMA",
                "net": ema.module,
                "clean_acc": (
                    ema_clean_acc
                ),
                "clean_f1": (
                    ema_clean_f1
                ),
                "dark_acc": (
                    ema_dark_acc
                ),
                "dark_f1": (
                    ema_dark_f1
                ),
            })

        for candidate in candidates:
            if CFG.monitor_dark_val:
                candidate[
                    "selection_score"
                ] = (
                    (
                        1.0
                        - CFG.dark_score_weight
                    )
                    * candidate["clean_f1"]
                    + CFG.dark_score_weight
                    * candidate["dark_f1"]
                )
            else:
                candidate[
                    "selection_score"
                ] = candidate["clean_f1"]

        selected = max(
            candidates,
            key=lambda item: item[
                "selection_score"
            ],
        )

        message = (
            f"\nEpoch {epoch + 1:02d} | "
            f"train loss={train_loss:.4f}"
        )

        for candidate in candidates:
            message += (
                f"\n  {candidate['name']} clean: "
                f"Acc={candidate['clean_acc'] * 100:.2f} "
                f"F1={candidate['clean_f1'] * 100:.2f}"
            )

            if CFG.monitor_dark_val:
                message += (
                    " | fixed-dark: "
                    f"Acc={candidate['dark_acc'] * 100:.2f} "
                    f"F1={candidate['dark_f1'] * 100:.2f}"
                )

            message += (
                " | select-score="
                f"{candidate['selection_score'] * 100:.2f}"
            )

        message += (
            f"\n  -> selected "
            f"{selected['name']}"
        )
        print(message)

        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "selected": selected["name"],
            "selection_score": (
                selected["selection_score"]
            ),
            "clean_acc": (
                selected["clean_acc"]
            ),
            "clean_f1": (
                selected["clean_f1"]
            ),
            "dark_acc": (
                selected["dark_acc"]
            ),
            "dark_f1": (
                selected["dark_f1"]
            ),
            "lr_body": (
                optimizer.param_groups[0]["lr"]
            ),
        }
        history.append(history_row)
        save_history(history)

        if (
            selected["selection_score"]
            > best_score
        ):
            best_score = (
                selected["selection_score"]
            )
            best_clean_f1 = (
                selected["clean_f1"]
            )
            best_dark_f1 = (
                selected["dark_f1"]
                if selected["dark_f1"]
                is not None
                else -1.0
            )
            best_epoch = epoch + 1
            no_improve = 0

            best_state = cpu_state_dict(
                selected["net"]
            )
            save_backbone(
                selected["net"],
                "best",
            )

            print(
                "  ↑ New best: "
                f"score={best_score * 100:.2f}, "
                f"clean F1={best_clean_f1 * 100:.2f}"
                + (
                    f", dark F1="
                    f"{best_dark_f1 * 100:.2f}"
                    if CFG.monitor_dark_val
                    else ""
                )
            )
        else:
            no_improve += 1
            print(
                "  No improvement: "
                f"{no_improve}/{CFG.patience}"
            )

            if (
                no_improve
                >= CFG.patience
            ):
                print(
                    "⏹ Early stopping after "
                    f"{CFG.patience} epochs "
                    "without improvement"
                )
                break

    save_backbone(
        model,
        "last",
    )

    if best_state is None:
        raise RuntimeError(
            "Training ended without best_state"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    (
        final_clean_val_acc,
        final_clean_val_f1,
    ) = evaluate(
        model,
        clean_val_loader,
    )

    print("\n" + "=" * 72)
    print(
        "Stage1 finished | "
        f"best epoch={best_epoch} | "
        f"selection score="
        f"{best_score * 100:.2f}"
    )
    print(
        "Best clean val: "
        f"Acc={final_clean_val_acc * 100:.2f} "
        f"Macro-F1={final_clean_val_f1 * 100:.2f}"
    )

    if CFG.monitor_dark_val:
        (
            final_dark_val_acc,
            final_dark_val_f1,
        ) = evaluate(
            model,
            dark_val_loader,
        )
        print(
            "Best fixed-dark val: "
            f"Acc={final_dark_val_acc * 100:.2f} "
            f"Macro-F1={final_dark_val_f1 * 100:.2f}"
        )

    if test_loader is not None:
        test_acc, test_f1 = evaluate(
            model,
            test_loader,
        )
        print(
            "Official test clean: "
            f"Acc={test_acc * 100:.2f} "
            f"Macro-F1={test_f1 * 100:.2f}"
        )

    if dark_test_loader is not None:
        (
            dark_test_acc,
            dark_test_f1,
        ) = evaluate(
            model,
            dark_test_loader,
        )
        print(
            "Official test fixed-dark-mid: "
            f"Acc={dark_test_acc * 100:.2f} "
            f"Macro-F1={dark_test_f1 * 100:.2f}"
        )

    print(f"Output directory: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
