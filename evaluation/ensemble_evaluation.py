"""
7-model Stage1 Backbone Evaluation
==================================

Single-model / selectable weighted soft-voting ensemble evaluation for GTSRB.

Features
--------
1. Select one model or any subset of the seven supported models.
2. Load every selected Stage1 checkpoint independently.
3. Reorder every model output to canonical GTSRB ClassId order (0..42).
4. Perform weighted soft-voting on softmax probabilities.
5. Report individual-model and ensemble Accuracy / Macro-F1.
6. Show ensemble per-class F1, confusion matrix, errors and low-confidence samples.
7. Optionally show t-SNE for one chosen backbone.
8. Evaluate clean / blur / dark / jpeg / noise / fog / rain robustness.
9. Display results only; no evaluation-result files are written.
-----------------------------

"""

import ast
import gc
import inspect
import json
import math
import os
import random
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.data import resolve_data_config

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None


# ================================================================
# 0. Configuration
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"evaluation", "eval", "scripts"}
    else SCRIPT_DIR
)
PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

STAGE1_MODEL_ROOT = Path(
    os.getenv(
        "STAGE1_MODEL_ROOT",
        str(PROJECT_ROOT / "weights" / "stage1"),
    )
).expanduser()

GTSRB_ROOT = Path(
    os.getenv(
        "GTSRB_ROOT",
        str(PROJECT_ROOT / "data" / "GTSRB"),
    )
).expanduser()


def env_path(name, default):
    """Return an environment-overridden path as a string."""
    return str(Path(os.getenv(name, str(default))).expanduser())


# One model:
# SELECTED_MODELS = ["maxvit"]
#
# Multiple models:
DEFAULT_SELECTED_MODELS = ["effnet", "caformer", "maxvit"]

# You can override without editing this file:
#   SELECTED_MODELS=effnet,caformer,maxvit python ensemble_evaluation.py
_selected_from_env = os.getenv("SELECTED_MODELS", "").strip()
if _selected_from_env:
    SELECTED_MODELS = [
        item.strip().lower()
        for item in _selected_from_env.split(",")
        if item.strip()
    ]
else:
    SELECTED_MODELS = list(DEFAULT_SELECTED_MODELS)


# None -> equal weights.
#
# To use custom weights, replace None with a dict such as:
# ENSEMBLE_WEIGHTS = {
#     "effnet": 0.45,
#     "caformer": 0.35,
#     "maxvit": 0.20,
# }
#
# Or set an environment variable:
#   ENSEMBLE_WEIGHTS_JSON='{"effnet":0.45,"caformer":0.35,"maxvit":0.20}'
ENSEMBLE_WEIGHTS = None

_weights_json = os.getenv("ENSEMBLE_WEIGHTS_JSON", "").strip()
if _weights_json:
    try:
        ENSEMBLE_WEIGHTS = json.loads(_weights_json)
    except json.JSONDecodeError as error:
        raise ValueError(
            "ENSEMBLE_WEIGHTS_JSON must be a valid JSON object, for example "
            """'{"effnet":0.45,"caformer":0.35,"maxvit":0.20}'."""
        ) from error


# The model whose backbone features are used for t-SNE.
# None -> the first selected model.
TSNE_MODEL_KEY = os.getenv("TSNE_MODEL_KEY", "").strip().lower() or None


MODEL_REGISTRY = {
    "convnext": {
        "ckpt": env_path(
            "CONVNEXT_CKPT",
            STAGE1_MODEL_ROOT / "model_v1.1_convnext_gtsrb.pth",
        ),
        "meta": env_path(
            "CONVNEXT_META",
            STAGE1_MODEL_ROOT / "model_v1.1_convnext_stage1_meta.json",
        ),
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "crop": 256,
        "resize": 288,
        "eval_batch": 32,
        "channels_last": True,
    },
    "swin": {
        "ckpt": env_path(
            "SWIN_CKPT",
            STAGE1_MODEL_ROOT / "model_v2.1_swin_gtsrb.pth",
        ),
        "meta": env_path(
            "SWIN_META",
            STAGE1_MODEL_ROOT / "model_v2.1_swin_gtsrb_stage1_meta.json",
        ),
        "timm_name": "swinv2_small_window16_256.ms_in1k",
        "crop": 256,
        "resize": 280,
        "eval_batch": 32,
        "channels_last": False,
    },
    "eva02": {
        "ckpt": env_path(
            "EVA02_CKPT",
            STAGE1_MODEL_ROOT / "model_v3.1_eva02_stage1_best.pth",
        ),
        "meta": env_path(
            "EVA02_META",
            STAGE1_MODEL_ROOT / "model_v3.1_eva02_stage1_meta.json",
        ),
        "timm_name": "eva02_base_patch14_224.mim_in22k",
        "crop": 224,
        "resize": 256,
        "eval_batch": 24,
        "channels_last": False,
    },
    "effnet": {
        "ckpt": env_path(
            "EFFNET_CKPT",
            STAGE1_MODEL_ROOT / "model_v4.1_effnet_gtsrb.pth",
        ),
        "meta": env_path(
            "EFFNET_META",
            STAGE1_MODEL_ROOT / "model_v4.1_effnet_stage1_meta.json",
        ),
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "crop": 256,
        "resize": 288,
        "eval_batch": 48,
        "channels_last": True,
    },
    "caformer": {
        "ckpt": env_path(
            "CAFORMER_CKPT",
            STAGE1_MODEL_ROOT / "model_v5.1_caformer_stage1_best.pth",
        ),
        "meta": env_path(
            "CAFORMER_META",
            STAGE1_MODEL_ROOT / "model_v5.1_caformer_stage1_meta.txt",
        ),
        "timm_name": "caformer_s18.sail_in1k",
        "crop": 224,
        "resize": 256,
        "eval_batch": 32,
        "channels_last": False,
    },
    "maxvit": {
        "ckpt": env_path(
            "MAXVIT_CKPT",
            STAGE1_MODEL_ROOT / "model_v6.1_maxvit_stage1_best.pth",
        ),
        "meta": env_path(
            "MAXVIT_META",
            STAGE1_MODEL_ROOT / "model_v6.1_maxvit_stage1_meta.txt",
        ),
        "timm_name": "maxvit_tiny_rw_224.sw_in1k",
        "crop": 224,
        "resize": 256,
        "eval_batch": 16,
        "channels_last": False,
    },
    "coatnet": {
        "ckpt": env_path(
            "COATNET_CKPT",
            STAGE1_MODEL_ROOT / "coatnet_stage1_best.pth",
        ),
        "meta": env_path(
            "COATNET_META",
            STAGE1_MODEL_ROOT / "coatnet_stage1_meta.json",
        ),
        "timm_name": "coatnet_0_rw_224.sw_in1k",
        "crop": 224,
        "resize": 256,
        "eval_batch": 24,
        "channels_last": False,
    },
}

TEST_DIR = env_path("GTSRB_TEST_DIR", GTSRB_ROOT / "Test")
TEST_CSV = env_path("GTSRB_TEST_CSV", GTSRB_ROOT / "Test.csv")

# Optional per-model evaluation batch override.
# Example:
# BATCH_SIZE_OVERRIDES = {"maxvit": 8}
BATCH_SIZE_OVERRIDES = {}

NUM_WORKERS = int(os.getenv("EVAL_NUM_WORKERS", "4"))
USE_TEST_ROI = os.getenv("USE_TEST_ROI", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
KEEP_ASPECT_RATIO = os.getenv(
    "KEEP_ASPECT_RATIO",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

RUN_TSNE = os.getenv("RUN_TSNE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TSNE_MAX_SAMPLES = int(os.getenv("TSNE_MAX_SAMPLES", "3000"))

RUN_ROBUSTNESS = os.getenv(
    "RUN_ROBUSTNESS",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

SHOW_INDIVIDUAL_ROBUSTNESS = os.getenv(
    "SHOW_INDIVIDUAL_ROBUSTNESS",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

SHOW_WRONG_MAX = int(os.getenv("SHOW_WRONG_MAX", "36"))
SHOW_LOW_CONF_MAX = int(os.getenv("SHOW_LOW_CONF_MAX", "36"))

EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))

# Explicit and version-safe Gaussian noise strength.
# Pixel-domain sigma is approximately 5.1 .. 20.4 for uint8 images.
NOISE_STD_RANGE = (0.02, 0.08)

ROBUSTNESS_MODES = [
    "clean",
    "blur",
    "dark_mild",
    "dark_mid",
    "dark_heavy",
    "jpeg",
    "noise",
    "fog",
    "rain",
]

CANONICAL_CLASS_IDS = list(range(43))
CANONICAL_CLASS_NAMES = [str(index) for index in CANONICAL_CLASS_IDS]
NUM_CANONICAL_CLASSES = len(CANONICAL_CLASS_IDS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# 1. Configuration validation and reproducibility
# ================================================================


def normalize_selected_models(selected_models):
    if isinstance(selected_models, str):
        selected_models = [selected_models]

    selected_models = [
        str(item).strip().lower()
        for item in selected_models
        if str(item).strip()
    ]
    return list(dict.fromkeys(selected_models))


SELECTED_MODELS = normalize_selected_models(SELECTED_MODELS)

if not SELECTED_MODELS:
    raise ValueError("SELECTED_MODELS cannot be empty.")

unknown_models = [
    key
    for key in SELECTED_MODELS
    if key not in MODEL_REGISTRY
]
if unknown_models:
    raise ValueError(
        f"Unknown model keys: {unknown_models}; "
        f"choose from {list(MODEL_REGISTRY)}"
    )

if TSNE_MODEL_KEY is None:
    TSNE_MODEL_KEY = SELECTED_MODELS[0]

if TSNE_MODEL_KEY not in SELECTED_MODELS:
    raise ValueError(
        f"TSNE_MODEL_KEY={TSNE_MODEL_KEY!r} must be included "
        "in SELECTED_MODELS."
    )


def normalize_ensemble_weights(selected_models, configured_weights):
    if configured_weights is None:
        return {
            key: 1.0 / len(selected_models)
            for key in selected_models
        }

    if not isinstance(configured_weights, dict):
        raise TypeError(
            "ENSEMBLE_WEIGHTS must be None or a dict mapping model key "
            "to a positive numeric weight."
        )

    missing_keys = [
        key
        for key in selected_models
        if key not in configured_weights
    ]
    if missing_keys:
        raise ValueError(
            "ENSEMBLE_WEIGHTS is missing selected models: "
            f"{missing_keys}"
        )

    raw_weights = {
        key: float(configured_weights[key])
        for key in selected_models
    }

    if any(
        not np.isfinite(value) or value <= 0
        for value in raw_weights.values()
    ):
        raise ValueError(
            "Every selected-model ensemble weight must be finite and positive."
        )

    weight_sum = sum(raw_weights.values())
    return {
        key: value / weight_sum
        for key, value in raw_weights.items()
    }


normalized_weights = normalize_ensemble_weights(
    SELECTED_MODELS,
    ENSEMBLE_WEIGHTS,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(EVAL_SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def amp_autocast():
    if torch.cuda.is_available():
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    # CPU evaluation does not need autocast here.
    return torch.amp.autocast(
        device_type="cpu",
        enabled=False,
    )


display(Markdown("## Ensemble Configuration"))
display(
    pd.DataFrame(
        [
            {
                "model": key,
                "weight": normalized_weights[key],
                "checkpoint": MODEL_REGISTRY[key]["ckpt"],
            }
            for key in SELECTED_MODELS
        ]
    )
)


# ================================================================
# 2. Official GTSRB test set in canonical ClassId order
# ================================================================


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_gtsrb_test_csv(test_csv, test_dir):
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(
            f"Test CSV not found: {test_csv}\n"
            "Set GTSRB_TEST_CSV or GTSRB_ROOT."
        )

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test image directory not found: {test_dir}\n"
            "Set GTSRB_TEST_DIR or GTSRB_ROOT."
        )

    frame = pd.read_csv(
        test_csv,
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
            "CSV has no Path/Filename column. "
            f"Columns: {list(frame.columns)}"
        )

    if "ClassId" not in frame.columns:
        raise RuntimeError(
            "CSV has no ClassId column. "
            f"Columns: {list(frame.columns)}"
        )

    roi_columns = [
        "Roi.X1",
        "Roi.Y1",
        "Roi.X2",
        "Roi.Y2",
    ]
    has_roi = all(
        column in frame.columns
        for column in roi_columns
    )

    paths = []
    labels = []
    rois = []
    missing_files = 0
    bad_labels = 0

    for _, row in frame.iterrows():
        class_id = _as_int(row["ClassId"])

        if (
            class_id is None
            or class_id not in CANONICAL_CLASS_IDS
        ):
            bad_labels += 1
            continue

        file_path = os.path.join(
            test_dir,
            os.path.basename(str(row[path_column])),
        )

        if not os.path.isfile(file_path):
            missing_files += 1
            continue

        paths.append(file_path)
        labels.append(class_id)

        if has_roi:
            rois.append(
                (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
            )
        else:
            rois.append(None)

    if missing_files:
        print(f"Skipped missing image files: {missing_files}")

    if bad_labels:
        print(f"Skipped invalid labels: {bad_labels}")

    if not paths:
        raise RuntimeError(
            "No valid GTSRB test samples were loaded."
        )

    return (
        paths,
        np.asarray(labels, dtype=np.int64),
        rois,
    )


TEST_PATHS, TEST_LABELS, TEST_ROIS = load_gtsrb_test_csv(
    TEST_CSV,
    TEST_DIR,
)

display(Markdown("## Test Set Summary"))
display(
    pd.DataFrame(
        [
            ["Test samples", len(TEST_PATHS)],
            [
                "Covered classes",
                (
                    f"{len(set(TEST_LABELS.tolist()))}/"
                    f"{NUM_CANONICAL_CLASSES}"
                ),
            ],
        ],
        columns=["Item", "Value"],
    )
)


# ================================================================
# 3. Version-safe Albumentations helpers
# ================================================================


def supports_argument(callable_object, argument_name):
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False

    return argument_name in signature.parameters


def make_compose(transforms):
    kwargs = {}

    if supports_argument(A.Compose, "strict"):
        kwargs["strict"] = True

    if supports_argument(A.Compose, "seed"):
        kwargs["seed"] = EVAL_SEED

    return A.Compose(
        transforms,
        **kwargs,
    )


def make_image_compression(qmin, qmax):
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
        "Unsupported Albumentations ImageCompression API."
    )


def make_random_fog():
    if supports_argument(
        A.RandomFog,
        "fog_coef_range",
    ):
        return A.RandomFog(
            fog_coef_range=(0.15, 0.35),
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
            fog_coef_lower=0.15,
            fog_coef_upper=0.35,
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


def make_gauss_noise(
    std_range=NOISE_STD_RANGE,
):
    if supports_argument(
        A.GaussNoise,
        "std_range",
    ):
        kwargs = {
            "std_range": tuple(std_range),
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
        variance_range = (
            (float(std_range[0]) * 255.0) ** 2,
            (float(std_range[1]) * 255.0) ** 2,
        )

        kwargs = {
            "var_limit": variance_range,
            "p": 1.0,
        }

        if supports_argument(
            A.GaussNoise,
            "mean",
        ):
            kwargs["mean"] = 0

        return A.GaussNoise(**kwargs)

    raise RuntimeError(
        "Unsupported Albumentations GaussNoise API."
    )


def build_eval_aug(
    mode,
    mean,
    std,
    crop,
    resize,
):
    if KEEP_ASPECT_RATIO:
        resize_op = A.SmallestMaxSize(
            max_size=resize,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        resize_op = A.Resize(
            height=resize,
            width=resize,
            interpolation=cv2.INTER_CUBIC,
        )

    operations = [
        resize_op,
        A.CenterCrop(
            height=crop,
            width=crop,
        ),
    ]

    if mode == "blur":
        operations.append(
            A.MotionBlur(
                blur_limit=(3, 7),
                p=1.0,
            )
        )

    elif mode == "dark_mild":
        operations.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.20, -0.10),
                contrast_limit=(-0.10, 0.00),
                p=1.0,
            )
        )

    elif mode == "dark_mid":
        operations.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, -0.20),
                contrast_limit=(-0.15, 0.00),
                p=1.0,
            )
        )

    elif mode == "dark_heavy":
        operations.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.50, -0.30),
                contrast_limit=(-0.20, 0.00),
                p=1.0,
            )
        )

    elif mode == "jpeg":
        operations.append(
            make_image_compression(25, 45)
        )

    elif mode == "noise":
        operations.append(
            make_gauss_noise()
        )

    elif mode == "fog":
        operations.append(
            make_random_fog()
        )

    elif mode == "rain":
        operations.append(
            make_random_rain()
        )

    elif mode != "clean":
        raise ValueError(
            f"Unknown evaluation mode: {mode}"
        )

    operations.extend(
        [
            A.Normalize(
                mean=mean,
                std=std,
            ),
            ToTensorV2(),
        ]
    )

    # Albumentations 2.x may only warn on invalid arguments.
    # Turn that specific warning into an exception so the evaluation
    # cannot silently proceed with a malformed corruption transform.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            category=UserWarning,
            message=r".*Argument\(s\).*not valid.*",
        )

        return make_compose(operations)


MODE_SEED_OFFSET = {
    mode: index * 100_000
    for index, mode in enumerate(
        ROBUSTNESS_MODES
    )
}


def set_augmentation_seed(
    augmentation,
    seed,
):
    if hasattr(
        augmentation,
        "set_random_seed",
    ):
        augmentation.set_random_seed(int(seed))
        return

    if hasattr(
        augmentation,
        "set_deterministic",
    ):
        try:
            augmentation.set_deterministic(True)
        except Exception:
            pass


class SignDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        augmentation,
        mode,
        rois=None,
        use_roi=False,
    ):
        self.paths = list(paths)
        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )
        self.augmentation = augmentation
        self.mode = mode
        self.rois = rois
        self.use_roi = bool(
            use_roi
            and rois is not None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(
            self.paths[index],
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(
                "Failed to read image: "
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

            image = image[y1:y2, x1:x2]

        # Give the same sample and corruption mode the same RNG seed
        # across models, which makes ensemble robustness comparison
        # substantially more reproducible.
        sample_seed = (
            EVAL_SEED
            + MODE_SEED_OFFSET[self.mode]
            + int(index)
        )

        random.seed(sample_seed)
        np.random.seed(
            sample_seed % (2**32)
        )
        set_augmentation_seed(
            self.augmentation,
            sample_seed,
        )

        tensor = self.augmentation(
            image=image
        )["image"]

        return (
            tensor,
            int(self.labels[index]),
            self.paths[index],
        )


# ================================================================
# 4. Model loading and output-class alignment
# ================================================================


def read_meta(meta_path):
    """
    Read Stage1 metadata.

    Supports:
    - normal JSON files;
    - JSON text stored with a .txt extension;
    - simple Python-dict text as a compatibility fallback.
    """
    if (
        not meta_path
        or not os.path.isfile(meta_path)
    ):
        return {}

    text = Path(meta_path).read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return {}

    try:
        meta = json.loads(text)
    except json.JSONDecodeError:
        try:
            meta = ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            raise RuntimeError(
                f"Could not parse metadata file: {meta_path}. "
                "Expected JSON or a Python dict literal."
            ) from error

    if not isinstance(meta, dict):
        raise TypeError(
            f"Metadata must be a dict, got {type(meta)} "
            f"from {meta_path}."
        )

    return meta


def load_checkpoint(path):
    path = str(path)

    if path.lower().endswith(
        ".safetensors"
    ):
        if safe_load_file is None:
            raise ImportError(
                "safetensors is required to load "
                f"{path}. Install it with "
                "`pip install safetensors`."
            )

        return safe_load_file(path)

    # Prefer weights_only=True where supported.
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )
    except Exception:
        # Some older checkpoints contain objects that are not permitted
        # by weights_only=True. This fallback is intentionally local-file
        # oriented; only load checkpoints you trust.
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def unwrap_checkpoint(checkpoint):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dict, "
            f"got {type(checkpoint)}"
        )

    # Common checkpoint-container conventions.
    for key in (
        "state_dict_ema",
        "model_ema",
        "model_state",
        "model_state_dict",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(
            value,
            dict,
        ):
            return value

    # Stage1 save_backbone() writes a raw state_dict,
    # so returning the checkpoint itself is expected.
    return checkpoint


def strip_prefix_if_common(
    state_dict,
    prefix,
    threshold=0.8,
):
    keys = list(
        state_dict.keys()
    )

    if not keys:
        return state_dict

    ratio = (
        sum(
            key.startswith(prefix)
            for key in keys
        )
        / len(keys)
    )

    if ratio < threshold:
        return state_dict

    return {
        (
            key[len(prefix):]
            if key.startswith(prefix)
            else key
        ): value
        for key, value
        in state_dict.items()
    }


def infer_classid_to_model_index(
    meta,
    num_classes,
):
    raw_mapping = (
        meta.get(
            "classid_to_idx",
            {},
        )
        or {}
    )

    mapping = {
        int(class_id): int(model_index)
        for class_id, model_index
        in raw_mapping.items()
    }

    if not mapping:
        class_names = (
            meta.get(
                "class_names",
                [],
            )
            or []
        )

        if class_names:
            candidate_mapping = {}

            for model_index, class_name in enumerate(
                class_names
            ):
                try:
                    candidate_mapping[
                        int(class_name)
                    ] = model_index
                except (
                    TypeError,
                    ValueError,
                ):
                    candidate_mapping = {}
                    break

            mapping = candidate_mapping

    if (
        not mapping
        and num_classes
        == NUM_CANONICAL_CLASSES
    ):
        mapping = {
            class_id: class_id
            for class_id
            in CANONICAL_CLASS_IDS
        }

    missing_class_ids = [
        class_id
        for class_id
        in CANONICAL_CLASS_IDS
        if class_id not in mapping
    ]

    if missing_class_ids:
        raise RuntimeError(
            "Cannot align model outputs to "
            "GTSRB ClassId order. Missing "
            f"ClassIds: {missing_class_ids}."
        )

    reorder = np.asarray(
        [
            mapping[class_id]
            for class_id
            in CANONICAL_CLASS_IDS
        ],
        dtype=np.int64,
    )

    if (
        reorder.min() < 0
        or reorder.max() >= num_classes
    ):
        raise RuntimeError(
            "Invalid output reorder indices: "
            f"min={reorder.min()}, "
            f"max={reorder.max()}, "
            f"num_classes={num_classes}"
        )

    if (
        len(set(reorder.tolist()))
        != NUM_CANONICAL_CLASSES
    ):
        raise RuntimeError(
            "ClassId-to-output mapping "
            "contains duplicate indices."
        )

    return reorder


def create_and_load_model(
    model_key,
):
    specification = MODEL_REGISTRY[
        model_key
    ]

    checkpoint_path = specification[
        "ckpt"
    ]
    meta_path = specification.get(
        "meta",
        "",
    )

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "Checkpoint not found for "
            f"{model_key}:\n"
            f"{checkpoint_path}\n"
            "Set the corresponding *_CKPT "
            "environment variable or place "
            "the file under weights/stage1/."
        )

    meta = read_meta(meta_path)

    if (
        meta_path
        and not os.path.isfile(meta_path)
    ):
        print(
            f"Warning: metadata file not found "
            f"for {model_key}: {meta_path}. "
            "Falling back to registry defaults "
            "and canonical class order when possible."
        )

    model_name = meta.get(
        "timm_name",
        specification["timm_name"],
    )

    class_names = (
        meta.get(
            "class_names",
            [],
        )
        or []
    )

    num_classes = (
        len(class_names)
        if class_names
        else NUM_CANONICAL_CLASSES
    )

    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to create timm model "
            f"{model_name!r} for {model_key}.\n"
            f"Original error: {error}"
        ) from error

    data_config = resolve_data_config(
        {},
        model=model,
    )
    mean = list(
        data_config["mean"]
    )
    std = list(
        data_config["std"]
    )

    checkpoint = unwrap_checkpoint(
        load_checkpoint(
            checkpoint_path
        )
    )

    for prefix in (
        "module.",
        "_orig_mod.",
        "model.",
    ):
        checkpoint = (
            strip_prefix_if_common(
                checkpoint,
                prefix,
            )
        )

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for key, value in checkpoint.items():
        if (
            key in model_state
            and torch.is_tensor(value)
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
        value.numel()
        for value in loadable.values()
    )
    total_numel = sum(
        value.numel()
        for value in model_state.values()
    )
    coverage = (
        loaded_numel
        / max(total_numel, 1)
    )

    if (
        len(loadable) < 50
        or coverage < 0.80
    ):
        raise RuntimeError(
            "Checkpoint coverage is too "
            f"low for {model_key}: "
            f"{len(loadable)} keys, "
            f"{coverage * 100:.2f}% "
            "of parameter elements."
        )

    reorder = (
        infer_classid_to_model_index(
            meta,
            num_classes,
        )
    )

    crop = int(
        meta.get(
            "crop",
            specification["crop"],
        )
    )
    resize = int(
        meta.get(
            "resize",
            specification["resize"],
        )
    )
    batch_size = int(
        BATCH_SIZE_OVERRIDES.get(
            model_key,
            specification["eval_batch"],
        )
    )
    channels_last = bool(
        specification.get(
            "channels_last",
            False,
        )
    )

    if batch_size <= 0:
        raise ValueError(
            f"Evaluation batch size for "
            f"{model_key} must be positive."
        )

    model = model.to(
        DEVICE
    ).eval()

    if channels_last:
        model = model.to(
            memory_format=torch.channels_last
        )

    return {
        "key": model_key,
        "model": model,
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "meta": meta_path,
        "num_classes": num_classes,
        "crop": crop,
        "resize": resize,
        "batch_size": batch_size,
        "channels_last": channels_last,
        "mean": mean,
        "std": std,
        "reorder": reorder,
        "loaded_keys": len(loadable),
        "skipped_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "coverage": coverage,
    }


def make_loader(
    bundle,
    mode,
):
    augmentation = build_eval_aug(
        mode=mode,
        mean=bundle["mean"],
        std=bundle["std"],
        crop=bundle["crop"],
        resize=bundle["resize"],
    )

    dataset = SignDataset(
        TEST_PATHS,
        TEST_LABELS,
        augmentation,
        mode=mode,
        rois=TEST_ROIS,
        use_roi=USE_TEST_ROI,
    )

    generator = torch.Generator()
    generator.manual_seed(
        EVAL_SEED
    )

    return DataLoader(
        dataset,
        batch_size=bundle[
            "batch_size"
        ],
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ================================================================
# 5. Probability prediction and ensemble
# ================================================================


@torch.inference_mode()
def predict_probabilities(
    bundle,
    loader,
):
    model = bundle["model"]
    model.eval()

    probability_batches = []
    target_batches = []
    paths = []

    reorder_tensor = torch.as_tensor(
        bundle["reorder"],
        dtype=torch.long,
        device=DEVICE,
    )

    for (
        images,
        labels,
        batch_paths,
    ) in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        if bundle[
            "channels_last"
        ]:
            images = images.contiguous(
                memory_format=(
                    torch.channels_last
                )
            )

        with amp_autocast():
            logits = model(images)

        # Softmax in float32 for numerically stable
        # probability averaging.
        probabilities = F.softmax(
            logits.float(),
            dim=-1,
        )

        if (
            probabilities.shape[1]
            != bundle["num_classes"]
        ):
            raise RuntimeError(
                f"{bundle['key']} returned "
                f"{probabilities.shape[1]} "
                "classes, expected "
                f"{bundle['num_classes']}."
            )

        # After this index_select:
        # output column j corresponds to canonical ClassId j.
        probabilities = (
            probabilities.index_select(
                1,
                reorder_tensor,
            )
        )

        probability_batches.append(
            probabilities.cpu().numpy()
        )
        target_batches.append(
            labels.numpy()
        )
        paths.extend(
            list(batch_paths)
        )

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )
    targets = np.concatenate(
        target_batches,
        axis=0,
    )

    return (
        probabilities,
        targets,
        paths,
    )


def combine_probabilities(
    probability_dict,
    weights,
):
    if not probability_dict:
        raise ValueError(
            "probability_dict cannot be empty."
        )

    first_key = next(
        iter(probability_dict)
    )
    reference_shape = (
        probability_dict[
            first_key
        ].shape
    )

    combined = np.zeros(
        reference_shape,
        dtype=np.float64,
    )

    for (
        model_key,
        probabilities,
    ) in probability_dict.items():
        if (
            probabilities.shape
            != reference_shape
        ):
            raise RuntimeError(
                "Probability shape mismatch "
                f"for {model_key}: "
                f"{probabilities.shape} vs "
                f"{reference_shape}"
            )

        combined += (
            float(weights[model_key])
            * probabilities.astype(
                np.float64
            )
        )

    row_sums = combined.sum(
        axis=1,
        keepdims=True,
    )
    combined = (
        combined
        / np.clip(
            row_sums,
            1e-12,
            None,
        )
    )

    return combined.astype(
        np.float32
    )


def metrics_from_probabilities(
    targets,
    probabilities,
):
    predictions = (
        probabilities.argmax(
            axis=1
        )
    )
    confidences = (
        probabilities.max(
            axis=1
        )
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=CANONICAL_CLASS_IDS,
        average="macro",
        zero_division=0,
    )

    return {
        "acc": float(accuracy),
        "f1": float(macro_f1),
        "preds": predictions,
        "confs": confidences,
    }


# ================================================================
# 6. Optional t-SNE helpers
# ================================================================


def generic_pool_features(
    features,
):
    if isinstance(
        features,
        (tuple, list),
    ):
        features = features[-1]

    if not torch.is_tensor(
        features
    ):
        raise TypeError(
            "Unsupported forward_features "
            f"output: {type(features)}"
        )

    if features.ndim == 4:
        # Heuristic:
        # NCHW: [B, C, H, W] normally C > W.
        # NHWC: [B, H, W, C] normally W < C.
        if (
            features.shape[1]
            > features.shape[-1]
        ):
            features = (
                F.adaptive_avg_pool2d(
                    features,
                    1,
                )
                .flatten(1)
            )
        else:
            features = features.mean(
                dim=(1, 2)
            )

    elif features.ndim == 3:
        features = features.mean(
            dim=1
        )

    elif features.ndim > 2:
        features = features.flatten(
            1
        )

    return features


def backbone_vector(
    model,
    images,
):
    features = model.forward_features(
        images
    )

    if hasattr(
        model,
        "forward_head",
    ):
        try:
            vector = model.forward_head(
                features,
                pre_logits=True,
            )

            if (
                torch.is_tensor(
                    vector
                )
                and vector.ndim == 2
            ):
                return vector

        except (
            TypeError,
            RuntimeError,
            AttributeError,
        ):
            pass

    return generic_pool_features(
        features
    )


@torch.inference_mode()
def extract_features(
    bundle,
    loader,
    max_samples=3000,
):
    model = bundle["model"]
    model.eval()

    feature_batches = []
    label_batches = []
    seen = 0

    for (
        images,
        labels,
        _,
    ) in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        if bundle[
            "channels_last"
        ]:
            images = images.contiguous(
                memory_format=(
                    torch.channels_last
                )
            )

        with amp_autocast():
            features = backbone_vector(
                model,
                images,
            )

        features = F.normalize(
            features.float(),
            dim=1,
        )

        feature_batches.append(
            features.cpu().numpy()
        )
        label_batches.append(
            labels.numpy()
        )

        seen += images.size(0)

        if seen >= max_samples:
            break

    features = np.concatenate(
        feature_batches,
        axis=0,
    )[:max_samples]
    labels = np.concatenate(
        label_batches,
        axis=0,
    )[:max_samples]

    return (
        features,
        labels,
    )


# ================================================================
# 7. Evaluation pipeline
# ================================================================


def evaluate_selected_models():
    probabilities_by_mode = {
        mode: {}
        for mode in ROBUSTNESS_MODES
    }

    shared_targets = None
    shared_paths = None
    loading_rows = []
    tsne_features = None
    tsne_labels = None

    active_modes = (
        ROBUSTNESS_MODES
        if RUN_ROBUSTNESS
        else ["clean"]
    )

    for (
        model_index,
        model_key,
    ) in enumerate(
        SELECTED_MODELS,
        start=1,
    ):
        print("=" * 80)
        print(
            f"[{model_index}/"
            f"{len(SELECTED_MODELS)}] "
            f"Loading {model_key} ..."
        )

        bundle = create_and_load_model(
            model_key
        )

        loading_rows.append(
            {
                "model": model_key,
                "timm_name": bundle[
                    "model_name"
                ],
                "crop": bundle[
                    "crop"
                ],
                "resize": bundle[
                    "resize"
                ],
                "batch": bundle[
                    "batch_size"
                ],
                "coverage_percent": (
                    bundle["coverage"]
                    * 100
                ),
                "loaded_keys": bundle[
                    "loaded_keys"
                ],
                "missing_keys": bundle[
                    "missing_keys"
                ],
                "skipped_keys": bundle[
                    "skipped_keys"
                ],
                "unexpected_keys": bundle[
                    "unexpected_keys"
                ],
            }
        )

        for mode in active_modes:
            print(
                f"  Predicting "
                f"{model_key} / {mode} ..."
            )

            loader = make_loader(
                bundle,
                mode,
            )

            (
                probabilities,
                targets,
                paths,
            ) = predict_probabilities(
                bundle,
                loader,
            )

            if shared_targets is None:
                shared_targets = targets
                shared_paths = paths
            else:
                if not np.array_equal(
                    shared_targets,
                    targets,
                ):
                    raise RuntimeError(
                        "Target order mismatch for "
                        f"model={model_key}, "
                        f"mode={mode}."
                    )

                if shared_paths != paths:
                    raise RuntimeError(
                        "Image order mismatch for "
                        f"model={model_key}, "
                        f"mode={mode}."
                    )

            probabilities_by_mode[
                mode
            ][model_key] = probabilities

            del loader
            gc.collect()

        if (
            RUN_TSNE
            and model_key
            == TSNE_MODEL_KEY
        ):
            print(
                "  Extracting t-SNE "
                f"features from {model_key} ..."
            )

            tsne_loader = make_loader(
                bundle,
                "clean",
            )

            (
                tsne_features,
                tsne_labels,
            ) = extract_features(
                bundle,
                tsne_loader,
                max_samples=(
                    TSNE_MAX_SAMPLES
                ),
            )

            del tsne_loader

        del bundle["model"]
        del bundle

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "probabilities_by_mode": (
            probabilities_by_mode
        ),
        "targets": shared_targets,
        "paths": shared_paths,
        "loading_rows": loading_rows,
        "tsne_features": tsne_features,
        "tsne_labels": tsne_labels,
    }


def plot_sample_grid(
    eval_res,
    indices,
    title_prefix,
    predictor_name,
    max_n=36,
    columns=6,
):
    indices = list(
        indices[:max_n]
    )

    if not indices:
        display(
            Markdown(
                f"### {title_prefix}: "
                "no samples"
            )
        )
        return

    rows = math.ceil(
        len(indices)
        / columns
    )

    plt.figure(
        figsize=(
            columns * 3.0,
            rows * 3.4,
        )
    )

    for (
        plot_index,
        sample_index,
    ) in enumerate(indices):
        image = cv2.imread(
            eval_res["paths"][
                sample_index
            ]
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        true_label = int(
            eval_res["targets"][
                sample_index
            ]
        )
        predicted_label = int(
            eval_res["preds"][
                sample_index
            ]
        )
        confidence = float(
            eval_res["confs"][
                sample_index
            ]
        )

        axis = plt.subplot(
            rows,
            columns,
            plot_index + 1,
        )
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"T: {true_label}\n"
            f"P: {predicted_label}\n"
            f"C: {confidence:.3f}",
            fontsize=9,
        )

    plt.suptitle(
        f"{title_prefix} — "
        f"{predictor_name}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def plot_tsne(
    features,
    labels,
    model_key,
):
    if (
        features is None
        or len(features) < 20
    ):
        display(
            Markdown(
                "t-SNE skipped: too few "
                "samples or no features."
            )
        )
        return

    perplexity = min(
        30,
        max(
            5,
            (len(features) - 1) // 3,
        ),
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=EVAL_SEED,
    )

    embedding = tsne.fit_transform(
        features
    )

    plt.figure(
        figsize=(11, 9)
    )
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        s=8,
        alpha=0.8,
    )
    plt.title(
        "t-SNE of Backbone Features "
        f"— {model_key}"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(
        scatter,
        label="GTSRB ClassId",
    )
    plt.tight_layout()
    plt.show()


def display_clean_results(
    evaluation,
):
    probabilities_by_mode = evaluation[
        "probabilities_by_mode"
    ]
    shared_targets = evaluation[
        "targets"
    ]
    shared_paths = evaluation[
        "paths"
    ]

    clean_probability_dict = (
        probabilities_by_mode[
            "clean"
        ]
    )

    combined_clean_probabilities = (
        combine_probabilities(
            clean_probability_dict,
            normalized_weights,
        )
    )

    combined_name = (
        SELECTED_MODELS[0]
        if len(SELECTED_MODELS) == 1
        else "ensemble"
    )

    clean_result_rows = []

    for model_key in SELECTED_MODELS:
        result = (
            metrics_from_probabilities(
                shared_targets,
                clean_probability_dict[
                    model_key
                ],
            )
        )

        clean_result_rows.append(
            {
                "predictor": model_key,
                "weight": (
                    normalized_weights[
                        model_key
                    ]
                ),
                "accuracy_percent": (
                    result["acc"] * 100
                ),
                "macro_f1_percent": (
                    result["f1"] * 100
                ),
            }
        )

    combined_clean_result = (
        metrics_from_probabilities(
            shared_targets,
            combined_clean_probabilities,
        )
    )

    if len(SELECTED_MODELS) > 1:
        clean_result_rows.append(
            {
                "predictor": "ensemble",
                "weight": 1.0,
                "accuracy_percent": (
                    combined_clean_result[
                        "acc"
                    ]
                    * 100
                ),
                "macro_f1_percent": (
                    combined_clean_result[
                        "f1"
                    ]
                    * 100
                ),
            }
        )

    display(
        Markdown(
            "## Clean Test Results"
        )
    )
    display(
        pd.DataFrame(
            clean_result_rows
        )
    )

    eval_res = {
        "acc": combined_clean_result[
            "acc"
        ],
        "f1": combined_clean_result[
            "f1"
        ],
        "preds": combined_clean_result[
            "preds"
        ],
        "targets": shared_targets,
        "confs": combined_clean_result[
            "confs"
        ],
        "paths": shared_paths,
    }

    return (
        eval_res,
        combined_name,
    )


def display_per_class_report(
    eval_res,
    combined_name,
):
    report_dict = (
        classification_report(
            eval_res["targets"],
            eval_res["preds"],
            labels=CANONICAL_CLASS_IDS,
            target_names=(
                CANONICAL_CLASS_NAMES
            ),
            digits=4,
            zero_division=0,
            output_dict=True,
        )
    )

    report_df = pd.DataFrame(
        report_dict
    ).T

    display(
        Markdown(
            "## Per-class Report — "
            f"{combined_name}"
        )
    )
    display(report_df)

    per_class_df = (
        report_df.iloc[
            :NUM_CANONICAL_CLASSES
        ].copy()
    )
    per_class_df["class"] = (
        CANONICAL_CLASS_NAMES
    )
    per_class_df[
        "f1-score"
    ] = per_class_df[
        "f1-score"
    ].astype(float)

    plt.figure(
        figsize=(16, 5)
    )
    plt.bar(
        per_class_df["class"],
        per_class_df["f1-score"]
        * 100,
    )
    plt.xticks(rotation=90)
    plt.ylim(0, 100.5)
    plt.ylabel("F1 Score (%)")
    plt.xlabel("GTSRB ClassId")
    plt.title(
        "Per-class F1 Score — "
        f"{combined_name}"
    )
    plt.tight_layout()
    plt.show()


def display_confusion_analysis(
    eval_res,
    combined_name,
):
    confusion = confusion_matrix(
        eval_res["targets"],
        eval_res["preds"],
        labels=CANONICAL_CLASS_IDS,
    )

    display(
        Markdown(
            "## Confusion Matrix — "
            f"{combined_name}"
        )
    )

    figure, axis = plt.subplots(
        figsize=(16, 16)
    )
    _ = figure

    display_object = (
        ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=(
                CANONICAL_CLASS_NAMES
            ),
        )
    )

    display_object.plot(
        ax=axis,
        xticks_rotation=90,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "Confusion Matrix — "
        f"{combined_name}"
    )
    axis.set_xlabel(
        "Predicted Label"
    )
    axis.set_ylabel(
        "True Label"
    )

    plt.tight_layout()
    plt.show()

    confusion_pairs = []

    for true_class in (
        CANONICAL_CLASS_IDS
    ):
        for predicted_class in (
            CANONICAL_CLASS_IDS
        ):
            count = int(
                confusion[
                    true_class,
                    predicted_class,
                ]
            )

            if (
                true_class
                != predicted_class
                and count > 0
            ):
                confusion_pairs.append(
                    {
                        "true": str(
                            true_class
                        ),
                        "pred": str(
                            predicted_class
                        ),
                        "count": count,
                    }
                )

    display(
        Markdown(
            "## Top Confusion Pairs"
        )
    )

    if not confusion_pairs:
        display(
            Markdown(
                "No misclassification "
                "pairs on the clean "
                "test set."
            )
        )
        return

    confusion_pair_df = (
        pd.DataFrame(
            confusion_pairs
        )
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    display(
        confusion_pair_df.head(30)
    )

    top_show = (
        confusion_pair_df
        .head(20)
        .copy()
    )

    pair_labels = [
        f"{row.true} → {row.pred}"
        for _, row
        in top_show.iterrows()
    ]

    plt.figure(
        figsize=(
            10,
            max(
                5,
                len(top_show) * 0.35,
            ),
        )
    )
    plt.barh(
        pair_labels[::-1],
        top_show[
            "count"
        ].values[::-1],
    )
    plt.xlabel("Count")
    plt.title(
        "Top Confusion Pairs — "
        f"{combined_name}"
    )
    plt.tight_layout()
    plt.show()


def display_sample_analysis(
    eval_res,
    combined_name,
):
    wrong_indices = np.where(
        eval_res["targets"]
        != eval_res["preds"]
    )[0]

    if len(wrong_indices):
        wrong_indices = wrong_indices[
            np.argsort(
                -eval_res["confs"][
                    wrong_indices
                ]
            )
        ]

    low_confidence_indices = (
        np.argsort(
            eval_res["confs"]
        )
    )

    display(
        Markdown(
            "## Wrong Prediction Samples"
        )
    )
    plot_sample_grid(
        eval_res,
        wrong_indices,
        "Wrong Prediction Samples",
        combined_name,
        max_n=SHOW_WRONG_MAX,
        columns=6,
    )

    display(
        Markdown(
            "## Low Confidence Samples"
        )
    )
    plot_sample_grid(
        eval_res,
        low_confidence_indices,
        "Low Confidence Samples",
        combined_name,
        max_n=SHOW_LOW_CONF_MAX,
        columns=6,
    )


def display_robustness_results(
    evaluation,
    combined_name,
):
    if not RUN_ROBUSTNESS:
        return

    display(
        Markdown(
            "## Robustness Evaluation"
        )
    )

    probabilities_by_mode = evaluation[
        "probabilities_by_mode"
    ]
    shared_targets = evaluation[
        "targets"
    ]

    robustness_rows = []
    combined_rows = []

    for mode in ROBUSTNESS_MODES:
        probability_dict = (
            probabilities_by_mode[
                mode
            ]
        )

        for model_key in (
            SELECTED_MODELS
        ):
            result = (
                metrics_from_probabilities(
                    shared_targets,
                    probability_dict[
                        model_key
                    ],
                )
            )

            robustness_rows.append(
                {
                    "predictor": model_key,
                    "mode": mode,
                    "accuracy_percent": (
                        result["acc"] * 100
                    ),
                    "macro_f1_percent": (
                        result["f1"] * 100
                    ),
                }
            )

        combined_probabilities = (
            combine_probabilities(
                probability_dict,
                normalized_weights,
            )
        )

        combined_result = (
            metrics_from_probabilities(
                shared_targets,
                combined_probabilities,
            )
        )

        combined_row = {
            "predictor": combined_name,
            "mode": mode,
            "accuracy_percent": (
                combined_result["acc"]
                * 100
            ),
            "macro_f1_percent": (
                combined_result["f1"]
                * 100
            ),
        }

        combined_rows.append(
            combined_row
        )

        if len(SELECTED_MODELS) > 1:
            robustness_rows.append(
                combined_row.copy()
            )

    robustness_df = pd.DataFrame(
        robustness_rows
    )
    combined_robustness_df = (
        pd.DataFrame(
            combined_rows
        )
    )

    if SHOW_INDIVIDUAL_ROBUSTNESS:
        display(
            Markdown(
                "### Individual Models "
                "and Ensemble"
            )
        )
        display(robustness_df)

        macro_f1_pivot = (
            robustness_df.pivot(
                index="mode",
                columns="predictor",
                values=(
                    "macro_f1_percent"
                ),
            )
            .reindex(
                ROBUSTNESS_MODES
            )
        )

        display(
            Markdown(
                "### Macro-F1 Comparison"
            )
        )
        display(macro_f1_pivot)

        plt.figure(
            figsize=(12, 6)
        )

        for predictor in (
            macro_f1_pivot.columns
        ):
            plt.plot(
                macro_f1_pivot.index,
                macro_f1_pivot[
                    predictor
                ].values,
                marker="o",
                label=predictor,
            )

        plt.ylim(0, 100.5)
        plt.ylabel("Macro-F1 (%)")
        plt.xlabel("Corruption Mode")
        plt.title(
            "Robustness Macro-F1 "
            "Comparison"
        )
        plt.xticks(rotation=25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    display(
        Markdown(
            "### Combined Predictor — "
            f"{combined_name}"
        )
    )
    display(
        combined_robustness_df
    )

    mode_labels = (
        combined_robustness_df[
            "mode"
        ].tolist()
    )
    accuracies = (
        combined_robustness_df[
            "accuracy_percent"
        ].values
    )
    macro_f1_values = (
        combined_robustness_df[
            "macro_f1_percent"
        ].values
    )

    x_positions = np.arange(
        len(mode_labels)
    )
    bar_width = 0.35

    plt.figure(
        figsize=(11, 5.5)
    )
    plt.bar(
        x_positions
        - bar_width / 2,
        accuracies,
        bar_width,
        label="Accuracy",
    )
    plt.bar(
        x_positions
        + bar_width / 2,
        macro_f1_values,
        bar_width,
        label="Macro-F1",
    )
    plt.ylim(0, 100.5)
    plt.xticks(
        x_positions,
        mode_labels,
        rotation=20,
    )
    plt.ylabel("Score (%)")
    plt.title(
        "Robustness Evaluation — "
        f"{combined_name}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    clean_accuracy = (
        combined_robustness_df.loc[
            combined_robustness_df[
                "mode"
            ] == "clean",
            "accuracy_percent",
        ].iloc[0]
    )

    clean_macro_f1 = (
        combined_robustness_df.loc[
            combined_robustness_df[
                "mode"
            ] == "clean",
            "macro_f1_percent",
        ].iloc[0]
    )

    combined_robustness_df = (
        combined_robustness_df.copy()
    )
    combined_robustness_df[
        "accuracy_drop"
    ] = (
        clean_accuracy
        - combined_robustness_df[
            "accuracy_percent"
        ]
    )
    combined_robustness_df[
        "macro_f1_drop"
    ] = (
        clean_macro_f1
        - combined_robustness_df[
            "macro_f1_percent"
        ]
    )

    display(
        Markdown(
            "### Performance Drop "
            "Compared with Clean"
        )
    )
    display(
        combined_robustness_df[
            [
                "mode",
                "accuracy_drop",
                "macro_f1_drop",
            ]
        ]
    )

    plt.figure(
        figsize=(11, 5.5)
    )
    plt.bar(
        x_positions
        - bar_width / 2,
        combined_robustness_df[
            "accuracy_drop"
        ].values,
        bar_width,
        label="Accuracy Drop",
    )
    plt.bar(
        x_positions
        + bar_width / 2,
        combined_robustness_df[
            "macro_f1_drop"
        ].values,
        bar_width,
        label="Macro-F1 Drop",
    )
    plt.xticks(
        x_positions,
        mode_labels,
        rotation=20,
    )
    plt.ylabel(
        "Drop (percentage points)"
    )
    plt.title(
        "Robustness Drop Compared "
        "with Clean — "
        f"{combined_name}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    evaluation = (
        evaluate_selected_models()
    )

    display(
        Markdown(
            "## Checkpoint Loading Summary"
        )
    )
    display(
        pd.DataFrame(
            evaluation[
                "loading_rows"
            ]
        )
    )

    (
        eval_res,
        combined_name,
    ) = display_clean_results(
        evaluation
    )

    display_per_class_report(
        eval_res,
        combined_name,
    )

    display_confusion_analysis(
        eval_res,
        combined_name,
    )

    display_sample_analysis(
        eval_res,
        combined_name,
    )

    if RUN_TSNE:
        display(
            Markdown(
                "## t-SNE Feature "
                "Visualization — "
                f"{TSNE_MODEL_KEY}"
            )
        )

        plot_tsne(
            evaluation[
                "tsne_features"
            ],
            evaluation[
                "tsne_labels"
            ],
            TSNE_MODEL_KEY,
        )

    display_robustness_results(
        evaluation,
        combined_name,
    )

    display(
        Markdown(
            "## Done — "
            f"selected={SELECTED_MODELS}, "
            f"weights={normalized_weights}"
        )
    )


if __name__ == "__main__":
    main()
