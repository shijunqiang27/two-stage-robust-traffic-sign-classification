# Stage 1 Backbone Evaluation
# 功能：
#   1. 加载已经训练好的 Stage1 best/last 模型
#   2. 在 GTSRB 官方 test 上评估 Acc / Macro-F1 / per-class F1
#   3. 展示混淆矩阵、错分样本、低置信度样本
#   4. 展示 t-SNE backbone 特征
#   5. 展示 clean / blur / dark / jpeg / noise / fog / rain 抗退化测试
# ================================================================

import os
import json
import math
import warnings
import re
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"evaluation", "eval", "scripts"}
    else SCRIPT_DIR
)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()

# ================================================================
#  0. Configuration
# ================================================================
GTSRB_ROOT = Path(
    os.getenv("GTSRB_ROOT", str(PROJECT_ROOT / "data" / "GTSRB"))
).expanduser()

STAGE1_CKPT = os.getenv(
    "STAGE1_CKPT",
    str(PROJECT_ROOT / "weights" / "stage1" / "convnext_stage1_best.pth"),
)
META_JSON = os.getenv(
    "STAGE1_META_JSON",
    str(PROJECT_ROOT / "weights" / "stage1" / "convnext_stage1_meta.json"),
)
TEST_DIR = os.getenv("GTSRB_TEST_DIR", str(GTSRB_ROOT / "Test"))
TEST_CSV = os.getenv("GTSRB_TEST_CSV", str(GTSRB_ROOT / "Test.csv"))

BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "64"))
NUM_WORKERS = int(os.getenv("EVAL_NUM_WORKERS", "4"))
USE_TEST_ROI = os.getenv("USE_TEST_ROI", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

RUN_TSNE = os.getenv("RUN_TSNE", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
TSNE_MAX_SAMPLES = int(os.getenv("TSNE_MAX_SAMPLES", "3000"))

RUN_ROBUSTNESS = os.getenv("RUN_ROBUSTNESS", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

SHOW_WRONG_MAX = int(os.getenv("SHOW_WRONG_MAX", "36"))
SHOW_LOW_CONF_MAX = int(os.getenv("SHOW_LOW_CONF_MAX", "36"))

# ================================================================
#  1. Dependencies
# ================================================================
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from IPython.display import display, Markdown

import timm
from timm.data import resolve_data_config

import albumentations as A
from albumentations.pytorch import ToTensorV2

from safetensors.torch import load_file as safe_load_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def amp_autocast():
    return torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available())

# ================================================================
#  2. 读取 meta，创建模型
# ================================================================
assert os.path.isfile(STAGE1_CKPT), f"找不到模型权重: {STAGE1_CKPT}"
assert os.path.isfile(META_JSON), f"找不到 meta 文件: {META_JSON}"

with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

MODEL_NAME = meta.get("timm_name", "convnextv2_base.fcmae_ft_in22k_in1k")
CLASS_NAMES = meta.get("class_names", [str(i) for i in range(43)])
NUM_CLASSES = len(CLASS_NAMES)
CROP = int(meta.get("crop", 256))
RESIZE = int(meta.get("resize", 288))

classid_to_idx = meta.get("classid_to_idx", {})
classid_to_idx = {int(k): int(v) for k, v in classid_to_idx.items()} if classid_to_idx else {}

if not classid_to_idx:
    ok_numeric = True
    tmp_map = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            tmp_map[int(name)] = i
        except Exception:
            ok_numeric = False
            break
    if ok_numeric:
        classid_to_idx = tmp_map

display(Markdown("## Model Info"))

info_df = pd.DataFrame([
    ["Model name", MODEL_NAME],
    ["Num classes", NUM_CLASSES],
    ["Resize", RESIZE],
    ["Crop", CROP],
    ["Checkpoint", STAGE1_CKPT],
    ["Meta JSON", META_JSON],
    ["Device", DEVICE],
], columns=["Item", "Value"])

display(info_df)

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
)

dcfg = resolve_data_config({}, model=model)
MEAN = list(dcfg["mean"])
STD = list(dcfg["std"])

def load_checkpoint(path):
    if path.endswith(".safetensors"):
        ckpt = safe_load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    return ckpt

ckpt = load_checkpoint(STAGE1_CKPT)

cleaned = {}
for k, v in ckpt.items():
    k = k.replace("module.", "")
    cleaned[k] = v

model_state = model.state_dict()
loadable = {}
skipped = []

for k, v in cleaned.items():
    if k in model_state and model_state[k].shape == v.shape:
        loadable[k] = v
    else:
        skipped.append(k)

missing, unexpected = model.load_state_dict(loadable, strict=False)

load_df = pd.DataFrame([
    ["Loaded keys", len(loadable)],
    ["Skipped mismatched keys", len(skipped)],
    ["Missing keys", len(missing)],
    ["Unexpected keys", len(unexpected)],
], columns=["Item", "Count"])

display(Markdown("## Checkpoint Loading Summary"))
display(load_df)

if len(loadable) < 100:
    raise RuntimeError(
        f"加载的 key 数量异常少：{len(loadable)}。"
        f"请检查 STAGE1_CKPT 是否为 Stage1 训练出的 best/last，"
        f"以及 MODEL_NAME、NUM_CLASSES 是否匹配。"
    )

model = model.to(DEVICE).eval()

CHANNELS_LAST = any(s in MODEL_NAME.lower() for s in ["convnext", "efficientnet", "effnet"])
if CHANNELS_LAST:
    model = model.to(memory_format=torch.channels_last)

# ================================================================
#  3. 读取 GTSRB test
# ================================================================
def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None

def load_gtsrb_test_csv(test_csv, test_dir, classid_to_idx):
    df = pd.read_csv(test_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    path_col = None
    if "Path" in df.columns:
        path_col = "Path"
    elif "Filename" in df.columns:
        path_col = "Filename"

    assert path_col is not None, f"CSV 中找不到 Path/Filename 列，实际列为: {list(df.columns)}"
    assert "ClassId" in df.columns, f"CSV 中找不到 ClassId 列，实际列为: {list(df.columns)}"

    has_roi = all(k in df.columns for k in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])

    paths, labels, rois = [], [], []
    miss_file, miss_cls = 0, 0

    for _, r in df.iterrows():
        cid = _as_int(r["ClassId"])
        if cid is None:
            miss_cls += 1
            continue

        if classid_to_idx:
            if cid not in classid_to_idx:
                miss_cls += 1
                continue
            lab = classid_to_idx[cid]
        else:
            lab = cid

        fp = os.path.join(test_dir, os.path.basename(str(r[path_col])))
        if not os.path.isfile(fp):
            miss_file += 1
            continue

        paths.append(fp)
        labels.append(lab)

        if has_roi:
            rois.append((
                int(r["Roi.X1"]),
                int(r["Roi.Y1"]),
                int(r["Roi.X2"]),
                int(r["Roi.Y2"]),
            ))
        else:
            rois.append(None)

    if miss_file:
        print(f"跳过找不到文件的样本数: {miss_file}")
    if miss_cls:
        print(f"跳过类别无法对齐的样本数: {miss_cls}")

    return paths, np.array(labels, dtype=np.int64), rois

TEST_PATHS, TEST_LABELS, TEST_ROIS = load_gtsrb_test_csv(TEST_CSV, TEST_DIR, classid_to_idx)

display(Markdown("## Test Set Summary"))

test_df = pd.DataFrame([
    ["Test samples", len(TEST_PATHS)],
    ["Covered classes", f"{len(set(TEST_LABELS.tolist()))}/{NUM_CLASSES}"],
], columns=["Item", "Value"])

display(test_df)

# ================================================================
#  4. Dataset + Aug
# ================================================================
def make_image_compression(qmin, qmax):
    try:
        return A.ImageCompression(quality_range=(qmin, qmax), p=1.0)
    except TypeError:
        return A.ImageCompression(quality_lower=qmin, quality_upper=qmax, p=1.0)

def make_random_fog():
    try:
        return A.RandomFog(fog_coef_range=(0.15, 0.35), p=1.0)
    except TypeError:
        try:
            return A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.35, p=1.0)
        except TypeError:
            return A.RandomFog(p=1.0)

def make_random_rain():
    try:
        return A.RandomRain(blur_value=2, p=1.0)
    except TypeError:
        return A.RandomRain(p=1.0)

def make_gauss_noise():
    try:
        return A.GaussNoise(var_limit=(20.0, 80.0), p=1.0)
    except TypeError:
        return A.GaussNoise(p=1.0)

def build_eval_aug(mode="clean"):
    ops = [
        A.Resize(RESIZE, RESIZE),
        A.CenterCrop(CROP, CROP),
    ]

    if mode == "blur":
        ops.append(A.MotionBlur(blur_limit=7, p=1.0))

    elif mode == "dark":
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=(-0.50, -0.30),
            contrast_limit=(-0.20, 0.00),
            p=1.0,
        ))

    elif mode == "jpeg":
        ops.append(make_image_compression(25, 45))

    elif mode == "noise":
        ops.append(make_gauss_noise())

    elif mode == "fog":
        ops.append(make_random_fog())

    elif mode == "rain":
        ops.append(make_random_rain())

    ops.extend([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return A.Compose(ops)

class SignDataset(Dataset):
    def __init__(self, paths, labels, aug, rois=None, use_roi=False):
        self.paths = paths
        self.labels = labels
        self.aug = aug
        self.rois = rois
        self.use_roi = use_roi and rois is not None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img is None:
            raise FileNotFoundError(f"读图失败: {self.paths[i]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_roi and self.rois[i] is not None:
            x1, y1, x2, y2 = self.rois[i]
            h, w = img.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            img = img[y1:y2, x1:x2]

        img = self.aug(image=img)["image"]
        return img, int(self.labels[i]), self.paths[i]

def make_loader(mode="clean", shuffle=False):
    ds = SignDataset(
        TEST_PATHS,
        TEST_LABELS,
        build_eval_aug(mode),
        rois=TEST_ROIS,
        use_roi=USE_TEST_ROI,
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

clean_loader = make_loader("clean")

# ================================================================
#  5. 详细评估
# ================================================================
@torch.no_grad()
def evaluate_detail(net, loader):
    net.eval()

    preds, targets, confs, paths = [], [], [], []

    for x, y, pth in loader:
        x = x.to(DEVICE, non_blocking=True)

        if CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            logits = net(x)
            prob = F.softmax(logits, dim=-1)

        conf, pred = prob.max(dim=1)

        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
        confs.append(conf.cpu().numpy())
        paths.extend(list(pth))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    confs = np.concatenate(confs)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return {
        "acc": acc,
        "f1": f1,
        "preds": preds,
        "targets": targets,
        "confs": confs,
        "paths": paths,
    }

eval_res = evaluate_detail(model, clean_loader)

display(Markdown("## Clean Test Result"))

summary_df = pd.DataFrame([
    ["Accuracy", eval_res["acc"] * 100],
    ["Macro-F1", eval_res["f1"] * 100],
], columns=["Metric", "Score (%)"])

display(summary_df)

# Per-class report as visual table
report_dict = classification_report(
    eval_res["targets"],
    eval_res["preds"],
    labels=list(range(NUM_CLASSES)),
    target_names=[str(c) for c in CLASS_NAMES],
    digits=4,
    zero_division=0,
    output_dict=True,
)

report_df = pd.DataFrame(report_dict).T
display(Markdown("## Per-class Report"))
display(report_df)

# Per-class F1 bar chart
per_class_df = report_df.iloc[:NUM_CLASSES].copy()
per_class_df["class"] = [str(c) for c in CLASS_NAMES]
per_class_df["f1-score"] = per_class_df["f1-score"].astype(float)

plt.figure(figsize=(16, 5))
plt.bar(per_class_df["class"], per_class_df["f1-score"] * 100)
plt.xticks(rotation=90)
plt.ylim(0, 100.5)
plt.ylabel("F1 Score (%)")
plt.xlabel("Class")
plt.title("Per-class F1 Score")
plt.tight_layout()
plt.show()

# ================================================================
#  6. 混淆矩阵可视化
# ================================================================
cm = confusion_matrix(
    eval_res["targets"],
    eval_res["preds"],
    labels=list(range(NUM_CLASSES)),
)

display(Markdown("## Confusion Matrix"))

fig, ax = plt.subplots(figsize=(16, 16))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[str(c) for c in CLASS_NAMES],
)
disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=False,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()
plt.show()

# Top confusion pairs
conf_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            conf_pairs.append({
                "true": str(CLASS_NAMES[i]),
                "pred": str(CLASS_NAMES[j]),
                "count": int(cm[i, j]),
            })

conf_pair_df = pd.DataFrame(conf_pairs).sort_values("count", ascending=False)

display(Markdown("## Top Confusion Pairs"))

if len(conf_pair_df) == 0:
    display(Markdown("没有错分对，模型在 clean test 上全部预测正确。"))
else:
    display(conf_pair_df.head(30))

    top_show = conf_pair_df.head(20).copy()
    labels = [f"{r.true} → {r.pred}" for _, r in top_show.iterrows()]

    plt.figure(figsize=(10, max(5, len(top_show) * 0.35)))
    plt.barh(labels[::-1], top_show["count"].values[::-1])
    plt.xlabel("Count")
    plt.title("Top Confusion Pairs")
    plt.tight_layout()
    plt.show()

# ================================================================
#  7. 错分样本、低置信度样本可视化
# ================================================================
def show_sample_grid(indices, title_prefix, max_n=36, cols=6):
    indices = list(indices[:max_n])

    if len(indices) == 0:
        display(Markdown(f"### {title_prefix}: no samples"))
        return

    rows = math.ceil(len(indices) / cols)

    plt.figure(figsize=(cols * 3.0, rows * 3.4))

    for plot_i, idx in enumerate(indices):
        img = cv2.imread(eval_res["paths"][idx])
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y_true = int(eval_res["targets"][idx])
        y_pred = int(eval_res["preds"][idx])
        conf = float(eval_res["confs"][idx])

        true_name = str(CLASS_NAMES[y_true])
        pred_name = str(CLASS_NAMES[y_pred])

        ax = plt.subplot(rows, cols, plot_i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"T: {true_name}\nP: {pred_name}\nC: {conf:.3f}",
            fontsize=9,
        )

    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.show()

wrong_idx = np.where(eval_res["targets"] != eval_res["preds"])[0]
wrong_idx = wrong_idx[np.argsort(-eval_res["confs"][wrong_idx])] if len(wrong_idx) else wrong_idx

low_conf_idx = np.argsort(eval_res["confs"])

display(Markdown("## Wrong Prediction Samples"))
show_sample_grid(
    wrong_idx,
    "Wrong Prediction Samples",
    max_n=SHOW_WRONG_MAX,
    cols=6,
)

display(Markdown("## Low Confidence Samples"))
show_sample_grid(
    low_conf_idx,
    "Low Confidence Samples",
    max_n=SHOW_LOW_CONF_MAX,
    cols=6,
)

# ================================================================
#  8. t-SNE backbone 特征可视化
# ================================================================
@torch.no_grad()
def extract_features(net, loader, max_samples=3000):
    net.eval()

    feats, ys = [], []
    seen = 0

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)

        if CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            f = net.forward_features(x)

            if isinstance(f, (tuple, list)):
                f = f[-1]

            if f.ndim == 4:
                # 兼容 NCHW 和 NHWC
                if f.shape[-1] > f.shape[1] and f.shape[1] <= 32:
                    f = f.mean(dim=(1, 2))
                else:
                    f = F.adaptive_avg_pool2d(f, 1).flatten(1)

            elif f.ndim == 3:
                f = f.mean(dim=1)

            f = F.normalize(f.float(), dim=1)

        feats.append(f.cpu().numpy())
        ys.append(y.numpy())

        seen += x.size(0)
        if seen >= max_samples:
            break

    feats = np.concatenate(feats, axis=0)[:max_samples]
    ys = np.concatenate(ys, axis=0)[:max_samples]

    return feats, ys

def plot_tsne(feats, ys):
    if len(feats) < 20:
        display(Markdown("t-SNE skipped: too few samples."))
        return

    perplexity = min(30, max(5, (len(feats) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(11, 9))
    sc = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=ys,
        s=8,
        alpha=0.8,
    )
    plt.title("t-SNE of Backbone Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(sc, label="Class ID")
    plt.tight_layout()
    plt.show()

if RUN_TSNE:
    display(Markdown("## t-SNE Feature Visualization"))
    tsne_loader = make_loader("clean")
    feats, ys = extract_features(model, tsne_loader, max_samples=TSNE_MAX_SAMPLES)
    plot_tsne(feats, ys)

# ================================================================
#  9. 抗退化测试可视化
# ================================================================
def evaluate_simple(net, loader):
    res = evaluate_detail(net, loader)
    return res["acc"], res["f1"]

if RUN_ROBUSTNESS:
    display(Markdown("## Robustness Evaluation"))

    modes = ["clean", "blur", "dark", "jpeg", "noise", "fog", "rain"]
    robust_rows = []

    for mode in modes:
        loader = make_loader(mode)
        acc, f1 = evaluate_simple(model, loader)

        robust_rows.append({
            "mode": mode,
            "accuracy": acc,
            "macro_f1": f1,
            "accuracy_percent": acc * 100,
            "macro_f1_percent": f1 * 100,
        })

    robust_df = pd.DataFrame(robust_rows)

    display(robust_df[["mode", "accuracy_percent", "macro_f1_percent"]])

    modes = robust_df["mode"].tolist()
    accs = robust_df["accuracy_percent"].values
    f1s = robust_df["macro_f1_percent"].values

    x = np.arange(len(modes))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width / 2, accs, width, label="Accuracy")
    plt.bar(x + width / 2, f1s, width, label="Macro-F1")

    ymin = max(0, min(accs.min(), f1s.min()) - 5)
    plt.ylim(ymin, 100.5)

    plt.xticks(x, modes)
    plt.ylabel("Score (%)")
    plt.title("Robustness Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 降幅图：相比 clean 下降多少
    clean_acc = robust_df.loc[robust_df["mode"] == "clean", "accuracy_percent"].iloc[0]
    clean_f1 = robust_df.loc[robust_df["mode"] == "clean", "macro_f1_percent"].iloc[0]

    robust_df["accuracy_drop"] = clean_acc - robust_df["accuracy_percent"]
    robust_df["macro_f1_drop"] = clean_f1 - robust_df["macro_f1_percent"]

    display(Markdown("### Performance Drop Compared with Clean"))

    display(robust_df[["mode", "accuracy_drop", "macro_f1_drop"]])

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width / 2, robust_df["accuracy_drop"].values, width, label="Accuracy Drop")
    plt.bar(x + width / 2, robust_df["macro_f1_drop"].values, width, label="Macro-F1 Drop")

    plt.xticks(x, modes)
    plt.ylabel("Drop (%)")
    plt.title("Robustness Drop Compared with Clean")
    plt.legend()
    plt.tight_layout()
    plt.show()

display(Markdown("## Done"))
