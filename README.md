# Two-Stage Robust Traffic Sign Classification on GTSRB

A PyTorch/timm project for **robust German Traffic Sign Recognition Benchmark (GTSRB) classification** with a two-stage training pipeline, seven supported backbones, low-light and corruption robustness evaluation, and selectable weighted soft-voting ensembles.

The project is designed around four main scripts:

- `src/stage1.py` — Stage 1 robust backbone training on GTSRB.
- `src/stage2.py` — Stage 2 fine-tuning / domain adaptation on a new ImageFolder-style dataset.
- `evaluation/test.py` — detailed single-model Stage 1 evaluation.
- `evaluation/ensemble_evaluation.py` — single-model or multi-model ensemble evaluation.

---

## Highlights

- **7 supported backbones**
  - ConvNeXt V2
  - Swin Transformer V2
  - EVA-02
  - EfficientNetV2
  - CAFormer
  - MaxViT
  - CoAtNet
- Two-stage training workflow.
- Low-light-aware training and validation.
- Traffic-sign-friendly image augmentation.
- EMA, mixed precision, gradient accumulation, learning-rate scheduling, and early stopping.
- Canonical GTSRB `ClassId` alignment across independently trained models.
- Selectable **weighted soft-voting ensemble**.
- Detailed clean-set analysis:
  - Accuracy
  - Macro-F1
  - Per-class precision / recall / F1
  - Confusion matrix
  - Top confusion pairs
  - Wrong predictions
  - Low-confidence predictions
  - t-SNE backbone feature visualization
- Robustness evaluation under:
  - blur
  - mild / medium / heavy darkness
  - JPEG compression
  - Gaussian noise
  - fog
  - rain
- Reproducible corruption sampling across different models during ensemble evaluation.

---

## Pipeline

```text
                       GTSRB
                         │
                         ▼
              ┌─────────────────────┐
              │       Stage 1       │
              │ Robust backbone     │
              │ training            │
              └──────────┬──────────┘
                         │
                  Stage 1 checkpoint
                         │
                         ▼
                 New training data
                         │
                         ▼
              ┌─────────────────────┐
              │       Stage 2       │
              │ Fine-tuning /       │
              │ adaptation          │
              └──────────┬──────────┘
                         │
                         ▼
                    Final model

Stage 1 checkpoints
        │
        ├──────────────► Single-model evaluation
        │
        └──────────────► Selectable weighted ensemble
                              │
                              ▼
                    Clean + robustness metrics
```

---

## Supported Backbones

| Key | timm model | Input crop |
|---|---|---:|
| `convnext` | `convnextv2_base.fcmae_ft_in22k_in1k` | 256 |
| `swin` | `swinv2_small_window16_256.ms_in1k` | 256 |
| `eva02` | `eva02_base_patch14_224.mim_in22k` | 224 |
| `effnet` | `tf_efficientnetv2_s.in21k_ft_in1k` | 256 |
| `caformer` | `caformer_s18.sail_in1k` | 224 |
| `maxvit` | `maxvit_tiny_rw_224.sw_in1k` | 224 |
| `coatnet` | `coatnet_0_rw_224.sw_in1k` | 224 |

---

## Original Pretrained Weights

The links below point directly to the **official timm model files hosted on Hugging Face**.

> These are the upstream pretrained initialization weights, **not** the GTSRB-trained checkpoints produced by this project.

| Backbone | Official pretrained weight | Upstream model card | Upstream license* |
|---|---|---|---|
| ConvNeXt V2 | [Download `model.safetensors`](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k/resolve/main/model.safetensors?download=true) | [timm/convnextv2_base.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k) | CC-BY-NC-4.0 |
| Swin V2 | [Download `model.safetensors`](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k/resolve/main/model.safetensors?download=true) | [timm/swinv2_small_window16_256.ms_in1k](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k) | MIT |
| EVA-02 | [Download `model.safetensors`](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k/resolve/main/model.safetensors?download=true) | [timm/eva02_base_patch14_224.mim_in22k](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k) | MIT |
| EfficientNetV2 | [Download `model.safetensors`](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/resolve/main/model.safetensors?download=true) | [timm/tf_efficientnetv2_s.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k) | Apache-2.0 |
| CAFormer | [Download `model.safetensors`](https://huggingface.co/timm/caformer_s18.sail_in1k/resolve/main/model.safetensors?download=true) | [timm/caformer_s18.sail_in1k](https://huggingface.co/timm/caformer_s18.sail_in1k) | Apache-2.0 |
| MaxViT | [Download `model.safetensors`](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) | [timm/maxvit_tiny_rw_224.sw_in1k](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k) | Apache-2.0 |
| CoAtNet | [Download `model.safetensors`](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) | [timm/coatnet_0_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k) | Apache-2.0 |

\* Always check the upstream model card before redistribution or commercial use. The upstream model/dataset licenses are independent of the license of this repository.

The same models can also be initialized directly through timm:

```python
import timm

model = timm.create_model(
    "convnextv2_base.fcmae_ft_in22k_in1k",
    pretrained=True,
)
```

---

## Trained Checkpoints

The trained weights are intended to be hosted on Hugging Face rather than committed directly to GitHub.

Replace `YOUR_USERNAME/YOUR_REPO` with your Hugging Face repository.

### Stage 1 — GTSRB-trained backbones

| Backbone | Trained checkpoint |
|---|---|
| ConvNeXt V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v1.1_convnext_gtsrb.pth) |
| Swin V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v2.1_swin_gtsrb.pth) |
| EVA-02 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v3.1_eva02_stage1_best.pth) |
| EfficientNetV2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v4.1_effnet_gtsrb.pth) |
| CAFormer | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v5.1_caformer_stage1_best.pth) |
| MaxViT | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v6.1_maxvit_stage1_best.pth) |
| CoAtNet | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v7.1_coatnet_stage1_best.pth) |

It is recommended to upload each Stage 1 metadata file next to its checkpoint as well, for example:

```text
model_v5.1_caformer_stage1_best.pth
model_v5.1_caformer_stage1_meta.json
```

### Stage 2 — fine-tuned models

| Backbone | Stage 2 checkpoint |
|---|---|
| ConvNeXt V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/convnext_best.pth) |
| Swin V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/swin_best.pth) |
| EVA-02 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/eva02_best.pth) |
| EfficientNetV2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/effnet_best.pth) |
| CAFormer | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/caformer_best.pth) |
| MaxViT | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/maxvit_best.pth) |
| CoAtNet | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/coatnet_best.pth) |

---

## Dataset

### GTSRB

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** for Stage 1 training and official test evaluation.

Official benchmark:

- [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html)

The evaluation code assumes canonical GTSRB `ClassId` values `0..42`.

Recommended local layout:

```text
data/
└── GTSRB/
    ├── Train/
    │   ├── 0/
    │   ├── 1/
    │   ├── ...
    │   └── 42/
    ├── Test/
    └── Test.csv
```

The dataset itself is **not included in this repository**.

### Stage 2 dataset

`stage2.py` expects an ImageFolder-style training dataset. By default:

```text
data/
└── Newtrain/
    ├── 0_class_name/
    ├── 1_class_name/
    ├── ...
    └── 42_class_name/
```

Folder names beginning with a numeric class ID are sorted numerically by the Stage 2 loader.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── stage1.py
│   └── stage2.py
│
├── evaluation/
│   ├── test.py
│   └── ensemble_evaluation.py
│
├── assets/
│   ├── tsne_features.png
│   ├── top_confusion_pairs.png
│   └── per_class_f1.png
│
├── data/                       # ignored by Git
│   ├── GTSRB/
│   └── Newtrain/
│
├── weights/                    # ignored by Git
│   ├── pretrained/
│   └── stage1/
│
└── outputs/                    # ignored by Git
```

---

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Core dependencies:

```text
torch
torchvision
timm
albumentations
opencv-python-headless
numpy
pandas
scikit-learn
matplotlib
safetensors
tqdm
Pillow
ipython
```

If you do not use a `requirements.txt` file yet:

```bash
pip install torch torchvision timm albumentations opencv-python-headless \
    numpy pandas scikit-learn matplotlib safetensors tqdm Pillow ipython
```

---

## Stage 1 Training

Stage 1 trains a robust backbone on GTSRB.

The training code includes model-specific input sizes and optimization settings, low-light simulation, traffic-sign-aware augmentation, mixed precision, EMA, class-aware training logic, early stopping, and optional pretrained-weight loading.

### Example

```bash
MODEL_KEY=coatnet \
GTSRB_ROOT=./data/GTSRB \
STAGE1_OUT_DIR=./outputs/stage1_coatnet \
python src/stage1.py
```

Available model keys:

```text
convnext
swin
eva02
effnet
caformer
maxvit
coatnet
```

### Use a locally downloaded upstream weight

Example:

```bash
MODEL_KEY=convnext \
CONVNEXT_WEIGHTS=./weights/pretrained/convnextv2_base.safetensors \
GTSRB_ROOT=./data/GTSRB \
python src/stage1.py
```

Alternatively, when local weights are unavailable, Stage 1 can use the configured timm/Hugging Face pretrained source.

### Typical Stage 1 outputs

```text
outputs/stage1_.../
├── <model>_stage1_best.pth
├── <model>_stage1_last.pth
├── <model>_stage1_meta.json
└── <model>_stage1_history.json
```

The metadata file records information such as model name, class ordering, input resolution, normalization-related settings, and robustness configuration required for reproducible evaluation.

---

## Stage 2 Training

Stage 2 initializes from a Stage 1 checkpoint and fine-tunes the selected backbone on a new training dataset.

### Example

```bash
MODEL_KEY=eva02 \
STAGE2_TRAIN_DIR=./data/Newtrain \
STAGE1_ROOT=./weights/stage1 \
STAGE2_OUT_DIR=./outputs/stage2_eva02 \
python src/stage2.py
```

You can override an individual Stage 1 checkpoint:

```bash
MODEL_KEY=eva02 \
STAGE1_EVA02_WEIGHTS=./weights/stage1/model_v3.1_eva02_stage1_best.pth \
STAGE2_TRAIN_DIR=./data/Newtrain \
python src/stage2.py
```

Typical outputs include:

```text
outputs/stage2_<model>/
├── <model>_best.pth
├── <model>_history.csv
├── <model>_history.json
├── <model>_summary.json
└── class_mapping.json
```

---

## Single-Model Evaluation

`evaluation/test.py` performs a detailed evaluation of one Stage 1 model on the official GTSRB test set.

It supports:

- Accuracy and Macro-F1
- Per-class classification report
- Per-class F1 chart
- Confusion matrix
- Top confusion pairs
- Wrong-prediction visualization
- Low-confidence sample visualization
- Backbone t-SNE
- Clean and corruption robustness evaluation

### Example

```bash
STAGE1_CKPT=./weights/stage1/model_v1.1_convnext_gtsrb.pth \
STAGE1_META_JSON=./weights/stage1/model_v1.1_convnext_stage1_meta.json \
GTSRB_ROOT=./data/GTSRB \
python evaluation/test.py
```

---

## Multi-Model Ensemble Evaluation

`evaluation/ensemble_evaluation.py` supports either a single model or any subset of the seven backbones.

Each selected model is:

1. loaded independently;
2. evaluated using its own preprocessing configuration;
3. reordered to canonical GTSRB `ClassId` order;
4. converted to softmax probabilities;
5. combined with weighted soft voting.

This prevents incorrect ensemble behavior when checkpoints were trained with different internal output orders.

### Equal-weight ensemble

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

When no explicit ensemble weights are supplied, all selected models receive equal weight.

### Custom weighted soft voting

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
ENSEMBLE_WEIGHTS_JSON='{"effnet":0.45,"caformer":0.35,"maxvit":0.20}' \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

### Single model through the ensemble evaluator

```bash
SELECTED_MODELS=maxvit \
python evaluation/ensemble_evaluation.py
```

### Choose the backbone used for t-SNE

An ensemble has no unique backbone representation, so t-SNE uses one selected backbone:

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
TSNE_MODEL_KEY=maxvit \
python evaluation/ensemble_evaluation.py
```

---

## Robustness Evaluation

The ensemble evaluator supports the following test modes:

| Mode | Purpose |
|---|---|
| `clean` | Original test image |
| `blur` | Motion blur |
| `dark_mild` | Mild low-light degradation |
| `dark_mid` | Medium low-light degradation |
| `dark_heavy` | Strong low-light degradation |
| `jpeg` | JPEG compression artifacts |
| `noise` | Gaussian image noise |
| `fog` | Synthetic fog |
| `rain` | Synthetic rain |

For stochastic corruption transforms, the evaluator derives a deterministic per-image seed from:

```text
evaluation seed + corruption-mode offset + sample index
```

Therefore, different selected backbones are evaluated on the same sampled corruption for a given test image and mode.

---


## Clean Test Performance

The following results are from the reported clean GTSRB test evaluation containing **12,630 test images across 43 classes**.

### Overall Metrics

| Metric | Score |
|---|---:|
| Accuracy | **99.7941%** |
| Macro Precision | **99.6782%** |
| Macro Recall | **99.7934%** |
| Macro F1 | **99.7329%** |
| Weighted Precision | **99.7993%** |
| Weighted Recall | **99.7941%** |
| Weighted F1 | **99.7944%** |

The model achieves near-perfect performance on most GTSRB classes. The most challenging classes in this run are concentrated in a small subset, especially ClassIds `18`, `21`, and `31`, which is consistent with the top-confusion-pair visualization below.

### Per-class Classification Report

| ClassId | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 1 | 0.998613 | 1.000000 | 0.999306 | 720 |
| 2 | 1.000000 | 1.000000 | 1.000000 | 750 |
| 3 | 1.000000 | 0.991111 | 0.995536 | 450 |
| 4 | 1.000000 | 1.000000 | 1.000000 | 660 |
| 5 | 0.995253 | 0.998413 | 0.996830 | 630 |
| 6 | 1.000000 | 1.000000 | 1.000000 | 150 |
| 7 | 1.000000 | 1.000000 | 1.000000 | 450 |
| 8 | 1.000000 | 1.000000 | 1.000000 | 450 |
| 9 | 1.000000 | 1.000000 | 1.000000 | 480 |
| 10 | 1.000000 | 1.000000 | 1.000000 | 660 |
| 11 | 0.995249 | 0.997619 | 0.996433 | 420 |
| 12 | 1.000000 | 1.000000 | 1.000000 | 690 |
| 13 | 1.000000 | 1.000000 | 1.000000 | 720 |
| 14 | 1.000000 | 1.000000 | 1.000000 | 270 |
| 15 | 1.000000 | 1.000000 | 1.000000 | 210 |
| 16 | 1.000000 | 1.000000 | 1.000000 | 150 |
| 17 | 1.000000 | 1.000000 | 1.000000 | 360 |
| 18 | 1.000000 | 0.961538 | 0.980392 | 390 |
| 19 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 20 | 1.000000 | 1.000000 | 1.000000 | 90 |
| 21 | 0.936170 | 0.977778 | 0.956522 | 90 |
| 22 | 1.000000 | 1.000000 | 1.000000 | 120 |
| 23 | 1.000000 | 1.000000 | 1.000000 | 150 |
| 24 | 1.000000 | 1.000000 | 1.000000 | 90 |
| 25 | 0.997912 | 0.995833 | 0.996872 | 480 |
| 26 | 1.000000 | 1.000000 | 1.000000 | 180 |
| 27 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 28 | 1.000000 | 1.000000 | 1.000000 | 150 |
| 29 | 1.000000 | 1.000000 | 1.000000 | 90 |
| 30 | 0.993377 | 1.000000 | 0.996678 | 150 |
| 31 | 0.964286 | 1.000000 | 0.981818 | 270 |
| 32 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 33 | 1.000000 | 1.000000 | 1.000000 | 210 |
| 34 | 1.000000 | 1.000000 | 1.000000 | 120 |
| 35 | 1.000000 | 1.000000 | 1.000000 | 390 |
| 36 | 0.991736 | 1.000000 | 0.995851 | 120 |
| 37 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 38 | 1.000000 | 1.000000 | 1.000000 | 690 |
| 39 | 1.000000 | 1.000000 | 1.000000 | 90 |
| 40 | 1.000000 | 0.988889 | 0.994413 | 90 |
| 41 | 1.000000 | 1.000000 | 1.000000 | 60 |
| 42 | 0.989011 | 1.000000 | 0.994475 | 90 |
| **Accuracy** | **0.997941** | **0.997941** | **0.997941** | **12,630** |
| **Macro avg** | **0.996782** | **0.997934** | **0.997329** | **12,630** |
| **Weighted avg** | **0.997993** | **0.997941** | **0.997944** | **12,630** |

> The values above are copied from the reported evaluation output. They correspond to the specific trained checkpoint / predictor configuration used for that run.

---

## Representative Evaluation Visualizations

The following figures are examples generated by the evaluation pipeline.

### Backbone feature t-SNE

The feature space forms many compact class-specific clusters, indicating strong separation between most GTSRB categories.

![t-SNE of backbone features](assets/tsne_features.png)

### Top confusion pairs

Errors are concentrated in a small number of class pairs. In the representative run shown below, the largest confusion counts are `18 → 31` and `18 → 21`.

![Top confusion pairs](assets/top_confusion_pairs.png)

### Per-class F1

Most classes achieve very high F1 scores in the shown evaluation run, while a small subset of classes remains comparatively more difficult.

![Per-class F1 score](assets/per_class_f1.png)

> These figures are representative outputs from one run. Final numbers depend on the selected checkpoint(s), preprocessing configuration, model subset, and ensemble weights.

---

## Ensemble Design Notes

### Canonical class alignment

GTSRB uses 43 canonical classes:

```text
0, 1, 2, ..., 42
```

Before soft voting, every model's output columns are remapped into this canonical order using its Stage 1 metadata.

This is important because averaging logits/probabilities from models with inconsistent class-index order would produce invalid ensemble predictions.

### Probability-level soft voting

For model \(m\), let:

```text
p_m(y | x) = softmax(logits_m(x))
```

The ensemble probability is:

```text
p_ensemble(y | x) = sum_m w_m * p_m(y | x)
```

with:

```text
w_m > 0
sum_m w_m = 1
```

The final prediction is the class with the largest ensemble probability.

### GPU memory behavior

The ensemble evaluator does **not** require all selected networks to stay in GPU memory simultaneously.

Models are loaded and evaluated one at a time; their probability arrays are retained on CPU and combined afterwards. This makes evaluation practical for larger backbones such as EVA-02.

---

## Configuration Through Environment Variables

The GitHub-ready scripts avoid hard-coded Kaggle/AutoDL user paths.

Common variables include:

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT` | Repository root |
| `MODEL_KEY` | Backbone selection |
| `GTSRB_ROOT` | GTSRB root directory |
| `GTSRB_TRAIN_DIR` | Override GTSRB training directory |
| `GTSRB_TEST_DIR` | Override GTSRB test directory |
| `GTSRB_TEST_CSV` | Override GTSRB `Test.csv` |
| `STAGE1_OUT_DIR` | Stage 1 output directory |
| `STAGE1_ROOT` / `STAGE1_MODEL_ROOT` | Stage 1 checkpoint directory |
| `STAGE2_TRAIN_DIR` | Stage 2 ImageFolder root |
| `STAGE2_OUT_DIR` | Stage 2 output directory |
| `SELECTED_MODELS` | Comma-separated ensemble model keys |
| `ENSEMBLE_WEIGHTS_JSON` | JSON object containing ensemble weights |
| `TSNE_MODEL_KEY` | Backbone used for t-SNE |
| `RUN_TSNE` | Enable / disable t-SNE |
| `RUN_ROBUSTNESS` | Enable / disable corruption evaluation |
| `EVAL_SEED` | Evaluation random seed |

---

## Notes on Large Files

Do not commit the following directly to normal Git history:

```text
*.pth
*.pt
*.ckpt
*.safetensors
data/
weights/
outputs/
```

Recommended options for trained checkpoints:

- Hugging Face Hub
- GitHub Releases
- Git LFS

For this project, Hugging Face Hub is recommended because it provides a clean location for checkpoint + metadata distribution.

---

## Reproducibility

For reproducible experiments:

1. keep the metadata JSON produced by Stage 1;
2. record the exact `timm`, PyTorch, and Albumentations versions;
3. keep the class-folder naming/order unchanged;
4. keep the configured training/evaluation seeds;
5. report the exact ensemble model subset and normalized weights;
6. use the official GTSRB test split for final comparison.

---

## Citation

If you use GTSRB, please cite the original benchmark work:

```bibtex
@article{stallkamp2012man,
  title   = {Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition},
  author  = {Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  journal = {Neural Networks},
  volume  = {32},
  pages   = {323--332},
  year    = {2012},
  publisher = {Elsevier}
}
```

If you use the pretrained backbones, please also cite the corresponding model papers and the `timm` library where appropriate.

---

## Acknowledgements

This project builds on:

- [GTSRB / German Traffic Sign Benchmarks](https://benchmark.ini.rub.de/)
- [PyTorch](https://pytorch.org/)
- [timm / PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [scikit-learn](https://scikit-learn.org/)
- [Hugging Face Hub](https://huggingface.co/)

The pretrained models, GTSRB dataset, and third-party libraries remain subject to their respective licenses and terms.

---

## License

Add the repository's code license in `LICENSE` before public release.

This repository's license does **not** override the licenses of GTSRB, pretrained model weights, or other third-party assets.
