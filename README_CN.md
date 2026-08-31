# 基于 GTSRB 的两阶段鲁棒交通标志分类

[English](README.md) | **简体中文**

本项目基于 **PyTorch + timm**，面向 **GTSRB（German Traffic Sign Recognition Benchmark）交通标志分类**，实现了一个两阶段训练框架，支持 7 种主干网络、低光照与多种图像退化鲁棒性评估，以及可选择的加权 Soft-Voting 多模型集成。

项目主要包含四个脚本：

- `src/stage1.py` —— Stage 1：在 GTSRB 上训练具有鲁棒性的交通标志主干网络。
- `src/stage2.py` —— Stage 2：在新的 ImageFolder 格式交通标志数据集上进行跨数据集微调 / 域适配。
- `evaluation/test.py` —— Stage 1 单模型详细评估。
- `evaluation/ensemble_evaluation.py` —— 单模型或多模型集成评估。

---

## 项目亮点

- **支持 7 种主干网络**
  - ConvNeXt V2
  - Swin Transformer V2
  - EVA-02
  - EfficientNetV2
  - CAFormer
  - MaxViT
  - CoAtNet
- 两阶段训练流程。
- 针对低光照场景的训练与验证。
- 面向交通标志任务的数据增强策略。
- 支持 EMA、混合精度、梯度累积、学习率调度与 Early Stopping。
- 自动对齐不同模型的 GTSRB `ClassId` 顺序。
- 支持可选的 **加权 Soft-Voting 多模型集成**。
- Clean 测试集详细分析：
  - Accuracy
  - Macro-F1
  - 每类别 Precision / Recall / F1
  - 混淆矩阵
  - 高频混淆类别对
  - 错分样本
  - 低置信度样本
  - t-SNE 主干特征可视化
- 鲁棒性测试：
  - 模糊
  - 轻度 / 中度 / 重度暗光
  - JPEG 压缩
  - 高斯噪声
  - 雾
  - 雨
- 多模型评估时使用可复现的图像退化随机种子，保证不同模型尽量接受相同的扰动样本。

---

## 整体流程

```text
                       GTSRB
                         │
                         ▼
              ┌─────────────────────┐
              │       Stage 1       │
              │  鲁棒交通标志主干    │
              │       训练           │
              └──────────┬──────────┘
                         │
                  Stage 1 权重
                         │
                         ▼
                  新交通标志数据集
                         │
                         ▼
              ┌─────────────────────┐
              │       Stage 2       │
              │   微调 / 域适配      │
              └──────────┬──────────┘
                         │
                         ▼
                      最终模型


Stage 1 checkpoints
        │
        ├──────────────► 单模型评估
        │
        └──────────────► 可选加权模型集成
                              │
                              ▼
                       Clean + 鲁棒性指标
```

---

## 支持的主干网络

| Key | timm 模型名称 | 输入 Crop |
|---|---|---:|
| `convnext` | `convnextv2_base.fcmae_ft_in22k_in1k` | 256 |
| `swin` | `swinv2_small_window16_256.ms_in1k` | 256 |
| `eva02` | `eva02_base_patch14_224.mim_in22k` | 224 |
| `effnet` | `tf_efficientnetv2_s.in21k_ft_in1k` | 256 |
| `caformer` | `caformer_s18.sail_in1k` | 224 |
| `maxvit` | `maxvit_tiny_rw_224.sw_in1k` | 224 |
| `coatnet` | `coatnet_0_rw_224.sw_in1k` | 224 |

---

## 原始预训练权重

下表中的链接指向 **timm 官方在 Hugging Face 上提供的预训练模型文件**。

> 这些是模型初始化使用的上游预训练权重，**不是本项目训练得到的 GTSRB 权重**。

| Backbone | 官方预训练权重 | 官方模型页面 | 上游许可证* |
|---|---|---|---|
| ConvNeXt V2 | [下载 `model.safetensors`](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k/resolve/main/model.safetensors?download=true) | [timm/convnextv2_base.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k) | CC-BY-NC-4.0 |
| Swin V2 | [下载 `model.safetensors`](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k/resolve/main/model.safetensors?download=true) | [timm/swinv2_small_window16_256.ms_in1k](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k) | MIT |
| EVA-02 | [下载 `model.safetensors`](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k/resolve/main/model.safetensors?download=true) | [timm/eva02_base_patch14_224.mim_in22k](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k) | MIT |
| EfficientNetV2 | [下载 `model.safetensors`](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/resolve/main/model.safetensors?download=true) | [timm/tf_efficientnetv2_s.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k) | Apache-2.0 |
| CAFormer | [下载 `model.safetensors`](https://huggingface.co/timm/caformer_s18.sail_in1k/resolve/main/model.safetensors?download=true) | [timm/caformer_s18.sail_in1k](https://huggingface.co/timm/caformer_s18.sail_in1k) | Apache-2.0 |
| MaxViT | [下载 `model.safetensors`](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) | [timm/maxvit_tiny_rw_224.sw_in1k](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k) | Apache-2.0 |
| CoAtNet | [下载 `model.safetensors`](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) | [timm/coatnet_0_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k) | Apache-2.0 |

\* 在重新分发或商业使用前，请务必检查各上游模型卡中的许可证。上游模型和数据集许可证独立于本仓库代码许可证。

也可以直接通过 timm 初始化：

```python
import timm

model = timm.create_model(
    "convnextv2_base.fcmae_ft_in22k_in1k",
    pretrained=True,
)
```

---

## 本项目训练好的权重

建议将训练完成的模型权重托管在 Hugging Face，而不是直接提交到 GitHub 普通 Git 历史中。

请将下面的：

```text
YOUR_USERNAME/YOUR_REPO
```

替换成你的 Hugging Face 用户名和模型仓库名。

### Stage 1 — GTSRB 训练权重

| Backbone | 训练权重 |
|---|---|
| ConvNeXt V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v1.1_convnext_gtsrb.pth) |
| Swin V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v2.1_swin_gtsrb.pth) |
| EVA-02 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v3.1_eva02_stage1_best.pth) |
| EfficientNetV2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v4.1_effnet_gtsrb.pth) |
| CAFormer | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v5.1_caformer_stage1_best.pth) |
| MaxViT | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v6.1_maxvit_stage1_best.pth) |
| CoAtNet | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage1/model_v7.1_coatnet_stage1_best.pth) |

建议同时上传每个 Stage 1 checkpoint 对应的 metadata 文件，例如：

```text
model_v5.1_caformer_stage1_best.pth
model_v5.1_caformer_stage1_meta.json
```

### Stage 2 — 跨数据集微调权重

| Backbone | Stage 2 权重 |
|---|---|
| ConvNeXt V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/convnext_best.pth) |
| Swin V2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/swin_best.pth) |
| EVA-02 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/eva02_best.pth) |
| EfficientNetV2 | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/effnet_best.pth) |
| CAFormer | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/caformer_best.pth) |
| MaxViT | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/maxvit_best.pth) |
| CoAtNet | [Hugging Face](https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/stage2/coatnet_best.pth) |

---

## 数据集

### GTSRB

Stage 1 使用 **German Traffic Sign Recognition Benchmark (GTSRB)** 进行训练，并在官方 Test Set 上进行最终评估。

官方页面：

- [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html)

评估代码默认 GTSRB 的标准 `ClassId` 为：

```text
0 ~ 42
```

推荐目录结构：

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

数据集本身**不包含在本仓库中**。

### Stage 2 目标数据集

`stage2.py` 面向 ImageFolder 风格的交通标志分类数据集。

默认结构：

```text
data/
└── Newtrain/
    ├── 0_class_name/
    ├── 1_class_name/
    ├── ...
    └── N_class_name/
```

Stage 2 **不要求目标数据集仍然是 GTSRB 的 43 个类别**。

其流程是：

```text
GTSRB Stage 1
       │
       ▼
交通标志领域 Backbone
       │
       ├── 丢弃原 GTSRB 分类头
       │
       ▼
重新创建 N 类分类头
       │
       ▼
目标交通标志数据集微调
```

因此可以用于其他单标签交通标志分类数据集，例如不同国家、不同类别数量、不同采集环境的数据集。

只要目标数据整理成：

```text
dataset/
├── class_a/
├── class_b/
└── class_c/
```

即可自动扫描类别数并重新建立分类头。

> 当前 Stage 2 是**图像分类任务**。如果原数据集是目标检测格式（包含 bbox），需要先将交通标志区域裁剪成分类图像后再使用。

对于类似：

```text
0_stop
1_yield
2_no_entry
```

这种以数字开头的类别文件夹，Stage 2 会按照数字顺序排序；普通字符串类别名则按照字符串顺序排序。

---

## 项目结构

```text
.
├── README.md
├── README_CN.md
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
├── data/                       # Git 忽略
│   ├── GTSRB/
│   └── Newtrain/
│
├── weights/                    # Git 忽略
│   ├── pretrained/
│   └── stage1/
│
└── outputs/                    # Git 忽略
```

---

## 环境安装

推荐使用 Python 3.10+。

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

主要依赖：

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

---

## Stage 1 训练

Stage 1 在 GTSRB 上训练交通标志鲁棒 Backbone。

训练中包含：

- 各模型对应的输入分辨率与优化配置
- 低光照模拟
- 面向交通标志的增强策略
- Mixed Precision
- EMA
- 类别不平衡处理
- Early Stopping
- 本地 / Hugging Face 预训练权重加载

### 示例

```bash
MODEL_KEY=coatnet \
GTSRB_ROOT=./data/GTSRB \
STAGE1_OUT_DIR=./outputs/stage1_coatnet \
python src/stage1.py
```

可选模型：

```text
convnext
swin
eva02
effnet
caformer
maxvit
coatnet
```

### 使用本地下载的原始预训练权重

```bash
MODEL_KEY=convnext \
CONVNEXT_WEIGHTS=./weights/pretrained/convnextv2_base.safetensors \
GTSRB_ROOT=./data/GTSRB \
python src/stage1.py
```

如果本地权重不存在，也可以使用配置好的 timm / Hugging Face 预训练源。

### Stage 1 常见输出

```text
outputs/stage1_.../
├── <model>_stage1_best.pth
├── <model>_stage1_last.pth
├── <model>_stage1_meta.json
└── <model>_stage1_history.json
```

Metadata 会记录模型名称、类别顺序、输入分辨率以及鲁棒性相关配置，便于后续评估和复现。

---

## Stage 2：跨交通标志数据集迁移学习

Stage 2 从 Stage 1 checkpoint 中加载交通标志领域的 backbone 参数，然后根据新数据集自动重建分类头并进行微调。

这意味着：

```text
Stage 1:
GTSRB 43 类
Backbone + 43-class head

              ↓

Stage 2:
复用 Backbone
删除旧分类头
重新创建 N-class head
```

因此：

```text
Stage1 类别数 ≠ Stage2 类别数
```

是允许的。

目标数据集可以是其他交通标志分类数据集，也可以是自行采集的数据，只要采用单标签 ImageFolder 目录结构。

### 示例

```bash
MODEL_KEY=eva02 \
STAGE2_TRAIN_DIR=./data/Newtrain \
STAGE1_ROOT=./weights/stage1 \
STAGE2_OUT_DIR=./outputs/stage2_eva02 \
python src/stage2.py
```

也可以直接指定某个 Stage 1 权重：

```bash
MODEL_KEY=eva02 \
STAGE1_EVA02_WEIGHTS=./weights/stage1/model_v3.1_eva02_stage1_best.pth \
STAGE2_TRAIN_DIR=./data/Newtrain \
python src/stage2.py
```

常见输出：

```text
outputs/stage2_<model>/
├── <model>_best.pth
├── <model>_history.csv
├── <model>_history.json
├── <model>_summary.json
└── class_mapping.json
```

---

## 单模型评估

`evaluation/test.py` 用于对单个 Stage 1 模型进行详细评估。

支持：

- Accuracy
- Macro-F1
- 每类别 Classification Report
- Per-class F1 图
- 混淆矩阵
- Top Confusion Pairs
- 错分样本
- 低置信度样本
- Backbone t-SNE
- Clean / 图像退化鲁棒性评估

### 示例

```bash
STAGE1_CKPT=./weights/stage1/model_v1.1_convnext_gtsrb.pth \
STAGE1_META_JSON=./weights/stage1/model_v1.1_convnext_stage1_meta.json \
GTSRB_ROOT=./data/GTSRB \
python evaluation/test.py
```

---

## 多模型集成评估

`evaluation/ensemble_evaluation.py` 支持：

- 单模型
- 任意多个模型组合
- 等权 Soft Voting
- 自定义权重 Soft Voting

每个模型都会：

1. 独立加载；
2. 使用各自对应的输入预处理；
3. 将输出类别顺序转换为标准 GTSRB `ClassId`；
4. 计算 softmax probability；
5. 按给定权重进行概率加权平均。

这样可以避免不同 checkpoint 内部类别顺序不一致导致错误集成。

### 等权集成

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

没有指定权重时，所有模型等权。

### 自定义加权 Soft Voting

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
ENSEMBLE_WEIGHTS_JSON='{"effnet":0.45,"caformer":0.35,"maxvit":0.20}' \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

### 只评估单个模型

```bash
SELECTED_MODELS=maxvit \
python evaluation/ensemble_evaluation.py
```

### 指定 t-SNE Backbone

模型集成本身没有唯一的 Backbone Feature，因此 t-SNE 必须选择其中一个模型：

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
TSNE_MODEL_KEY=maxvit \
python evaluation/ensemble_evaluation.py
```

---

## 鲁棒性评估

支持以下测试模式：

| Mode | 含义 |
|---|---|
| `clean` | 原始测试图像 |
| `blur` | 运动模糊 |
| `dark_mild` | 轻度暗光 |
| `dark_mid` | 中度暗光 |
| `dark_heavy` | 重度暗光 |
| `jpeg` | JPEG 压缩失真 |
| `noise` | 高斯噪声 |
| `fog` | 模拟雾 |
| `rain` | 模拟雨 |

对于随机退化操作，每张图像使用：

```text
evaluation seed + corruption-mode offset + sample index
```

生成随机种子。

因此在多模型对比中，同一图像在同一退化模式下会尽量使用相同随机扰动，从而让模型之间的鲁棒性比较更加公平。

---

## Clean Test 性能

以下结果来自一次 GTSRB 官方测试集评估，共：

- **12,630 张测试图像**
- **43 个类别**

### 总体指标

| 指标 | 得分 |
|---|---:|
| Accuracy | **99.7941%** |
| Macro Precision | **99.6782%** |
| Macro Recall | **99.7934%** |
| Macro F1 | **99.7329%** |
| Weighted Precision | **99.7993%** |
| Weighted Recall | **99.7941%** |
| Weighted F1 | **99.7944%** |

绝大多数类别都取得了接近满分的识别结果。相对困难的类别主要集中在少数类别，例如 `18`、`21` 和 `31`，这与后面的混淆类别对结果一致。

### 每类别 Classification Report

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

> 以上数据来自对应实验输出，具体结果取决于所使用的 checkpoint / 模型组合及 ensemble 权重。

---

## 代表性评估可视化

### Backbone Feature t-SNE

特征空间中可以观察到大量紧凑且类别分离明显的簇，说明模型对于大多数 GTSRB 类别已经学习到了较强的判别特征。

![t-SNE of backbone features](assets/tsne_features.png)

### 高频混淆类别对

错误主要集中在少数类别对中。在该次实验中较明显的混淆包括：

```text
18 → 31
18 → 21
```

![Top confusion pairs](assets/top_confusion_pairs.png)

### Per-class F1

大多数类别 F1 接近 1.0，仅少量类别相对困难。

![Per-class F1 score](assets/per_class_f1.png)

> 图中结果来自一次代表性实验。最终结果会受到 checkpoint、输入预处理、模型组合以及 ensemble 权重影响。

---

## 集成设计说明

### 标准类别顺序对齐

GTSRB 标准类别为：

```text
0, 1, 2, ..., 42
```

在 Soft Voting 前，每个模型的输出列都会根据 Stage 1 metadata 映射到统一的 GTSRB `ClassId` 顺序。

如果不同模型内部类别顺序不同，却直接平均 logits / probabilities，会得到错误的 ensemble 结果。

### Probability-level Soft Voting

对模型 \(m\)：

```text
p_m(y | x) = softmax(logits_m(x))
```

集成概率：

```text
p_ensemble(y | x) = sum_m w_m * p_m(y | x)
```

其中：

```text
w_m > 0
sum_m w_m = 1
```

最终预测取 ensemble probability 最大的类别。

### GPU 显存策略

集成评估**不要求所有模型同时放在 GPU 中**。

模型会依次：

```text
加载模型
   ↓
推理
   ↓
概率保存到 CPU
   ↓
释放模型显存
   ↓
加载下一个模型
```

最后再在 CPU 上组合概率。

因此即使 EVA-02 等较大模型也更容易进行多模型评估。

---

## 环境变量配置

GitHub 版本脚本已经尽量避免写死 Kaggle / AutoDL 用户路径。

常见变量：

| 环境变量 | 作用 |
|---|---|
| `PROJECT_ROOT` | 项目根目录 |
| `MODEL_KEY` | 选择 Backbone |
| `GTSRB_ROOT` | GTSRB 数据集根目录 |
| `GTSRB_TRAIN_DIR` | 指定 GTSRB Train |
| `GTSRB_TEST_DIR` | 指定 GTSRB Test |
| `GTSRB_TEST_CSV` | 指定 GTSRB `Test.csv` |
| `STAGE1_OUT_DIR` | Stage 1 输出目录 |
| `STAGE1_ROOT` / `STAGE1_MODEL_ROOT` | Stage 1 权重目录 |
| `STAGE2_TRAIN_DIR` | Stage 2 数据集目录 |
| `STAGE2_OUT_DIR` | Stage 2 输出目录 |
| `SELECTED_MODELS` | 多模型 key，逗号分隔 |
| `ENSEMBLE_WEIGHTS_JSON` | Ensemble 权重 JSON |
| `TSNE_MODEL_KEY` | t-SNE 使用的 Backbone |
| `RUN_TSNE` | 是否运行 t-SNE |
| `RUN_ROBUSTNESS` | 是否运行鲁棒性评估 |
| `EVAL_SEED` | 评估随机种子 |

---

## 大文件管理

不建议直接提交以下文件到普通 Git 历史：

```text
*.pth
*.pt
*.ckpt
*.safetensors
data/
weights/
outputs/
```

训练权重推荐托管到：

- Hugging Face Hub
- GitHub Releases
- Git LFS

本项目更推荐 Hugging Face Hub，便于同时托管：

```text
checkpoint
metadata
模型说明
下载链接
```

---

## 实验复现建议

为了提高实验可复现性，建议：

1. 保留 Stage 1 生成的 metadata 文件；
2. 记录准确的 `timm`、PyTorch 和 Albumentations 版本；
3. 不要随意改变类别文件夹顺序；
4. 保持训练与评估随机种子；
5. 记录 ensemble 使用的模型子集及权重；
6. 最终 GTSRB 指标使用官方 Test Split。

---

## Citation

如果使用 GTSRB，请引用原始论文：

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

如果使用对应的预训练 Backbone，也建议引用对应模型论文以及 `timm` 项目。

---

## 致谢

本项目基于以下开源项目与数据集：

- [GTSRB / German Traffic Sign Benchmarks](https://benchmark.ini.rub.de/)
- [PyTorch](https://pytorch.org/)
- [timm / PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [scikit-learn](https://scikit-learn.org/)
- [Hugging Face Hub](https://huggingface.co/)

预训练模型、GTSRB 数据集以及第三方库均遵循各自的许可证与使用条款。

---

## License

在正式公开仓库前，请在 `LICENSE` 文件中指定本项目代码许可证。

本仓库自身的代码许可证**不会覆盖**：

- GTSRB 数据集许可证
- 预训练模型权重许可证
- 第三方依赖许可证
