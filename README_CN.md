[English](README.md) | [简体中文](README_CN.md)

# 两阶段鲁棒交通标志分类

基于 **PyTorch + timm** 的交通标志分类项目，以 **GTSRB** 为 Stage 1 基础数据集，支持 7 种主干网络、跨数据集微调、加权 Soft-Voting 集成以及多种图像退化鲁棒性测试。

## 快速开始

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

推荐 Python 3.10+。

### 我应该运行哪个脚本？

| 需求 | 脚本 |
|---|---|
| 在 GTSRB 上训练鲁棒 Backbone | `src/stage1.py` |
| 将 Stage 1 模型迁移到其他交通标志数据集 | `src/stage2.py` |
| 评估一个 Stage 1 模型 | `evaluation/test.py` |
| 多模型评估 / 加权集成 | `evaluation/ensemble_evaluation.py` |

---

## 支持的主干网络

| Key | timm 模型 |
|---|---|
| `convnext` | `convnextv2_base.fcmae_ft_in22k_in1k` |
| `swin` | `swinv2_small_window16_256.ms_in1k` |
| `eva02` | `eva02_base_patch14_224.mim_in22k` |
| `effnet` | `tf_efficientnetv2_s.in21k_ft_in1k` |
| `caformer` | `caformer_s18.sail_in1k` |
| `maxvit` | `maxvit_tiny_rw_224.sw_in1k` |
| `coatnet` | `coatnet_0_rw_224.sw_in1k` |

---

## 原始预训练权重

以下为 timm 官方托管在 Hugging Face 上的上游预训练权重。

| Backbone | 直接下载 |
|---|---|
| ConvNeXt V2 | [model.safetensors](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k/resolve/main/model.safetensors?download=true) |
| Swin V2 | [model.safetensors](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k/resolve/main/model.safetensors?download=true) |
| EVA-02 | [model.safetensors](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k/resolve/main/model.safetensors?download=true) |
| EfficientNetV2 | [model.safetensors](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/resolve/main/model.safetensors?download=true) |
| CAFormer | [model.safetensors](https://huggingface.co/timm/caformer_s18.sail_in1k/resolve/main/model.safetensors?download=true) |
| MaxViT | [model.safetensors](https://huggingface.co/timm/maxvit_tiny_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) |
| CoAtNet | [model.safetensors](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/model.safetensors?download=true) |

---

## 本项目训练权重

https://huggingface.co/Shijunqiang/TrafficSignClassification

---

## 数据集目录

### GTSRB

```text
data/
└── GTSRB/
    ├── Train/
    │   ├── 0/
    │   ├── 1/
    │   └── ...
    ├── Test/
    └── Test.csv
```

官方数据集：[GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

### Stage 2 目标数据集

Stage 2 使用单标签 ImageFolder 风格数据：

```text
data/
└── Newtrain/
    ├── class_a/
    ├── class_b/
    └── class_c/
```

目标数据集**不需要仍然是 GTSRB 的 43 类**。

Stage 2 会复用 Stage 1 Backbone，丢弃原分类头，再根据新数据集类别数建立新的分类头。

如果原数据集是目标检测格式，请先根据 bbox 裁剪交通标志图像。

---

## 使用方法

### 1. Stage 1：在 GTSRB 上训练

```bash
MODEL_KEY=coatnet \
GTSRB_ROOT=./data/GTSRB \
STAGE1_OUT_DIR=./outputs/stage1_coatnet \
python src/stage1.py
```

使用本地原始预训练权重：

```bash
MODEL_KEY=convnext \
CONVNEXT_WEIGHTS=./weights/pretrained/convnextv2_base.safetensors \
GTSRB_ROOT=./data/GTSRB \
python src/stage1.py
```

常见输出：

```text
<model>_stage1_best.pth
<model>_stage1_last.pth
<model>_stage1_meta.json
<model>_stage1_history.json
```

### 2. Stage 2：迁移到其他交通标志数据集

```bash
MODEL_KEY=eva02 \
STAGE2_TRAIN_DIR=./data/Newtrain \
STAGE1_ROOT=./weights/stage1 \
STAGE2_OUT_DIR=./outputs/stage2_eva02 \
python src/stage2.py
```

### 3. 单模型评估

```bash
STAGE1_CKPT=./weights/stage1/model_v1.1_convnext_gtsrb.pth \
STAGE1_META_JSON=./weights/stage1/model_v1.1_convnext_stage1_meta.json \
GTSRB_ROOT=./data/GTSRB \
python evaluation/test.py
```

### 4. 多模型集成

等权：

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

自定义权重：

```bash
SELECTED_MODELS=effnet,caformer,maxvit \
ENSEMBLE_WEIGHTS_JSON='{"effnet":0.45,"caformer":0.35,"maxvit":0.20}' \
GTSRB_ROOT=./data/GTSRB \
STAGE1_MODEL_ROOT=./weights/stage1 \
python evaluation/ensemble_evaluation.py
```

---

## 评估

支持：

- Accuracy / Macro-F1
- 每类别指标
- 混淆矩阵
- 错分 / 低置信度样本
- t-SNE
- Clean / Blur / Dark / JPEG / Noise / Fog / Rain 鲁棒性测试

### 已报告的 GTSRB Clean Test 结果

| 指标 | 得分 |
|---|---:|
| Accuracy | **99.7941%** |
| Macro Precision | **99.6782%** |
| Macro Recall | **99.7934%** |
| Macro F1 | **99.7329%** |
| Weighted F1 | **99.7944%** |

测试集：**12,630 张图像 / 43 类**

> 具体结果会随 checkpoint、模型组合、预处理和 ensemble 权重变化。

![Per-class F1](assets/per_class_f1.png)

![Top confusion pairs](assets/top_confusion_pairs.png)

![t-SNE features](assets/tsne_features.png)

---

## 项目结构

```text
.
├── README.md
├── README_CN.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── stage1.py
│   └── stage2.py
├── evaluation/
│   ├── test.py
│   └── ensemble_evaluation.py
└── assets/
    ├── per_class_f1.png
    ├── top_confusion_pairs.png
    └── tsne_features.png
```

---

## 实验复现

- checkpoint 与 Stage 1 metadata 一起保存；
- 记录 PyTorch / timm / Albumentations 版本；
- 不随意改变类别文件夹排序；
- 固定训练和评估随机种子；
- 记录集成模型组合及权重。

---

## Citation

```bibtex
@article{stallkamp2012man,
  title={Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition},
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  journal={Neural Networks},
  volume={32},
  pages={323--332},
  year={2012},
  publisher={Elsevier}
}
```

    
