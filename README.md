# VIPL - Urban Scene Segmentation Project
# VIPL - 城市场景分割项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 📋 Overview

A computer vision toolkit for urban scene segmentation using Mask2Former, SAM 2.1, and SAM 3 on the Cityscapes dataset. Provides 5 scripts covering ground extraction, building segmentation, multi-class parsing, and instance segmentation.

### 🎯 Features

- **Ground Segmentation**: Road + sidewalk via Mask2Former panoptic segmentation
- **Building Segmentation**: Dedicated building mask extraction
- **Multi-Class Parsing**: 5-class urban parsing (road, sidewalk, vegetation, terrain, sky)
- **Instance Segmentation (SAM 2.1)**: Automatic mask generation for all objects
- **Text-Prompted Segmentation (SAM 3)**: Language-guided 6-class segmentation
- **GPU Acceleration**: CUDA support with optimized batch processing
- **Color Visualization**: Contrast images for every segmentation type

### 🏗️ Project Structure

```
VIPL/
├── datadets_cityscapes/            # Cityscapes dataset (not included)
│   ├── leftImg8bit/{train,val,test}/{city}/
│   └── gtFine/{train,val,test}/{city}/
├── models/                         # Pre-trained models (not included)
│   ├── mask2former/mask2former_cityscapes/
│   ├── sam2/sam2.1_hiera_large.pt
│   └── sam3/
├── repos/
│   ├── Mask2Former/                # (not included)
│   ├── detectron2/                 # (not included)
│   └── ground_parser/              # ✅ Scripts in this repo
│       ├── run_cityscapes_ground_mask2former.py
│       ├── run_cityscapes_multiclass_mask2former.py
│       ├── run_cityscapes_building_mask2former.py
│       ├── run_cityscapes_ground_sam2.py
│       └── run_cityscapes_ground_sam3.py
└── outputs_masks*/                 # Generated masks (not included)
```

### 🚀 Getting Started

#### Prerequisites
- Python 3.8+, PyTorch 1.10+, CUDA GPU (8 GB+ VRAM)

#### Installation

```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl/VIPL/repos/detectron2 && pip install -e .
cd ../Mask2Former && pip install -r requirements.txt
pip install transformers opencv-python pillow tqdm omegaconf hydra-core
```

#### Dataset
Download [Cityscapes](https://www.cityscapes-dataset.com/) → extract to `VIPL/datadets_cityscapes/`

#### Models

| Model | Download | Path |
|-------|----------|------|
| Mask2Former (Cityscapes) | [Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) | `models/mask2former/mask2former_cityscapes/` |
| SAM 2.1 Large | [Meta AI](https://github.com/facebookresearch/segment-anything-2) | `models/sam2/sam2.1_hiera_large.pt` |
| SAM 3 | [Hugging Face](https://huggingface.co/facebook/sam3) | `models/sam3/` |

---

### 💻 Scripts

| Script | Model | Target Classes | Output Dir |
|--------|-------|---------------|------------|
| `run_cityscapes_ground_mask2former.py` | Mask2Former | road, sidewalk | `outputs_masks/` |
| `run_cityscapes_multiclass_mask2former.py` | Mask2Former | road, sidewalk, vegetation, terrain, sky | `outputs_masks_multiclass/` |
| `run_cityscapes_building_mask2former.py` | Mask2Former | building | `outputs_masks_building/` |
| `run_cityscapes_ground_sam2.py` | SAM 2.1 | all objects (automatic) | `outputs_masks_sam/` |
| `run_cityscapes_ground_sam3.py` | SAM 3 | road, sidewalk, building, vegetation, terrain, sky | `outputs_masks_sam3/` |

All scripts run from `VIPL/repos/ground_parser/`. Edit paths at the top of each file before running.

---

#### 1. Ground Segmentation — `run_cityscapes_ground_mask2former.py`

```bash
python run_cityscapes_ground_mask2former.py
```

Config:
```python
CITYSCAPES_ROOT = r"path/to/datadets_cityscapes"
OUTPUT_ROOT     = r"path/to/outputs_masks"
MODEL_PATH      = r"path/to/models/mask2former/mask2former_cityscapes"
TARGET_LABELS   = {"road": ["road"], "sidewalk": ["sidewalk", "pavement"]}
```

Output per image:
```
{base}_road.png       # Binary mask (255=road)
{base}_sidewalk.png
{base}_ground.png     # Combined road + sidewalk
{base}_contrast.png   # Overlay: red=road, blue=sidewalk
```

---

#### 2. Multi-Class Segmentation — `run_cityscapes_multiclass_mask2former.py`

```bash
python run_cityscapes_multiclass_mask2former.py
```

Config:
```python
TARGET_LABELS = {
    "road": ["road"], "sidewalk": ["sidewalk", "pavement"],
    "vegetation": ["vegetation", "tree", "plant"],
    "terrain": ["terrain", "ground"], "sky": ["sky"]
}
```

Output: `{base}_{class}.png` for each class + `{base}_contrast.png`

---

#### 3. Building Segmentation — `run_cityscapes_building_mask2former.py`

```bash
python run_cityscapes_building_mask2former.py
```

Config:
```python
TARGET_LABELS = {"building": ["building", "buildings"]}
```

Output: `{base}_building.png` + `{base}_contrast.png`

---

#### 4. Instance Segmentation (SAM 2.1) — `run_cityscapes_ground_sam2.py`

```bash
python run_cityscapes_ground_sam2.py
```

Config:
```python
CHECKPOINT_PATH        = r"path/to/models/sam2/sam2.1_hiera_large.pt"
POINTS_PER_SIDE        = 16    # 32=accurate, 16=~4x faster
PRED_IOU_THRESH        = 0.86
STABILITY_SCORE_THRESH = 0.92
MIN_MASK_REGION_AREA   = 200
```

Output: `{base}_sam_masks.png` — color-coded instance segmentation

---

#### 5. Text-Prompted Segmentation (SAM 3) — `run_cityscapes_ground_sam3.py`

```bash
python run_cityscapes_ground_sam3.py
```

Config:
```python
MODEL_DIR    = r"path/to/models/sam3"
CATEGORIES   = {"road": "road", "sidewalk": "sidewalk", "building": "building",
                "vegetation": "vegetation", "terrain": "terrain", "sky": "sky"}
SCORE_THRESH = 0.5
MASK_THRESH  = 0.5
```

Output: `{base}_{class}.png` per category + `{base}_contrast.png`

---

### 🧪 Experiments & Results

#### Performance

| Script | Model | Speed (img/s) | GPU Memory | Classes |
|--------|-------|:---:|:---:|:---:|
| `ground_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 2 |
| `multiclass_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 5 |
| `building_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 1 |
| `ground_sam2` | SAM 2.1 | ~0.5–1 | 8–12 GB | all |
| `ground_sam3` | SAM 3 | ~0.5–1 | 8–12 GB | 6 |

> Tested on NVIDIA RTX 3080/4090, input 2048×1024.

#### Optimization Tips

1. **SAM speed**: `POINTS_PER_SIDE = 16` → ~4× faster than default 32
2. **Noise filter**: Increase `MIN_MASK_REGION_AREA` to suppress small spurious masks
3. **Recall**: Lower `PRED_IOU_THRESH` to detect more small objects
4. **Memory**: Use FP16 for larger batch sizes

#### Class → Color Mapping

| Class | Scripts | Contrast Color |
|-------|---------|:-:|
| Road | ground, multiclass, SAM 3 | Red |
| Sidewalk | ground, multiclass, SAM 3 | Blue |
| Building | building, SAM 3 | Yellow |
| Vegetation | multiclass, SAM 3 | Green |
| Terrain | multiclass, SAM 3 | Brown |
| Sky | multiclass, SAM 3 | Cyan |
| All objects | SAM 2.1 | Random per instance |

---

### 📚 Research Context

Supports research in:
- Autonomous driving perception
- Urban scene reconstruction and simulation
- Robot navigation and traversable surface detection
- 3D scene generation from segmentation priors

### 🛠️ Technologies

| Tool | Role | Reference |
|------|------|-----------|
| Mask2Former | Panoptic segmentation | [arXiv:2112.01527](https://arxiv.org/abs/2112.01527) |
| SAM 2.1 | Universal instance segmentation | [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) |
| SAM 3 | Text-prompted segmentation | [Meta AI](https://github.com/facebookresearch/sam3) |
| Detectron2 | Detection/segmentation platform | [GitHub](https://github.com/facebookresearch/detectron2) |
| PyTorch | Deep learning framework | [pytorch.org](https://pytorch.org/) |
| Transformers | Model loading | [huggingface.co](https://huggingface.co/) |
| Cityscapes | Urban benchmark dataset | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |

### ⚠️ Notes

- Model files and datasets are **not** included (Mask2Former ~200 MB, SAM 2.1 ~900 MB, Cityscapes ~11 GB)
- Mask2Former: 8 GB+ VRAM recommended; SAM: 12 GB+ VRAM recommended
- Cityscapes requires registration and license agreement

### 📝 License
MIT License — see [LICENSE](LICENSE)

### 🙏 Acknowledgments
- [Cityscapes](https://www.cityscapes-dataset.com/) · [Mask2Former](https://github.com/facebookresearch/Mask2Former) · [SAM](https://github.com/facebookresearch/segment-anything-2) · [Detectron2](https://github.com/facebookresearch/detectron2)

---

<a name="chinese"></a>
## 📋 项目概述

使用 Mask2Former、SAM 2.1 和 SAM 3 在 Cityscapes 数据集上进行城市场景语义分割的工具包。提供 5 个脚本，涵盖地面提取、建筑分割、多类别解析和实例分割。

### 🎯 主要功能

- **地面分割**：使用 Mask2Former 提取道路和人行道
- **建筑分割**：专用建筑物掩码提取
- **多类别解析**：5 类城市解析（道路、人行道、植被、地形、天空）
- **实例分割（SAM 2.1）**：全自动物体掩码生成
- **文本提示分割（SAM 3）**：基于语言的 6 类分割
- **GPU 加速**：CUDA 支持，优化批处理
- **颜色可视化**：每种分割类型生成对比图

### 🏗️ 项目结构

```
VIPL/
├── datadets_cityscapes/            # Cityscapes 数据集（不含）
│   ├── leftImg8bit/{train,val,test}/{city}/
│   └── gtFine/{train,val,test}/{city}/
├── models/                         # 预训练模型（不含）
│   ├── mask2former/mask2former_cityscapes/
│   ├── sam2/sam2.1_hiera_large.pt
│   └── sam3/
├── repos/
│   ├── Mask2Former/                # 不含
│   ├── detectron2/                 # 不含
│   └── ground_parser/              # ✅ 本仓库核心脚本
│       ├── run_cityscapes_ground_mask2former.py
│       ├── run_cityscapes_multiclass_mask2former.py
│       ├── run_cityscapes_building_mask2former.py
│       ├── run_cityscapes_ground_sam2.py
│       └── run_cityscapes_ground_sam3.py
└── outputs_masks*/                 # 生成掩码（不含）
```

### 🚀 快速开始

#### 环境要求
- Python 3.8+，PyTorch 1.10+，CUDA GPU（8 GB+ 显存）

#### 安装

```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl/VIPL/repos/detectron2 && pip install -e .
cd ../Mask2Former && pip install -r requirements.txt
pip install transformers opencv-python pillow tqdm omegaconf hydra-core
```

#### 数据集
从[官方网站](https://www.cityscapes-dataset.com/)下载 → 解压至 `VIPL/datadets_cityscapes/`

#### 模型下载

| 模型 | 下载 | 放置路径 |
|------|------|---------|
| Mask2Former (Cityscapes) | [Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) | `models/mask2former/mask2former_cityscapes/` |
| SAM 2.1 Large | [Meta AI](https://github.com/facebookresearch/segment-anything-2) | `models/sam2/sam2.1_hiera_large.pt` |
| SAM 3 | [Hugging Face](https://huggingface.co/facebook/sam3) | `models/sam3/` |

---

### 💻 脚本列表

| 脚本 | 模型 | 目标类别 | 输出目录 |
|------|------|---------|---------|
| `run_cityscapes_ground_mask2former.py` | Mask2Former | 道路、人行道 | `outputs_masks/` |
| `run_cityscapes_multiclass_mask2former.py` | Mask2Former | 道路、人行道、植被、地形、天空 | `outputs_masks_multiclass/` |
| `run_cityscapes_building_mask2former.py` | Mask2Former | 建筑 | `outputs_masks_building/` |
| `run_cityscapes_ground_sam2.py` | SAM 2.1 | 所有物体（自动） | `outputs_masks_sam/` |
| `run_cityscapes_ground_sam3.py` | SAM 3 | 道路、人行道、建筑、植被、地形、天空 | `outputs_masks_sam3/` |

所有脚本在 `VIPL/repos/ground_parser/` 目录下运行，运行前修改脚本顶部路径配置。

---

#### 1. 地面分割 — `run_cityscapes_ground_mask2former.py`

```bash
python run_cityscapes_ground_mask2former.py
```

配置：
```python
CITYSCAPES_ROOT = r"path/to/datadets_cityscapes"
OUTPUT_ROOT     = r"path/to/outputs_masks"
MODEL_PATH      = r"path/to/models/mask2former/mask2former_cityscapes"
TARGET_LABELS   = {"road": ["road"], "sidewalk": ["sidewalk", "pavement"]}
```

每张图像输出：
```
{base}_road.png       # 道路二值掩码（255=道路）
{base}_sidewalk.png   # 人行道二值掩码
{base}_ground.png     # 组合掩码
{base}_contrast.png   # 叠加图（红=道路，蓝=人行道）
```

---

#### 2. 多类别分割 — `run_cityscapes_multiclass_mask2former.py`

```bash
python run_cityscapes_multiclass_mask2former.py
```

配置：
```python
TARGET_LABELS = {
    "road": ["road"], "sidewalk": ["sidewalk", "pavement"],
    "vegetation": ["vegetation", "tree", "plant"],
    "terrain": ["terrain", "ground"], "sky": ["sky"]
}
```

每张图像输出：各类别 `{base}_{class}.png` + `{base}_contrast.png`

---

#### 3. 建筑分割 — `run_cityscapes_building_mask2former.py`

```bash
python run_cityscapes_building_mask2former.py
```

配置：
```python
TARGET_LABELS = {"building": ["building", "buildings"]}
```

每张图像输出：`{base}_building.png` + `{base}_contrast.png`

---

#### 4. 实例分割（SAM 2.1）— `run_cityscapes_ground_sam2.py`

```bash
python run_cityscapes_ground_sam2.py
```

配置：
```python
CHECKPOINT_PATH        = r"path/to/models/sam2/sam2.1_hiera_large.pt"
POINTS_PER_SIDE        = 16    # 32=精确，16=快速（约4倍速度提升）
PRED_IOU_THRESH        = 0.86
STABILITY_SCORE_THRESH = 0.92
MIN_MASK_REGION_AREA   = 200
```

输出：`{base}_sam_masks.png` — 颜色编码的实例分割

---

#### 5. 文本提示分割（SAM 3）— `run_cityscapes_ground_sam3.py`

```bash
python run_cityscapes_ground_sam3.py
```

配置：
```python
MODEL_DIR    = r"path/to/models/sam3"
CATEGORIES   = {"road": "road", "sidewalk": "sidewalk", "building": "building",
                "vegetation": "vegetation", "terrain": "terrain", "sky": "sky"}
SCORE_THRESH = 0.5
MASK_THRESH  = 0.5
```

输出：各类别 `{base}_{class}.png` + `{base}_contrast.png`

---

### 🧪 实验与结果

#### 性能对比

| 脚本 | 模型 | 速度（图/秒） | 显存 | 类别数 |
|------|------|:---:|:---:|:---:|
| `ground_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 2 |
| `multiclass_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 5 |
| `building_mask2former` | Mask2Former | ~2–3 | 4–6 GB | 1 |
| `ground_sam2` | SAM 2.1 | ~0.5–1 | 8–12 GB | 全部 |
| `ground_sam3` | SAM 3 | ~0.5–1 | 8–12 GB | 6 |

> 测试硬件：NVIDIA RTX 3080/4090，输入分辨率 2048×1024

#### 优化建议

1. **SAM 速度**：`POINTS_PER_SIDE = 16` 约快 4 倍
2. **噪声过滤**：增大 `MIN_MASK_REGION_AREA` 抑制小噪声区域
3. **召回率**：降低 `PRED_IOU_THRESH` 检测更多小物体
4. **显存**：使用 FP16 混合精度扩大批次

#### 类别颜色映射

| 类别 | 脚本 | 对比图颜色 |
|------|------|:---:|
| 道路 | ground、multiclass、SAM 3 | 红 |
| 人行道 | ground、multiclass、SAM 3 | 蓝 |
| 建筑 | building、SAM 3 | 黄 |
| 植被 | multiclass、SAM 3 | 绿 |
| 地形 | multiclass、SAM 3 | 棕 |
| 天空 | multiclass、SAM 3 | 青 |
| 所有物体 | SAM 2.1 | 随机（按实例） |

---

### 📚 研究背景

支持以下方向的研究：
- 自动驾驶感知
- 城市场景重建与仿真
- 机器人导航与可通行表面检测
- 基于分割先验的 3D 场景生成

### 🛠️ 使用技术

| 工具 | 用途 | 来源 |
|------|------|------|
| Mask2Former | 全景分割 | [arXiv:2112.01527](https://arxiv.org/abs/2112.01527) |
| SAM 2.1 | 通用实例分割 | [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) |
| SAM 3 | 文本提示分割 | [Meta AI](https://github.com/facebookresearch/sam3) |
| Detectron2 | 检测/分割平台 | [GitHub](https://github.com/facebookresearch/detectron2) |
| PyTorch | 深度学习框架 | [pytorch.org](https://pytorch.org/) |
| Transformers | 模型加载 | [huggingface.co](https://huggingface.co/) |
| Cityscapes | 城市基准数据集 | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |

### ⚠️ 注意事项

- 模型文件和数据集**不包含**在仓库中（Mask2Former ~200 MB，SAM 2.1 ~900 MB，Cityscapes ~11 GB）
- Mask2Former 推荐 8 GB+ 显存；SAM 推荐 12 GB+ 显存
- Cityscapes 数据集需注册并同意许可协议

### 📝 许可证
MIT 许可证 — 详见 [LICENSE](LICENSE)

### 🙏 致谢
[Cityscapes](https://www.cityscapes-dataset.com/) · [Mask2Former](https://github.com/facebookresearch/Mask2Former) · [SAM](https://github.com/facebookresearch/segment-anything-2) · [Detectron2](https://github.com/facebookresearch/detectron2)
