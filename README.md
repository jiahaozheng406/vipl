# VIPL - Urban Scene Segmentation Project
# VIPL - 城市场景分割项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 📋 Overview

A comprehensive computer vision toolkit for urban scene understanding and semantic segmentation using state-of-the-art models (Mask2Former, SAM 2.1) on the Cityscapes dataset. This project provides multiple segmentation approaches for extracting ground surfaces, multi-class objects, and instance-level masks from urban street scenes.

### 🎯 Features

- **Ground Surface Extraction**: Automatically extracts binary masks for roads and sidewalks using Mask2Former
- **Multi-Class Segmentation**: Extracts masks for road, sidewalk, vegetation, terrain, and sky
- **Instance Segmentation**: Full object segmentation using SAM 2.1 with automatic mask generation
- **Cityscapes Dataset Support**: Processes standard Cityscapes dataset structure (train/val/test splits)
- **GPU Acceleration**: CUDA support for faster inference with optimized batch processing
- **Visualization**: Generates color-coded contrast images for easy inspection
- **Multiple Output Formats**: Produces separate masks for each class and combined visualizations
- **Flexible Configuration**: Easy-to-modify paths and parameters for different setups

### 🏗️ Project Structure

```
VIPL/
├── datadets_cityscapes/        # Cityscapes dataset (not included)
│   ├── leftImg8bit/            # Input images
│   └── gtFine/                 # Ground truth annotations
├── models/                     # Pre-trained models (not included)
│   └── mask2former/            # Mask2Former model files
├── repos/                      # External repositories
│   ├── Mask2Former/            # Mask2Former implementation
│   ├── detectron2/             # Detectron2 framework
│   └── ground_parser/          # Ground segmentation scripts
└── outputs_masks/              # Generated segmentation masks (not included)
```

**Note**: Large files (datasets, models, outputs) are excluded from the repository.

### 🚀 Getting Started

#### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- Git

#### Installation

1. Clone the repository:
```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl
```

2. Install dependencies:
```bash
cd VIPL/repos/Mask2Former
pip install -r requirements.txt
```

3. Install Detectron2:
```bash
cd ../detectron2
pip install -e .
```

4. Install additional requirements:
```bash
pip install transformers opencv-python pillow tqdm
```

#### Dataset Setup

1. Download the Cityscapes dataset from [official website](https://www.cityscapes-dataset.com/)
2. Extract to `VIPL/datadets_cityscapes/`
3. Ensure the following structure:

#### Model Setup

Download the pre-trained Mask2Former model for Cityscapes and place it in:
```
VIPL/models/mask2former/mask2former_cityscapes/
```

### 💻 Usage

The project includes three main segmentation scripts:

#### 1. Ground Surface Segmentation (Mask2Former)

Extracts binary masks for roads and sidewalks:

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_mask2former.py
```

**Output files per image:**
- `{base}_road.png` - Binary mask for road surfaces (255=road, 0=background)
- `{base}_sidewalk.png` - Binary mask for sidewalk surfaces
- `{base}_ground.png` - Combined ground surface mask (road + sidewalk)
- `{base}_contrast.png` - Visualization overlay (red=road, blue=sidewalk)

**Configuration:**
```python
CITYSCAPES_ROOT = r"path/to/cityscapes"
OUTPUT_ROOT = r"path/to/output"
MODEL_PATH = r"path/to/mask2former_cityscapes"
```

#### 2. Multi-Class Segmentation (Mask2Former)

Extracts masks for multiple urban scene classes:

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_multiclass_mask2former.py
```

**Target classes:**
- Road
- Sidewalk
- Vegetation (trees, plants)
- Terrain (ground)
- Sky

**Output files per image:**
- `{base}_road.png`
- `{base}_sidewalk.png`
- `{base}_vegetation.png`
- `{base}_terrain.png`
- `{base}_sky.png`
- `{base}_contrast.png` - Multi-color visualization

#### 3. Instance Segmentation (SAM 2.1)

Full automatic object segmentation using Segment Anything Model:

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_sam.py
```

**Features:**
- Automatic mask generation for all objects
- Optimized parameters for faster processing (16 points per side)
- Color-coded instance masks
- Configurable thresholds for quality control

**Configuration:**
```python
CHECKPOINT_PATH = r"path/to/sam2.1_hiera_large.pt"
POINTS_PER_SIDE = 16  # Sampling density
PRED_IOU_THRESH = 0.86  # Quality threshold
MIN_MASK_REGION_AREA = 200  # Minimum object size
```

**Output:**
- `{base}_sam_masks.png` - Color-coded instance segmentation
- Individual mask files for each detected object

### 📊 Supported Classes

#### Mask2Former (Cityscapes)
The segmentation models target the following Cityscapes classes:
- **Road**: Main road surfaces
- **Sidewalk**: Pedestrian walkways and pavements
- **Vegetation**: Trees, plants, and greenery
- **Terrain**: Natural ground surfaces
- **Sky**: Sky regions

#### SAM 2.1 (Universal)
- Automatic detection of all objects in the scene
- Instance-level segmentation without predefined classes
- Suitable for general object extraction and analysis

### 🧪 Experiments & Results

#### Performance Metrics

| Model | Speed (img/s) | GPU Memory | Accuracy |
|-------|--------------|------------|----------|
| Mask2Former (Ground) | ~2-3 | 4-6 GB | High precision for road/sidewalk |
| Mask2Former (Multi) | ~2-3 | 4-6 GB | Excellent for urban classes |
| SAM 2.1 (Optimized) | ~0.5-1 | 8-12 GB | Universal object detection |

**Hardware tested:** NVIDIA RTX 3080/4090

#### Output Examples

**Ground Segmentation:**
- Input: Urban street scene (2048×1024)
- Output: Binary masks + color overlay
- Processing time: ~0.3-0.5s per image

**Multi-Class Segmentation:**
- Input: Urban street scene
- Output: 5 separate class masks + visualization
- Processing time: ~0.3-0.5s per image

**SAM Instance Segmentation:**
- Input: Urban street scene
- Output: 50-200 instance masks per image
- Processing time: ~1-2s per image (optimized)

#### Optimization Tips

1. **Batch Processing**: Process multiple images in parallel
2. **GPU Memory**: Use mixed precision (FP16) for larger batches
3. **SAM Speed**: Reduce `POINTS_PER_SIDE` from 32 to 16 for 4x speedup
4. **Quality vs Speed**: Adjust IOU thresholds based on requirements

### 🔧 Configuration

Each script has configurable parameters at the top of the file:

#### Ground Segmentation Script
```python
# run_cityscapes_ground_mask2former.py
CITYSCAPES_ROOT = r"E:\vipl\VIPL\datadets_cityscapes"
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks"
MODEL_PATH = r"E:\vipl\VIPL\models\mask2former\mask2former_cityscapes"

TARGET_LABELS = {
    "road": ["road"],
    "sidewalk": ["sidewalk", "side walk", "pavement"]
}
```

#### Multi-Class Segmentation Script
```python
# run_cityscapes_multiclass_mask2former.py
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_multiclass"

TARGET_LABELS = {
    "road": ["road"],
    "sidewalk": ["sidewalk", "side walk", "pavement"],
    "vegetation": ["vegetation", "tree", "plant"],
    "terrain": ["terrain", "ground"],
    "sky": ["sky"]
}
```

#### SAM Instance Segmentation Script
```python
# run_cityscapes_ground_sam.py
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_sam"
CHECKPOINT_PATH = r"E:\vipl\VIPL\models\sam2\sam2.1_hiera_large.pt"

# Performance tuning
POINTS_PER_SIDE = 16  # 16=fast, 32=accurate
PRED_IOU_THRESH = 0.86  # Higher=stricter quality
STABILITY_SCORE_THRESH = 0.92
MIN_MASK_REGION_AREA = 200  # Filter small objects
```

### 📚 Research Context

This project focuses on ground surface segmentation for urban scene understanding, which is fundamental for:
- Autonomous driving and navigation
- Urban scene reconstruction
- Robotics simulation environments
- 3D scene generation

### 🛠️ Technologies Used

- **Mask2Former**: Universal image segmentation architecture for panoptic segmentation
  - Paper: [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
  - Pre-trained on Cityscapes dataset (19 classes)

- **SAM 2.1 (Segment Anything Model)**: Universal instance segmentation
  - Paper: [Segment Anything](https://arxiv.org/abs/2304.02643)
  - Zero-shot object segmentation capability

- **Detectron2**: Facebook AI Research's detection and segmentation platform
  - Modular design for computer vision tasks

- **PyTorch**: Deep learning framework (1.10+)
- **Transformers**: Hugging Face model library for Mask2Former
- **OpenCV**: Computer vision operations and image processing
- **Cityscapes Dataset**: Urban street scene benchmark (5000 fine annotations)

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

### 🙏 Acknowledgments

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) for providing the urban scene dataset
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) for the segmentation architecture
- [Detectron2](https://github.com/facebookresearch/detectron2) for the detection framework

### ⚠️ Notes

- **Large Files**: Model files and datasets are not included in this repository due to size constraints
  - Mask2Former model: ~200MB
  - SAM 2.1 model: ~900MB
  - Cityscapes dataset: ~11GB (leftImg8bit) + ~250MB (gtFine)

- **Hardware Requirements**:
  - GPU with 8GB+ VRAM recommended for Mask2Former
  - GPU with 12GB+ VRAM recommended for SAM 2.1
  - CPU-only mode available but significantly slower

- **Processing Time**:
  - Depends on dataset size and hardware capabilities
  - Full Cityscapes dataset (~5000 images): 30-60 minutes with GPU

- **Dataset License**:
  - Ensure proper Cityscapes dataset license compliance
  - Academic use: Free with registration
  - Commercial use: Requires separate license

- **Model Weights**:
  - Download Mask2Former from [Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic)
  - Download SAM 2.1 from [Meta AI](https://github.com/facebookresearch/segment-anything-2)

### 📈 Use Cases

- **Autonomous Driving**: Road and sidewalk detection for navigation
- **Urban Planning**: Analyzing street infrastructure and vegetation coverage
- **3D Reconstruction**: Ground plane extraction for scene reconstruction
- **Robotics**: Traversable surface detection for mobile robots
- **Dataset Annotation**: Semi-automatic mask generation for training data

---

<a name="chinese"></a>
## 📋 项目概述

这是一个综合性的计算机视觉工具包，使用最先进的模型（Mask2Former、SAM 2.1）在Cityscapes数据集上进行城市场景理解和语义分割。本项目提供多种分割方法，用于从城市街景中提取地面、多类别物体和实例级掩码。

### 🎯 主要功能

- **地面提取**：使用Mask2Former自动提取道路和人行道的二值掩码
- **多类别分割**：提取道路、人行道、植被、地形和天空的掩码
- **实例分割**：使用SAM 2.1进行全物体自动分割
- **Cityscapes数据集支持**：处理标准Cityscapes数据集结构（训练/验证/测试集）
- **GPU加速**：支持CUDA加速推理，优化批处理性能
- **可视化**：生成带有颜色编码的对比图像，便于检查
- **多种输出格式**：为每个类别生成独立掩码和组合可视化
- **灵活配置**：易于修改路径和参数以适应不同设置

### 🏗️ 项目结构

```
VIPL/
├── datadets_cityscapes/        # Cityscapes数据集（不包含在仓库中）
│   ├── leftImg8bit/            # 输入图像
│   └── gtFine/                 # 真值标注
├── models/                     # 预训练模型（不包含在仓库中）
│   └── mask2former/            # Mask2Former模型文件
├── repos/                      # 外部仓库
│   ├── Mask2Former/            # Mask2Former实现
│   ├── detectron2/             # Detectron2框架
│   └── ground_parser/          # 地面分割脚本
└── outputs_masks/              # 生成的分割掩码（不包含在仓库中）
```

**注意**：大文件（数据集、模型、输出）不包含在仓库中。

### 🚀 快速开始

#### 环境要求

- Python 3.8+
- PyTorch 1.10+
- 支持CUDA的GPU（推荐）
- Git

#### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl
```

2. 安装依赖：
```bash
cd VIPL/repos/Mask2Former
pip install -r requirements.txt
```

3. 安装Detectron2：
```bash
cd ../detectron2
pip install -e .
```

4. 安装额外依赖：
```bash
pip install transformers opencv-python pillow tqdm
```

#### 数据集配置

1. 从[官方网站](https://www.cityscapes-dataset.com/)下载Cityscapes数据集
2. 解压到 `VIPL/datadets_cityscapes/`
3. 确保以下目录结构：

#### 模型配置

下载Cityscapes预训练的Mask2Former模型并放置在：
```
VIPL/models/mask2former/mask2former_cityscapes/
```

### 💻 使用方法

项目包含三个主要分割脚本：

#### 1. 地面分割（Mask2Former）

提取道路和人行道的二值掩码：

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_mask2former.py
```

**每张图像的输出文件：**
- `{base}_road.png` - 道路表面的二值掩码（255=道路，0=背景）
- `{base}_sidewalk.png` - 人行道表面的二值掩码
- `{base}_ground.png` - 组合地面掩码（道路+人行道）
- `{base}_contrast.png` - 可视化叠加图（红色=道路，蓝色=人行道）

**配置：**
```python
CITYSCAPES_ROOT = r"cityscapes数据集路径"
OUTPUT_ROOT = r"输出路径"
MODEL_PATH = r"mask2former_cityscapes模型路径"
```

#### 2. 多类别分割（Mask2Former）

提取多个城市场景类别的掩码：

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_multiclass_mask2former.py
```

**目标类别：**
- 道路（Road）
- 人行道（Sidewalk）
- 植被（Vegetation）- 树木、植物
- 地形（Terrain）- 地面
- 天空（Sky）

**每张图像的输出文件：**
- `{base}_road.png`
- `{base}_sidewalk.png`
- `{base}_vegetation.png`
- `{base}_terrain.png`
- `{base}_sky.png`
- `{base}_contrast.png` - 多色可视化

#### 3. 实例分割（SAM 2.1）

使用Segment Anything Model进行全自动物体分割：

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_sam.py
```

**特性：**
- 自动为所有物体生成掩码
- 优化参数以加快处理速度（每边16个采样点）
- 颜色编码的实例掩码
- 可配置的质量控制阈值

**配置：**
```python
CHECKPOINT_PATH = r"sam2.1_hiera_large.pt路径"
POINTS_PER_SIDE = 16  # 采样密度
PRED_IOU_THRESH = 0.86  # 质量阈值
MIN_MASK_REGION_AREA = 200  # 最小物体大小
```

**输出：**
- `{base}_sam_masks.png` - 颜色编码的实例分割
- 每个检测到的物体的独立掩码文件

### 📊 支持的类别

#### Mask2Former（Cityscapes）
分割模型针对以下Cityscapes类别：
- **道路（Road）**：主要道路表面
- **人行道（Sidewalk）**：行人步道和人行道
- **植被（Vegetation）**：树木、植物和绿化
- **地形（Terrain）**：自然地面
- **天空（Sky）**：天空区域

#### SAM 2.1（通用）
- 自动检测场景中的所有物体
- 实例级分割，无需预定义类别
- 适用于通用物体提取和分析

### 🧪 实验与结果

#### 性能指标

| 模型 | 速度（图/秒） | GPU显存 | 准确度 |
|------|-------------|---------|--------|
| Mask2Former（地面） | ~2-3 | 4-6 GB | 道路/人行道高精度 |
| Mask2Former（多类） | ~2-3 | 4-6 GB | 城市类别优秀 |
| SAM 2.1（优化） | ~0.5-1 | 8-12 GB | 通用物体检测 |

**测试硬件：** NVIDIA RTX 3080/4090

#### 输出示例

**地面分割：**
- 输入：城市街景（2048×1024）
- 输出：二值掩码 + 颜色叠加
- 处理时间：每张图像约0.3-0.5秒

**多类别分割：**
- 输入：城市街景
- 输出：5个独立类别掩码 + 可视化
- 处理时间：每张图像约0.3-0.5秒

**SAM实例分割：**
- 输入：城市街景
- 输出：每张图像50-200个实例掩码
- 处理时间：每张图像约1-2秒（优化后）

#### 优化建议

1. **批处理**：并行处理多张图像
2. **GPU显存**：使用混合精度（FP16）处理更大批次
3. **SAM速度**：将`POINTS_PER_SIDE`从32降至16可提速4倍
4. **质量与速度**：根据需求调整IOU阈值

### 🔧 配置说明

每个脚本在文件顶部都有可配置参数：

#### 地面分割脚本
```python
# run_cityscapes_ground_mask2former.py
CITYSCAPES_ROOT = r"E:\vipl\VIPL\datadets_cityscapes"
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks"
MODEL_PATH = r"E:\vipl\VIPL\models\mask2former\mask2former_cityscapes"

TARGET_LABELS = {
    "road": ["road"],
    "sidewalk": ["sidewalk", "side walk", "pavement"]
}
```

#### 多类别分割脚本
```python
# run_cityscapes_multiclass_mask2former.py
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_multiclass"

TARGET_LABELS = {
    "road": ["road"],
    "sidewalk": ["sidewalk", "side walk", "pavement"],
    "vegetation": ["vegetation", "tree", "plant"],
    "terrain": ["terrain", "ground"],
    "sky": ["sky"]
}
```

#### SAM实例分割脚本
```python
# run_cityscapes_ground_sam.py
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_sam"
CHECKPOINT_PATH = r"E:\vipl\VIPL\models\sam2\sam2.1_hiera_large.pt"

# 性能调优
POINTS_PER_SIDE = 16  # 16=快速，32=精确
PRED_IOU_THRESH = 0.86  # 越高=质量要求越严格
STABILITY_SCORE_THRESH = 0.92
MIN_MASK_REGION_AREA = 200  # 过滤小物体
```

### 📚 研究背景

本项目专注于城市场景理解中的地面分割，这对以下领域至关重要：
- 自动驾驶与导航
- 城市场景重建
- 机器人仿真环境
- 3D场景生成

### 🛠️ 使用技术

- **Mask2Former**：用于全景分割的通用图像分割架构
  - 论文：[Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
  - 在Cityscapes数据集上预训练（19个类别）

- **SAM 2.1（Segment Anything Model）**：通用实例分割
  - 论文：[Segment Anything](https://arxiv.org/abs/2304.02643)
  - 零样本物体分割能力

- **Detectron2**：Facebook AI Research的检测和分割平台
  - 模块化设计用于计算机视觉任务

- **PyTorch**：深度学习框架（1.10+）
- **Transformers**：Hugging Face模型库，用于Mask2Former
- **OpenCV**：计算机视觉操作和图像处理
- **Cityscapes数据集**：城市街景基准数据集（5000张精细标注）

### 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

### 🤝 贡献

欢迎贡献！请随时提交Pull Request。

### 📧 联系方式

如有问题或合作机会，请在GitHub上提交issue。

### 🙏 致谢

- [Cityscapes数据集](https://www.cityscapes-dataset.com/) 提供城市场景数据集
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) 提供分割架构
- [Detectron2](https://github.com/facebookresearch/detectron2) 提供检测框架

### ⚠️ 注意事项

- **大文件**：由于大小限制，模型文件和数据集不包含在本仓库中
  - Mask2Former模型：约200MB
  - SAM 2.1模型：约900MB
  - Cityscapes数据集：约11GB（leftImg8bit）+ 250MB（gtFine）

- **硬件要求**：
  - Mask2Former推荐使用8GB+显存的GPU
  - SAM 2.1推荐使用12GB+显存的GPU
  - 可使用CPU模式但速度显著较慢

- **处理时间**：
  - 取决于数据集大小和硬件性能
  - 完整Cityscapes数据集（约5000张图像）：使用GPU需30-60分钟

- **数据集许可**：
  - 请确保遵守Cityscapes数据集许可协议
  - 学术使用：注册后免费
  - 商业使用：需要单独许可

- **模型权重**：
  - 从[Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic)下载Mask2Former
  - 从[Meta AI](https://github.com/facebookresearch/segment-anything-2)下载SAM 2.1

### 📈 应用场景

- **自动驾驶**：用于导航的道路和人行道检测
- **城市规划**：分析街道基础设施和植被覆盖率
- **3D重建**：场景重建的地面平面提取
- **机器人**：移动机器人的可通行表面检测
- **数据集标注**：半自动生成训练数据的掩码
