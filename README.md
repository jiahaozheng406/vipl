# VIPL - Urban Scene Segmentation Project
# VIPL - 城市场景分割项目

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 📋 Overview

A computer vision project for urban scene understanding and ground surface segmentation using Mask2Former on the Cityscapes dataset.

### 🎯 Features

- **Ground Surface Extraction**: Automatically extracts binary masks for roads and sidewalks
- **Cityscapes Dataset Support**: Processes standard Cityscapes dataset structure
- **GPU Acceleration**: CUDA support for faster inference
- **Batch Processing**: Handles multiple images across train/val/test splits
- **Visualization**: Generates contrast images with color-coded segmentation overlays
- **Multiple Output Formats**: Produces separate masks for road, sidewalk, and combined ground surfaces

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

#### Ground Mask Extraction

Run the ground segmentation script:

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_mask2former.py
```

The script will:
- Load the Mask2Former model
- Process all images in the Cityscapes dataset
- Generate masks for roads, sidewalks, and combined ground surfaces
- Save visualization images with color-coded overlays

#### Output Files

For each input image, the following outputs are generated:
- `{base}_road.png` - Binary mask for road surfaces
- `{base}_sidewalk.png` - Binary mask for sidewalk surfaces
- `{base}_ground.png` - Combined ground surface mask
- `{base}_contrast.png` - Visualization overlay (red=road, blue=sidewalk)

### 📊 Supported Classes

The segmentation model targets the following Cityscapes classes:
- **Road**: Main road surfaces
- **Sidewalk**: Pedestrian walkways and pavements

### 🔧 Configuration

Edit the configuration section in `run_cityscapes_ground_mask2former.py`:

```python
CITYSCAPES_ROOT = r"path/to/cityscapes"
OUTPUT_ROOT = r"path/to/output"
MODEL_PATH = r"path/to/model"
```

### 📚 Research Context

This project focuses on ground surface segmentation for urban scene understanding, which is fundamental for:
- Autonomous driving and navigation
- Urban scene reconstruction
- Robotics simulation environments
- 3D scene generation

### 🛠️ Technologies Used

- **Mask2Former**: Universal image segmentation architecture
- **Detectron2**: Facebook AI Research's detection and segmentation platform
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **OpenCV**: Computer vision operations
- **Cityscapes Dataset**: Urban street scene benchmark

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

- Large model files and datasets are not included in this repository
- GPU with sufficient VRAM (8GB+) recommended for processing
- Processing time depends on dataset size and hardware capabilities
- Ensure proper Cityscapes dataset license compliance for academic/commercial use

---

<a name="chinese"></a>
## 📋 项目概述

这是一个基于Mask2Former模型在Cityscapes数据集上进行城市场景理解和地面分割的计算机视觉项目。

### 🎯 主要功能

- **地面提取**：自动提取道路和人行道的二值掩码
- **Cityscapes数据集支持**：处理标准Cityscapes数据集结构
- **GPU加速**：支持CUDA加速推理
- **批量处理**：处理训练集/验证集/测试集的多张图像
- **可视化**：生成带有颜色编码的分割叠加对比图
- **多种输出格式**：生成道路、人行道和组合地面的独立掩码

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

#### 地面掩码提取

运行地面分割脚本：

```bash
cd VIPL/repos/ground_parser
python run_cityscapes_ground_mask2former.py
```

脚本将会：
- 加载Mask2Former模型
- 处理Cityscapes数据集中的所有图像
- 生成道路、人行道和组合地面的掩码
- 保存带有颜色编码叠加的可视化图像

#### 输出文件

对于每张输入图像，将生成以下输出：
- `{base}_road.png` - 道路表面的二值掩码
- `{base}_sidewalk.png` - 人行道表面的二值掩码
- `{base}_ground.png` - 组合地面掩码
- `{base}_contrast.png` - 可视化叠加图（红色=道路，蓝色=人行道）

### 📊 支持的类别

分割模型针对以下Cityscapes类别：
- **道路（Road）**：主要道路表面
- **人行道（Sidewalk）**：行人步道和人行道

### 🔧 配置说明

编辑 `run_cityscapes_ground_mask2former.py` 中的配置部分：

```python
CITYSCAPES_ROOT = r"cityscapes数据集路径"
OUTPUT_ROOT = r"输出路径"
MODEL_PATH = r"模型路径"
```

### 📚 研究背景

本项目专注于城市场景理解中的地面分割，这对以下领域至关重要：
- 自动驾驶与导航
- 城市场景重建
- 机器人仿真环境
- 3D场景生成

### 🛠️ 使用技术

- **Mask2Former**：通用图像分割架构
- **Detectron2**：Facebook AI Research的检测和分割平台
- **PyTorch**：深度学习框架
- **Transformers**：Hugging Face模型库
- **OpenCV**：计算机视觉操作
- **Cityscapes数据集**：城市街景基准数据集

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

- 本仓库不包含大型模型文件和数据集
- 建议使用具有足够显存（8GB+）的GPU进行处理
- 处理时间取决于数据集大小和硬件性能
- 请确保遵守Cityscapes数据集的学术/商业使用许可
