# VIPL - Urban Scene Perception & Simulation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## Overview

An end-to-end pipeline that takes urban street images and produces a composed 3D simulation scene:

1. **Semantic Parsing** — Mask2Former / SAM3 segment images into road, building, car, tree, pole, grass, etc.
2. **3D Reconstruction** — VGGT estimates depth & point clouds; SAM3 provides per-instance masks
3. **OBB Extraction** — DBSCAN clusters + Open3D oriented bounding boxes for each object instance
4. **Asset Matching** — Match each OBB to the best UrbanVerse GLB asset by shape ratio + text query
5. **Scene Composition** — Place all matched GLB assets at calibrated OBB poses → output composed 3D scene

### Project Structure

```
VIPL/
├── datadets_cityscapes/              # Datasets (not included)
│   ├── leftImg8bit/                  # Cityscapes images
│   ├── njuptVolvo/                   # Custom street-view images
│   └── datasets_VKITTI2/            # Virtual KITTI 2.0
├── models/                           # Pre-trained weights (not included)
│   ├── mask2former/, sam2/, sam3/, vggt/
├── repos/
│   ├── parser/                       # Stage 1: Semantic segmentation scripts
│   │   ├── run_cityscapes_ground_mask2former.py
│   │   ├── run_cityscapes_multiclass_mask2former.py
│   │   ├── run_cityscapes_building_mask2former.py
│   │   ├── run_cityscapes_ground_sam2.py
│   │   ├── run_cityscapes_ground_sam3.py
│   │   └── run_video_car_sam3.py
│   ├── box/                          # Stage 2-3: 3D reconstruction + OBB
│   │   ├── core/                     # Shared utilities
│   │   │   ├── vggt_infer.py         # VGGT depth/pointcloud inference
│   │   │   ├── sam3_infer.py         # SAM3 instance segmentation
│   │   │   ├── box_utils.py          # OBB fitting (DBSCAN + Open3D)
│   │   │   ├── geometry_utils.py     # 3D geometry helpers
│   │   │   ├── pointcloud_utils.py   # Point cloud I/O
│   │   │   ├── image_utils.py        # Image processing
│   │   │   ├── io_utils.py           # File I/O
│   │   │   └── video_utils.py        # Video frame extraction
│   │   ├── SAM_VGGT_box_njuptvolvo.py  # Full pipeline: images → OBB
│   │   └── sam_vggt_box_car.py         # Car-specific pipeline
│   ├── asset/                        # Stage 4: UrbanVerse asset matching
│   │   └── box_Ubranverse_njuptvolvo_asset.py
│   ├── all2asset/                    # Stage 5: End-to-end orchestration
│   │   ├── all2asset_njupyvolvoscene.py    # Full pipeline runner
│   │   └── compose_scene_njuptvolvo.py     # Scene composition + preview
│   ├── detectron2/                   # (external, not included)
│   └── Mask2Former/                  # (external, not included)
└── outputs (generated, not included)
```

### Pipeline Flow

```
Input Images (6 views)
    │
    ▼
[parser] Mask2Former / SAM3 → semantic masks (road, car, tree, building, ...)
    │
    ▼
[box] VGGT → depth + point cloud → SAM3 instance masks → DBSCAN → OBB per instance
    │
    ▼
[asset] OBB shape ratio + text query → UrbanVerse API → best GLB asset per object
    │
    ▼
[all2asset] Orchestrate full pipeline + global scale calibration
    │
    ▼
[compose_scene] Place GLB assets at OBB poses → scene_composed.glb + preview PNG
```

### Quick Start

#### Prerequisites
- Python 3.10+, PyTorch 2.0+, CUDA GPU (12 GB+ VRAM recommended)
- Key packages: `trimesh`, `open3d`, `transformers`, `opencv-python`, `scipy`, `scikit-learn`

#### Installation

```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install trimesh open3d transformers opencv-python scipy scikit-learn pillow tqdm requests
```

#### Run Full Pipeline

```bash
# Stage 1-4: Segmentation → 3D → OBB → Asset matching
cd VIPL/repos/all2asset
python all2asset_njupyvolvoscene.py

# Stage 5: Compose all assets into one 3D scene
python compose_scene_njuptvolvo.py
```

#### Outputs

```
outputs/njuptVolvo/
├── car/                    # Per-instance: pointcloud PLY + OBB JSON + asset JSON
├── tree/
├── pole/
├── road/
├── grass/
├── building/
└── scene/
    ├── scene_composed.glb  # Full 3D scene (open in Blender / online GLB viewer)
    └── scene_composed.png  # Multi-view preview
```

### Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Semantic Segmentation | Mask2Former, SAM 2.1, SAM 3 | Multi-class urban parsing |
| 3D Reconstruction | VGGT | Monocular depth → point cloud |
| Instance Clustering | DBSCAN + Open3D OBB | Per-object bounding boxes |
| Asset Matching | UrbanVerse API | Shape-ratio + text similarity retrieval |
| Scene Composition | trimesh | GLB placement with calibrated OBB poses |

### Notes

- Model weights and datasets are **not included** (~40 GB total)
- UrbanVerse asset matching requires internet access for initial GLB download
- Scene composition uses global scale calibration (VGGT → metric) via a reference car OBB

### License
MIT License — see [LICENSE](LICENSE)

### Acknowledgments
[Cityscapes](https://www.cityscapes-dataset.com/) · [Mask2Former](https://github.com/facebookresearch/Mask2Former) · [SAM](https://github.com/facebookresearch/segment-anything-2) · [VGGT](https://github.com/facebookresearch/vggt) · [UrbanVerse](https://urban-verse.github.io/) · [Detectron2](https://github.com/facebookresearch/detectron2) · [Open3D](http://www.open3d.org/)

---

<a name="chinese"></a>
## 项目概述

端到端城市场景感知与仿真 Pipeline：从街景图片到组装好的 3D 仿真场景。

1. **语义分割** — Mask2Former / SAM3 将图像分割为道路、建筑、车辆、树木、电线杆、草地等
2. **3D 重建** — VGGT 估计深度和点云；SAM3 提供逐实例掩码
3. **OBB 提取** — DBSCAN 聚类 + Open3D 有向包围盒
4. **资产匹配** — 按形状比例 + 文本查询匹配 UrbanVerse GLB 资产
5. **场景组装** — 将所有 GLB 资产按标定后的 OBB 位姿放置 → 输出完整 3D 场景

### 快速开始

```bash
git clone https://github.com/jiahaozheng406/vipl.git
cd vipl

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install trimesh open3d transformers opencv-python scipy scikit-learn pillow tqdm requests

# 运行完整 Pipeline
cd VIPL/repos/all2asset
python all2asset_njupyvolvoscene.py      # 分割 → 3D → OBB → 资产匹配
python compose_scene_njuptvolvo.py       # 组装场景 + 预览图
```

### 注意事项

- 模型权重和数据集**不包含**在仓库中（总计约 40 GB）
- UrbanVerse 资产匹配首次运行需要联网下载 GLB
- 场景组装使用全局尺度标定（以参考车辆 OBB 反推 VGGT→米制缩放因子）

### 许可证
MIT 许可证 — 详见 [LICENSE](LICENSE)

### 致谢
[Cityscapes](https://www.cityscapes-dataset.com/) · [Mask2Former](https://github.com/facebookresearch/Mask2Former) · [SAM](https://github.com/facebookresearch/segment-anything-2) · [VGGT](https://github.com/facebookresearch/vggt) · [UrbanVerse](https://urban-verse.github.io/) · [Detectron2](https://github.com/facebookresearch/detectron2) · [Open3D](http://www.open3d.org/)
</content>
</invoke>