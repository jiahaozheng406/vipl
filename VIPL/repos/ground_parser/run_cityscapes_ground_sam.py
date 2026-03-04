#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cityscapes 全物体分割 using SAM 2.1
使用 SAM 2.1 进行全物体分割，所有物体用不同颜色标记
优化版：使用更快的参数和批处理
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
CITYSCAPES_ROOT = r"E:\vipl\VIPL\datadets_cityscapes"
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_sam"

# SAM 2.1 模型配置
CHECKPOINT_PATH = r"E:\vipl\VIPL\models\sam2\sam2.1_hiera_large.pt"

# 优化参数：减少采样点以提高速度
POINTS_PER_SIDE = 16  # 从 32 降到 16，速度提升 4 倍
PRED_IOU_THRESH = 0.86  # 略微降低阈值
STABILITY_SCORE_THRESH = 0.92  # 略微降低阈值
MIN_MASK_REGION_AREA = 200  # 增加最小区域面积，过滤小物体

# ==================== 初始化 ====================
print("=" * 60)
print("Cityscapes 全物体分割 using SAM 2.1 (优化版)")
print("=" * 60)

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Device] GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\n[Output] 输出目录: {OUTPUT_ROOT}")

# 检查模型文件
if not os.path.exists(CHECKPOINT_PATH):
    print(f"\n[ERROR] 模型文件不存在: {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"\n[Model] 模型文件: {CHECKPOINT_PATH}")
print(f"[Model] 模型大小: {os.path.getsize(CHECKPOINT_PATH) / 1024**3:.2f} GB")

# ==================== 加载 SAM 2.1 模型 ====================
print(f"\n[Model] 加载 SAM 2.1 模型...")

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import sam2

    # 加载配置
    sam2_dir = os.path.dirname(sam2.__file__)
    config_file = os.path.join(sam2_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml")

    if os.path.exists(config_file):
        print(f"[Model] 找到配置文件: {config_file}")
        cfg = OmegaConf.load(config_file)

        # 加载默认配置
        if 'defaults' in cfg:
            for default in cfg.defaults:
                if isinstance(default, dict):
                    for key, value in default.items():
                        default_config_path = os.path.join(sam2_dir, "configs", key, f"{value}.yaml")
                        if os.path.exists(default_config_path):
                            default_cfg = OmegaConf.load(default_config_path)
                            cfg = OmegaConf.merge(default_cfg, cfg)

        # 构建模型
        sam2_model = instantiate(cfg.model, _recursive_=True)

        # 加载权重
        print(f"[Model] 加载权重...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        sam2_model.load_state_dict(state_dict, strict=True)
        sam2_model = sam2_model.to(device)
        sam2_model.eval()

        # 使用 FP16 加速（如果是 GPU）
        if device.type == 'cuda':
            sam2_model = sam2_model.half()
            print("[Model] 使用 FP16 精度加速")

        print("[Model] SAM 2.1 模型加载成功")
    else:
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    # 创建自动 mask 生成器（优化参数）
    print(f"[Model] 创建自动 mask 生成器（优化参数）...")
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=0,  # 禁用裁剪以提高速度
        crop_n_points_downscale_factor=1,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )

    print("[Model] Mask 生成器创建成功")
    print(f"[Optimization] 采样点数: {POINTS_PER_SIDE}x{POINTS_PER_SIDE} = {POINTS_PER_SIDE**2}")

except Exception as e:
    print(f"[ERROR] 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 收集图像 ====================
print(f"\n[Dataset] 扫描数据集: {CITYSCAPES_ROOT}")
image_paths = []
for split in ["train", "val", "test"]:
    split_dir = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", split)
    if os.path.exists(split_dir):
        for city_dir in Path(split_dir).iterdir():
            if city_dir.is_dir():
                for img_path in city_dir.glob("*_leftImg8bit.png"):
                    image_paths.append(img_path)

print(f"[Dataset] 找到 {len(image_paths)} 张图像")

if len(image_paths) == 0:
    print("[ERROR] 未找到任何图像！")
    sys.exit(1)

# ==================== 颜色生成器 ====================
def generate_colors(n):
    """生成 n 个不同的颜色"""
    np.random.seed(42)
    colors = []
    for i in range(n):
        import colorsys
        hue = (i * 137.508) % 360
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors

# ==================== 地面区域分类器 ====================
def classify_ground_mask(mask, image):
    """判断 mask 是否为地面区域"""
    h, w = mask.shape
    y_coords, x_coords = np.where(mask > 0)

    if len(y_coords) == 0:
        return False, None

    avg_y = np.mean(y_coords) / h

    if avg_y < 0.3:
        return False, None

    bbox_h = y_coords.max() - y_coords.min() + 1
    bbox_w = x_coords.max() - x_coords.min() + 1
    aspect_ratio = bbox_w / max(bbox_h, 1)
    area_ratio = np.sum(mask > 0) / (h * w)

    if area_ratio < 0.01:
        return False, None

    masked_pixels = image[mask > 0]
    if len(masked_pixels) == 0:
        return False, None

    avg_color = np.mean(masked_pixels, axis=0)
    brightness = np.mean(avg_color)

    is_ground = (avg_y > 0.4 and aspect_ratio > 1.5 and area_ratio > 0.05)

    if not is_ground:
        return False, None

    x_center = np.mean(x_coords) / w
    edge_score = abs(x_center - 0.5) * 2

    if brightness > 100 and (edge_score > 0.3 or area_ratio < 0.15):
        return True, 'sidewalk'
    else:
        return True, 'road'

# ==================== 推理函数 ====================
def process_image(img_path):
    """处理单张图像"""
    try:
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            return None, None, None, None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 SAM 2.1 生成所有 masks
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            masks = mask_generator.generate(image_rgb)

        # 按面积排序
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # 创建输出图像
        h, w = image.shape[:2]

        # 1. 全物体彩色分割图
        all_objects_colored = image_rgb.copy()
        colors = generate_colors(len(masks))

        for idx, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = colors[idx % len(colors)]
            all_objects_colored[mask] = all_objects_colored[mask] * 0.5 + np.array(color) * 0.5

        # 2. 地面分割
        road_mask = np.zeros((h, w), dtype=np.uint8)
        sidewalk_mask = np.zeros((h, w), dtype=np.uint8)

        for mask_data in masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            is_ground, mask_type = classify_ground_mask(mask, image_rgb)

            if is_ground:
                if mask_type == 'road':
                    road_mask = np.maximum(road_mask, mask)
                elif mask_type == 'sidewalk':
                    sidewalk_mask = np.maximum(sidewalk_mask, mask)

        ground_mask = np.maximum(road_mask, sidewalk_mask)

        # 3. 地面对比图
        ground_contrast = image_rgb.copy()
        ground_contrast[road_mask > 0] = ground_contrast[road_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        ground_contrast[sidewalk_mask > 0] = ground_contrast[sidewalk_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5

        return road_mask, sidewalk_mask, ground_mask, ground_contrast, all_objects_colored

    except Exception as e:
        print(f"\n[ERROR] 处理失败 {img_path.name}: {e}")
        return None, None, None, None, None

# ==================== 批量处理 ====================
print("\n" + "=" * 60)
print("开始处理图像...")
print("=" * 60)

processed_count = 0
failed_count = 0

for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
    # 处理图像
    road_mask, sidewalk_mask, ground_mask, ground_contrast, all_objects_colored = process_image(img_path)

    if road_mask is None:
        failed_count += 1
        continue

    # 构建输出路径
    relative_path = img_path.relative_to(Path(CITYSCAPES_ROOT) / "leftImg8bit")
    base_name = img_path.stem.replace("_leftImg8bit", "")

    output_dir = os.path.join(OUTPUT_ROOT, relative_path.parent)
    os.makedirs(output_dir, exist_ok=True)

    # 保存地面 masks
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_road.png"), road_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_sidewalk.png"), sidewalk_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_ground.png"), ground_mask)

    # 保存地面对比图
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_ground_contrast.png"),
        cv2.cvtColor(ground_contrast.astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    # 保存全物体彩色分割图
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_all_objects.png"),
        cv2.cvtColor(all_objects_colored.astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    processed_count += 1

    # 每 10 张打印进度
    if (idx + 1) % 10 == 0:
        print(f"\n[Progress] 已处理 {idx + 1}/{len(image_paths)} (成功: {processed_count}, 失败: {failed_count})")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n[Output] Mask 保存在: {OUTPUT_ROOT}")
print(f"\n[Stats] 总共: {len(image_paths)} 张")
print(f"[Stats] 成功: {processed_count} 张")
print(f"[Stats] 失败: {failed_count} 张")
print("\n输出文件:")
print("  {base}_road.png - Road mask")
print("  {base}_sidewalk.png - Sidewalk mask")
print("  {base}_ground.png - Ground mask")
print("  {base}_ground_contrast.png - Ground 对比图（红=road，蓝=sidewalk）")
print("  {base}_all_objects.png - 全物体彩色分割图（天空、建筑、车辆等所有物体）")
