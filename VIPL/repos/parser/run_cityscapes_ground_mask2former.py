#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cityscapes Ground Mask Extraction using Mask2Former
提取 road 和 sidewalk 的二值 mask
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm

# ==================== 配置 ====================
CITYSCAPES_ROOT = r"E:\vipl\VIPL\datadets_cityscapes"
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks"

# 本地模型路径
MODEL_PATH = r"E:\vipl\VIPL\models\mask2former\mask2former_cityscapes"

# 目标类别（Cityscapes 标准）
TARGET_LABELS = {
    "road": ["road"],
    "sidewalk": ["sidewalk", "side walk", "pavement"]
}

# ==================== 初始化 ====================
print("=" * 60)
print("Cityscapes Ground Mask Extraction")
print("=" * 60)

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\n[Output] 输出目录: {OUTPUT_ROOT}")

# 加载模型（使用本地路径）
print(f"\n[Model] 加载本地模型: {MODEL_PATH}")

try:
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_PATH)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("[Model] 模型加载成功")
except Exception as e:
    print(f"[ERROR] 模型加载失败: {e}")
    sys.exit(1)

# 打印 label 映射
print(f"\n[Labels] 模型标签映射:")
id2label = model.config.id2label
for idx, label in id2label.items():
    print(f"  {idx}: {label}")

# 查找目标类别 ID
def find_label_ids(id2label, target_names):
    """查找目标类别的 ID（容错匹配）"""
    ids = []
    for idx, label in id2label.items():
        label_lower = label.lower().replace(" ", "").replace("-", "")
        for target in target_names:
            target_lower = target.lower().replace(" ", "").replace("-", "")
            if target_lower in label_lower or label_lower in target_lower:
                ids.append(idx)
                print(f"  匹配: {label} (ID={idx}) -> {target}")
                break
    return ids

road_ids = find_label_ids(id2label, TARGET_LABELS["road"])
sidewalk_ids = find_label_ids(id2label, TARGET_LABELS["sidewalk"])

print(f"\n[Labels] Road IDs: {road_ids}")
print(f"[Labels] Sidewalk IDs: {sidewalk_ids}")

if not road_ids and not sidewalk_ids:
    print("[ERROR] 未找到 road 或 sidewalk 类别！")
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

# ==================== 推理函数 ====================
def process_image(img_path):
    """处理单张图像"""
    try:
        # 读取图像
        image = Image.open(img_path).convert("RGB")

        # 推理
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 后处理得到 panoptic segmentation
        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]

        # 获取分割图
        panoptic_seg = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        # 创建 mask
        h, w = panoptic_seg.shape
        road_mask = np.zeros((h, w), dtype=np.uint8)
        sidewalk_mask = np.zeros((h, w), dtype=np.uint8)

        # 遍历所有分割区域
        for segment in segments_info:
            segment_id = segment["id"]
            label_id = segment["label_id"]

            # 提取该区域的 mask
            segment_mask = (panoptic_seg == segment_id)

            # 分配到对应类别
            if label_id in road_ids:
                road_mask[segment_mask] = 255
            elif label_id in sidewalk_ids:
                sidewalk_mask[segment_mask] = 255

        # 合并 ground mask
        ground_mask = np.maximum(road_mask, sidewalk_mask)

        return road_mask, sidewalk_mask, ground_mask, np.array(image)

    except Exception as e:
        print(f"[ERROR] 处理失败 {img_path.name}: {e}")
        return None, None, None, None

# ==================== 批量处理 ====================
print(f"\n[Processing] 开始处理...")

for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
    # 处理图像
    road_mask, sidewalk_mask, ground_mask, original_img = process_image(img_path)

    if road_mask is None:
        continue

    # 构建输出路径
    relative_path = img_path.relative_to(Path(CITYSCAPES_ROOT) / "leftImg8bit")
    base_name = img_path.stem.replace("_leftImg8bit", "")

    output_dir = os.path.join(OUTPUT_ROOT, relative_path.parent)
    os.makedirs(output_dir, exist_ok=True)

    # 保存 mask
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_road.png"), road_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_sidewalk.png"), sidewalk_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_ground.png"), ground_mask)

    # 保存对比图（红色=road，蓝色=sidewalk）
    contrast_img = original_img.copy()
    contrast_img[road_mask > 0] = contrast_img[road_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    contrast_img[sidewalk_mask > 0] = contrast_img[sidewalk_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    contrast_path = os.path.join(output_dir, f"{base_name}_contrast.png")
    cv2.imwrite(contrast_path, cv2.cvtColor(contrast_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 每 50 张打印进度
    if (idx + 1) % 50 == 0:
        print(f"\n[Progress] 已处理 {idx + 1}/{len(image_paths)} 张图像")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n[Output] Mask 保存在: {OUTPUT_ROOT}")
print(f"\n[Stats] 总共处理: {len(image_paths)} 张图像")
print("\n输出结构:")
print("  {split}/{city}/{base}_road.png")
print("  {split}/{city}/{base}_sidewalk.png")
print("  {split}/{city}/{base}_ground.png")
print("  {split}/{city}/{base}_contrast.png")
