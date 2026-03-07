#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cityscapes Building Mask Extraction using Mask2Former
提取 building 的二值 mask
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
OUTPUT_ROOT = r"E:\vipl\VIPL\outputs_masks_building"

# 本地模型路径
MODEL_PATH = r"E:\vipl\VIPL\models\mask2former\mask2former_cityscapes"

# 目标类别
TARGET_LABELS = {
    "building": ["building", "buildings"]
}

# ==================== 初始化 ====================
print("=" * 60)
print("Cityscapes Building Mask Extraction using Mask2Former")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\n[Output] 输出目录: {OUTPUT_ROOT}")

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

# 查找 building 类别 ID
id2label = model.config.id2label

def find_label_ids(id2label, target_names):
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

print(f"\n[Labels] 查找 building 类别:")
building_ids = find_label_ids(id2label, TARGET_LABELS["building"])
print(f"[Labels] Building IDs: {building_ids}")

if not building_ids:
    print("[ERROR] 未找到 building 类别！")
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
    try:
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]

        panoptic_seg = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        h, w = panoptic_seg.shape
        building_mask = np.zeros((h, w), dtype=np.uint8)

        for segment in segments_info:
            if segment["label_id"] in building_ids:
                building_mask[panoptic_seg == segment["id"]] = 255

        return building_mask, np.array(image)

    except Exception as e:
        print(f"[ERROR] 处理失败 {img_path.name}: {e}")
        return None, None

# ==================== 批量处理 ====================
print(f"\n[Processing] 开始处理...")

processed_count = 0
failed_count = 0

for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
    building_mask, original_img = process_image(img_path)

    if building_mask is None:
        failed_count += 1
        continue

    relative_path = img_path.relative_to(Path(CITYSCAPES_ROOT) / "leftImg8bit")
    base_name = img_path.stem.replace("_leftImg8bit", "")
    output_dir = os.path.join(OUTPUT_ROOT, str(relative_path.parent))
    os.makedirs(output_dir, exist_ok=True)

    # 保存 building binary mask
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_building.png"), building_mask)

    # 保存对比图（蓝色=building）
    contrast_img = original_img.copy().astype(np.float32)
    contrast_img[building_mask > 0] = contrast_img[building_mask > 0] * 0.5 + np.array([70, 70, 70]) * 0.5
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_building_contrast.png"),
        cv2.cvtColor(contrast_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    processed_count += 1

    if (idx + 1) % 50 == 0:
        print(f"\n[Progress] 已处理 {idx + 1}/{len(image_paths)} (成功: {processed_count}, 失败: {failed_count})")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n[Output] Mask 保存在: {OUTPUT_ROOT}")
print(f"[Stats]  总共: {len(image_paths)} 张")
print(f"[Stats]  成功: {processed_count} 张")
print(f"[Stats]  失败: {failed_count} 张")
print("\n输出文件:")
print("  {split}/{city}/{base}_building.png          - building binary mask (0/255)")
print("  {split}/{city}/{base}_building_contrast.png - 对比图")
