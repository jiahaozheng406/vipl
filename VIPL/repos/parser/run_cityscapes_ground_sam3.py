#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cityscapes 语义分割 using SAM 3
使用 SAM 3 文本提示分割 6 类: road, sidewalk, building, vegetation, terrain, sky
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
OUTPUT_ROOT     = r"E:\vipl\VIPL\outputs_masks_sam3"
MODEL_DIR       = r"E:\vipl\VIPL\models\sam3"

# 要提取的类别及文本提示
CATEGORIES = {
    "road":       "road",
    "sidewalk":   "sidewalk",
    "building":   "building",
    "vegetation": "vegetation",
    "terrain":    "terrain",
    "sky":        "sky",
}

# SAM3 推理阈值
SCORE_THRESH = 0.5
MASK_THRESH  = 0.5

# ==================== 初始化 ====================
print("=" * 60)
print("Cityscapes 语义分割 using SAM 3")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Device] GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\n[Output] 输出目录: {OUTPUT_ROOT}")

if not os.path.exists(MODEL_DIR):
    print(f"\n[ERROR] 模型目录不存在: {MODEL_DIR}")
    sys.exit(1)

# ==================== 加载 SAM 3 模型 ====================
print(f"\n[Model] 加载 SAM 3 模型: {MODEL_DIR}")

try:
    from transformers import Sam3Processor, Sam3Model

    processor = Sam3Processor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = Sam3Model.from_pretrained(MODEL_DIR, local_files_only=True)
    model = model.to(device)
    model.eval()
    print("[Model] SAM 3 加载成功")

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

# ==================== 推理函数 ====================
def process_image(img_path):
    """
    对单张图像逐类别用文本提示推理，返回各类 binary mask (uint8, 0/255)
    """
    try:
        image = Image.open(str(img_path)).convert("RGB")
        h, w = image.size[1], image.size[0]

        result_masks = {}

        for cat_name, prompt in CATEGORIES.items():
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=SCORE_THRESH,
                mask_threshold=MASK_THRESH,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            # 合并所有匹配实例为一张 binary mask
            combined = np.zeros((h, w), dtype=np.uint8)
            for mask_tensor in results["masks"]:
                m = mask_tensor.cpu().numpy().astype(np.uint8)
                combined = np.maximum(combined, m)

            result_masks[cat_name] = combined * 255

        return result_masks

    except Exception as e:
        print(f"\n[ERROR] 处理失败 {img_path.name}: {e}")
        return None


# ==================== 颜色叠加可视化 ====================
CATEGORY_COLORS = {
    "road":       (128,  64, 128),   # 紫红
    "sidewalk":   (244,  35, 232),   # 品红
    "building":   ( 70,  70,  70),   # 深灰
    "vegetation": (107, 142,  35),   # 橄榄绿
    "terrain":    (152, 251, 152),   # 浅绿
    "sky":        ( 70, 130, 180),   # 钢蓝
}

def make_overlay(image_rgb, masks):
    """将所有类别叠加到原图上生成彩色可视化"""
    overlay = image_rgb.copy().astype(np.float32)
    for cat_name, mask in masks.items():
        if mask.max() == 0:
            continue
        color = np.array(CATEGORY_COLORS[cat_name], dtype=np.float32)
        region = mask > 0
        overlay[region] = overlay[region] * 0.5 + color * 0.5
    return overlay.astype(np.uint8)


# ==================== 批量处理 ====================
print("\n" + "=" * 60)
print("开始处理图像...")
print("=" * 60)

processed_count = 0
failed_count = 0

for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
    masks = process_image(img_path)

    if masks is None:
        failed_count += 1
        continue

    # 构建输出路径，保持 split/city 目录结构
    relative_path = img_path.relative_to(Path(CITYSCAPES_ROOT) / "leftImg8bit")
    base_name = img_path.stem.replace("_leftImg8bit", "")
    output_dir = os.path.join(OUTPUT_ROOT, str(relative_path.parent))
    os.makedirs(output_dir, exist_ok=True)

    # 保存各类 binary mask
    for cat_name, mask in masks.items():
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_{cat_name}.png"),
            mask
        )

    # 保存彩色叠加可视化
    image_rgb = np.array(Image.open(str(img_path)).convert("RGB"))
    overlay = make_overlay(image_rgb, masks)
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    processed_count += 1

    if (idx + 1) % 10 == 0:
        print(f"\n[Progress] {idx + 1}/{len(image_paths)} (成功: {processed_count}, 失败: {failed_count})")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n[Output] 保存目录: {OUTPUT_ROOT}")
print(f"[Stats]  总共: {len(image_paths)} 张")
print(f"[Stats]  成功: {processed_count} 张")
print(f"[Stats]  失败: {failed_count} 张")
print("\n输出文件 (每张图像):")
for cat in CATEGORIES:
    print(f"  {{base}}_{cat}.png       - {cat} binary mask (0/255)")
print(f"  {{base}}_overlay.png    - 6类彩色叠加可视化")
