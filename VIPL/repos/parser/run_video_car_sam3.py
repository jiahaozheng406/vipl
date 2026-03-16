#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取指定视频，使用 SAM 3 只分割 car，并将结果保存到指定文件夹
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
VIDEO_PATH  = r"E:\vipl\VIPL\datadets_cityscapes\Open_video_page.mp4"
OUTPUT_ROOT = r"E:\vipl\VIPL\repos\parser\outputs_masks_sam3_car"
MODEL_DIR   = r"E:\vipl\VIPL\models\sam3"

# 只分割 car
TARGET_NAME   = "car"
TEXT_PROMPT   = "car"

# 每隔多少帧处理一次
FRAME_INTERVAL = 10  # 1表示每帧都处理；如果太慢可改成 2 / 5 / 10

# SAM3 推理阈值
SCORE_THRESH = 0.1
MASK_THRESH  = 0.4

# 是否保存叠加可视化
SAVE_OVERLAY = True

# car 可视化颜色（BGR/RGB都可，下面按 RGB 用）
TARGET_COLOR = (255, 0, 0)  # 红色


# ==================== 初始化 ====================
print("=" * 60)
print("SAM 3 视频分割：car")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Device] GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

if not os.path.exists(VIDEO_PATH):
    print(f"\n[ERROR] 视频不存在: {VIDEO_PATH}")
    sys.exit(1)

if not os.path.exists(MODEL_DIR):
    print(f"\n[ERROR] 模型目录不存在: {MODEL_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"\n[Input ] 视频路径: {VIDEO_PATH}")
print(f"[Output] 输出目录: {OUTPUT_ROOT}")


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


# ==================== 推理函数 ====================
def process_pil_image(image_pil):
    """
    对单张 PIL 图像使用文本提示 'car' 推理
    返回:
        mask_uint8: uint8 二值图, 0/255
    """
    try:
        w, h = image_pil.size

        inputs = processor(
            images=image_pil,
            text=TEXT_PROMPT,
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

        combined = np.zeros((h, w), dtype=np.uint8)

        if "masks" in results and results["masks"] is not None:
            for mask_tensor in results["masks"]:
                m = mask_tensor.detach().cpu().numpy().astype(np.uint8)
                combined = np.maximum(combined, m)

        return combined * 255

    except Exception as e:
        print(f"[ERROR] 单帧处理失败: {e}")
        return None


def make_overlay(image_rgb, mask_uint8):
    """
    将 car mask 叠加到原图
    image_rgb: HWC RGB
    mask_uint8: HxW, 0/255
    """
    overlay = image_rgb.copy().astype(np.float32)
    region = mask_uint8 > 0
    color = np.array(TARGET_COLOR, dtype=np.float32)
    overlay[region] = overlay[region] * 0.5 + color * 0.5
    return overlay.astype(np.uint8)


# ==================== 打开视频 ====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] 无法打开视频: {VIDEO_PATH}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"\n[Video ] 总帧数: {total_frames}")
print(f"[Video ] FPS: {fps:.2f}")
print(f"[Video ] 抽帧间隔: {FRAME_INTERVAL}")

# 为了显示处理进度，估算需要处理多少帧
estimated = (total_frames + FRAME_INTERVAL - 1) // FRAME_INTERVAL if total_frames > 0 else None


# ==================== 逐帧处理 ====================
print("\n" + "=" * 60)
print("开始处理视频...")
print("=" * 60)

frame_idx = 0
processed_count = 0
failed_count = 0

pbar = tqdm(total=estimated, desc="Processing frames") if estimated is not None else tqdm(desc="Processing frames")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL != 0:
        frame_idx += 1
        continue

    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        mask = process_pil_image(image_pil)

        if mask is None:
            failed_count += 1
            frame_idx += 1
            pbar.update(1)
            continue

        base_name = f"frame_{frame_idx:06d}"

        # 保存 car mask
        mask_path = os.path.join(OUTPUT_ROOT, f"{base_name}_{TARGET_NAME}.png")
        cv2.imwrite(mask_path, mask)

        # 保存 overlay
        if SAVE_OVERLAY:
            overlay_rgb = make_overlay(frame_rgb, mask)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            overlay_path = os.path.join(OUTPUT_ROOT, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay_bgr)

        processed_count += 1

    except Exception as e:
        print(f"\n[ERROR] 第 {frame_idx} 帧处理失败: {e}")
        failed_count += 1

    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()


# ==================== 完成 ====================
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n[Output] 保存目录: {OUTPUT_ROOT}")
print(f"[Stats ] 视频总帧数: {total_frames}")
print(f"[Stats ] 实际处理帧数: {processed_count + failed_count}")
print(f"[Stats ] 成功: {processed_count}")
print(f"[Stats ] 失败: {failed_count}")
print(f"\n输出文件:")
print(f"  frame_xxxxxx_car.png      - car 二值 mask (0/255)")
if SAVE_OVERLAY:
    print(f"  frame_xxxxxx_overlay.png  - car 叠加可视化")