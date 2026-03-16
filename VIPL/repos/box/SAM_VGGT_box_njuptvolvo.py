"""
njuptVolvo 单车多视角重建 — 单文件最小可运行版本
1.jpg~6.jpg → SAM3 text-prompted 分割最大 car → VGGT 联合几何推理
→ camera+depth 反投影 → 融合 object 点云 → Open3D OBB
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

# ── 路径配置 ──────────────────────────────────────────────
IMAGE_DIR = r"E:\vipl\VIPL\datadets_cityscapes\njuptVolvo"
IMAGE_NAMES = [f"{i}.jpg" for i in range(1, 7)]
VGGT_MODEL_PATH = r"E:\vipl\VIPL\models\vggt\model.safetensors"
VGGT_CONFIG_PATH = r"E:\vipl\VIPL\models\vggt\config.json"
SAM3_MODEL_PATH = r"E:\vipl\VIPL\models\sam3\sam3.pt"
SAM3_CONFIG_PATH = r"E:\vipl\VIPL\models\sam3\config.json"
OUTPUT_ROOT = r"E:\vipl\VIPL\repos\box\outputs\njuptVolvo"

# ── 分割参数 ──────────────────────────────────────────────
TARGET_LABEL = "car"
SAM_SCORE_THRESH = 0.3
SAM_MASK_THRESH = 0.5

# ── 几何参数 ──────────────────────────────────────────────
DEPTH_CONF_THRESH = 1.0          # depth 置信度阈值（VGGT conf 普遍偏低，不宜过高）
VOXEL_SIZE = 0.008               # 体素下采样尺寸，保留更多细节
OUTLIER_NB_NEIGHBORS = 20        # 统计离群点邻居数
OUTLIER_STD_RATIO = 1.0          # 统计离群点标准差倍数，更激进去离群
DBSCAN_EPS = 0.05              # DBSCAN 邻域半径，更严格
DBSCAN_MIN_POINTS = 150         # DBSCAN 最小簇点数，要求更密集

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("njuptVolvo")

# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def ensure_dir(path: str | Path) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── SAM3 分割 ─────────────────────────────────────────────

def load_sam3(model_dir: str, device: torch.device):
    """加载 SAM3 模型和 processor"""
    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
    model = Sam3Model.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    return processor, model


def segment_largest_car(
    image_path: str,
    processor,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 SAM3 text prompt "car" 分割图片，返回面积最大的 car mask (H,W bool)
    以及彩色叠加图 (H,W,3 uint8)。
    """
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)

    inputs = processor(images=image, text=TARGET_LABEL, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM_SCORE_THRESH,
        mask_threshold=SAM_MASK_THRESH,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks")
    if masks is None or (hasattr(masks, "__len__") and len(masks) == 0):
        LOG.warning("未检测到 car: %s", image_path)
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=bool), image_rgb

    # 选面积最大的 mask
    best_mask = None
    best_area = 0
    for i in range(masks.shape[0] if torch.is_tensor(masks) else len(masks)):
        m = masks[i].detach().cpu().numpy().astype(bool) if torch.is_tensor(masks) else np.asarray(masks[i], dtype=bool)
        area = int(m.sum())
        if area > best_area:
            best_area = area
            best_mask = m

    if best_mask is None or best_area == 0:
        LOG.warning("所有 mask 面积为 0: %s", image_path)
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=bool), image_rgb

    # 彩色叠加
    overlay = image_rgb.astype(np.float32).copy()
    color = np.array([255.0, 64.0, 64.0])
    overlay[best_mask] = overlay[best_mask] * 0.45 + color * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    LOG.info("SAM3 分割完成: %s  mask面积=%d", Path(image_path).name, best_area)
    return best_mask, overlay


# ── VGGT 几何推理 ─────────────────────────────────────────

def load_vggt(model_path: str, config_path: str, device: torch.device):
    """加载 VGGT 模型，返回 model 和辅助函数"""
    model_root = Path(model_path).parent
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dtype = torch.float32
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    model = VGGT(
        img_size=config.get("img_size", 518),
        patch_size=config.get("patch_size", 14),
        embed_dim=config.get("embed_dim", 1024),
    )
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    return {
        "model": model,
        "dtype": dtype,
        "load_images": load_and_preprocess_images_square,
        "pose_enc_to_extri_intri": pose_encoding_to_extri_intri,
        "unproject": unproject_depth_map_to_point_map,
        "inference_res": config.get("img_size", 518),
    }

def infer_geometry(image_paths: list[str], vggt_ctx: dict, device: torch.device):
    """
    对多张图片做联合几何推理，返回:
    extrinsics (S,3,4), intrinsics (S,3,3),
    depth_maps (S,H,W), depth_conf (S,H,W),
    world_points (S,H,W,3), input_images_rgb (S,H,W,3 uint8)
    """
    from contextlib import nullcontext

    model = vggt_ctx["model"]
    dtype = vggt_ctx["dtype"]
    load_images = vggt_ctx["load_images"]
    pose_enc_fn = vggt_ctx["pose_enc_to_extri_intri"]
    unproject_fn = vggt_ctx["unproject"]
    res = vggt_ctx["inference_res"]

    # 加载并预处理到 1024，再 resize 到 518
    images_1024, _ = load_images(image_paths, target_size=1024)
    images_518 = F.interpolate(images_1024, size=(res, res), mode="bilinear", align_corners=False)

    # 保存 518 分辨率的 RGB 用于点云着色
    input_rgb = (
        np.clip(images_518.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255)
        .round().astype(np.uint8)
    )

    images_518 = images_518.to(device)
    autocast = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    with torch.inference_mode(), autocast:
        batched = images_518.unsqueeze(0)  # (1, S, 3, H, W)
        agg_tokens, patch_idx = model.aggregator(batched)
        pose_enc = model.camera_head(agg_tokens)[-1]
        extrinsic, intrinsic = pose_enc_fn(pose_enc, batched.shape[-2:])
        depth_map, depth_conf = model.depth_head(
            agg_tokens, images=batched, patch_start_idx=patch_idx,
        )

    ext_np = extrinsic.squeeze(0).cpu().numpy().astype(np.float32)
    int_np = intrinsic.squeeze(0).cpu().numpy().astype(np.float32)
    depth_np = depth_map.squeeze(0).cpu().numpy().astype(np.float32)  # (S,H,W,1)
    conf_np = depth_conf.squeeze(0).cpu().numpy().astype(np.float32)  # (S,H,W)
    world_pts = unproject_fn(depth_np, ext_np, int_np).astype(np.float32)  # (S,H,W,3)

    LOG.info("VGGT 推理完成: %d 帧, 分辨率 %d", len(image_paths), res)
    return {
        "extrinsics": ext_np,
        "intrinsics": int_np,
        "depth_maps": depth_np[..., 0],   # (S,H,W)
        "depth_conf": conf_np,
        "world_points": world_pts,
        "input_rgb": input_rgb,
        "inference_res": res,
    }


# ── mask → 3D 点提取 ─────────────────────────────────────

def resize_mask_to_square(mask: np.ndarray, target_size: int) -> np.ndarray:
    """将任意尺寸 mask 先 pad 成正方形再 resize 到 target_size"""
    h, w = mask.shape[:2]
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    left = (max_dim - w) // 2
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    square[top:top + h, left:left + w] = (mask > 0).astype(np.uint8)
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    # 腐蚀 mask 边缘，去掉边界飘点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    resized = cv2.erode(resized, kernel, iterations=1)
    return resized.astype(bool)


def extract_masked_points(
    mask_bool: np.ndarray,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    colors_rgb: np.ndarray,
    conf_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从 mask 区域提取满足 depth 置信度的 3D 点和颜色"""
    valid = mask_bool.copy()
    valid &= np.isfinite(depth_conf) & (depth_conf >= conf_thresh)
    valid &= np.isfinite(world_points).all(axis=-1)

    pts = world_points[valid].astype(np.float32)
    clrs = colors_rgb[valid].astype(np.uint8)
    return pts, clrs

# ── 点云清理 ──────────────────────────────────────────────

def clean_pointcloud(
    pts: np.ndarray,
    clrs: np.ndarray | None,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray | None]:
    """体素下采样 + 统计离群点移除，返回 (pcd, points, colors)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if clrs is not None and len(clrs) == len(pts):
        c = clrs.astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1).astype(np.float64))

    if VOXEL_SIZE > 0 and len(pts) > 1:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    if len(pcd.points) >= max(5, OUTLIER_NB_NEIGHBORS):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO,
        )

    out_pts = np.asarray(pcd.points, dtype=np.float32)
    out_clrs = None
    if len(pcd.colors) == len(out_pts):
        out_clrs = np.clip(np.asarray(pcd.colors) * 255, 0, 255).astype(np.uint8)
    return pcd, out_pts, out_clrs


# ── depth 可视化 ──────────────────────────────────────────

def depth_preview(depth: np.ndarray) -> np.ndarray:
    """将 depth map 归一化为 0-255 灰度图"""
    d = depth.astype(np.float32).copy()
    finite = np.isfinite(d) & (d > 0)
    preview = np.zeros_like(d, dtype=np.uint8)
    if not finite.any():
        return preview
    lo = float(np.percentile(d[finite], 1))
    hi = float(np.percentile(d[finite], 99))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    preview[finite] = (norm[finite] * 255).astype(np.uint8)
    return preview


# ── 可视化：点云 + OBB 离屏渲染 ──────────────────────────

def render_pointcloud_with_bbox(
    pcd: o3d.geometry.PointCloud,
    bbox: o3d.geometry.OrientedBoundingBox | None,
    save_path: str,
    width: int = 1280,
    height: int = 720,
) -> bool:
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(pcd)
        if bbox is not None:
            vis.add_geometry(bbox)
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0], dtype=np.float64)
        opt.point_size = 2.0
        vis.poll_events()
        vis.update_renderer()
        ensure_dir(Path(save_path).parent)
        vis.capture_screen_image(str(save_path), do_render=True)
        return True
    except Exception as e:
        LOG.warning("Open3D 渲染失败: %s", e)
        return False
    finally:
        vis.destroy_window()

# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("设备: %s", device)

    # 输出目录
    out = Path(OUTPUT_ROOT)
    dir_overlays = ensure_dir(out / "overlays")
    dir_masks    = ensure_dir(out / "masks")
    dir_depth    = ensure_dir(out / "depth")
    dir_pc       = ensure_dir(out / "pointclouds")
    dir_boxes    = ensure_dir(out / "boxes")

    image_paths = [str(Path(IMAGE_DIR) / n) for n in IMAGE_NAMES]
    for p in image_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"输入图片不存在: {p}")

    # ── Stage 1: SAM3 分割 ────────────────────────────────
    LOG.info("=" * 50)
    LOG.info("Stage 1: SAM3 car 分割")
    sam3_dir = str(Path(SAM3_MODEL_PATH).parent)
    processor, sam_model = load_sam3(sam3_dir, device)

    masks_bool: list[np.ndarray] = []
    for img_path in image_paths:
        name = Path(img_path).stem
        mask, overlay = segment_largest_car(img_path, processor, sam_model, device)
        masks_bool.append(mask)
        # 保存 mask 和叠加图
        cv2.imwrite(str(dir_masks / f"{name}_car_mask.png"),
                     (mask.astype(np.uint8) * 255))
        cv2.imwrite(str(dir_overlays / f"{name}_car_overlay.png"),
                     cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    del processor, sam_model
    release_gpu()
    LOG.info("SAM3 完成，释放显存")

    # ── Stage 2: VGGT 几何推理 ────────────────────────────
    LOG.info("=" * 50)
    LOG.info("Stage 2: VGGT 联合几何推理")
    vggt_ctx = load_vggt(VGGT_MODEL_PATH, VGGT_CONFIG_PATH, device)
    geo = infer_geometry(image_paths, vggt_ctx, device)

    # 保存 depth 图
    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        d = geo["depth_maps"][i]
        np.save(str(dir_depth / f"{name}_depth.npy"), d)
        cv2.imwrite(str(dir_depth / f"{name}_depth.png"), depth_preview(d))

    del vggt_ctx
    release_gpu()
    LOG.info("VGGT 完成，释放显存")

    # ── Stage 3: mask + depth → object 点云 ───────────────
    LOG.info("=" * 50)
    LOG.info("Stage 3: 提取 object 点云")
    res = geo["inference_res"]
    all_pts_list: list[np.ndarray] = []
    all_clrs_list: list[np.ndarray] = []

    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        mask_resized = resize_mask_to_square(masks_bool[i], res)
        pts, clrs = extract_masked_points(
            mask_resized,
            geo["world_points"][i],
            geo["depth_conf"][i],
            geo["input_rgb"][i],
            DEPTH_CONF_THRESH,
        )
        LOG.info("  帧 %s: 提取 %d 个点", name, len(pts))

        if len(pts) > 0:
            all_pts_list.append(pts)
            all_clrs_list.append(clrs)
            # 保存单帧点云
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            c = clrs.astype(np.float32) / 255.0
            pcd_i.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
            o3d.io.write_point_cloud(str(dir_pc / f"{name}_car.ply"), pcd_i)

    if not all_pts_list:
        LOG.error("所有帧均未提取到点云，pipeline 终止")
        return

    # ── Stage 4: 融合 + 清理 ──────────────────────────────
    LOG.info("=" * 50)
    LOG.info("Stage 4: 融合并清理点云")
    merged_pts = np.concatenate(all_pts_list, axis=0)
    merged_clrs = np.concatenate(all_clrs_list, axis=0)
    LOG.info("融合后点数: %d", len(merged_pts))

    pcd_clean, clean_pts, clean_clrs = clean_pointcloud(merged_pts, merged_clrs)
    LOG.info("清理后点数: %d", len(clean_pts))

    # 保存融合点云（清理后、聚类前）
    o3d.io.write_point_cloud(str(dir_pc / "car_merged_raw.ply"),
        _make_pcd(merged_pts, merged_clrs))

    # ── Stage 4.5: DBSCAN 聚类，只保留最大簇 ─────────────
    LOG.info("DBSCAN 聚类: eps=%.3f, min_points=%d", DBSCAN_EPS, DBSCAN_MIN_POINTS)
    labels = np.asarray(pcd_clean.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False,
    ))
    if labels.size > 0 and labels.max() >= 0:
        # 选最大簇
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        keep_mask = labels == largest_label
        LOG.info("  共 %d 个簇, 最大簇 label=%d 含 %d 点, 丢弃 %d 个离群/小簇点",
                 len(unique_labels), largest_label, keep_mask.sum(),
                 len(labels) - keep_mask.sum())
        pcd_clean = pcd_clean.select_by_index(np.where(keep_mask)[0])
        clean_pts = np.asarray(pcd_clean.points, dtype=np.float32)
        clean_clrs = None
        if len(pcd_clean.colors) == len(clean_pts):
            clean_clrs = np.clip(np.asarray(pcd_clean.colors) * 255, 0, 255).astype(np.uint8)
    else:
        LOG.warning("  DBSCAN 未找到有效簇，跳过聚类过滤")

    o3d.io.write_point_cloud(str(dir_pc / "car_merged_cleaned.ply"), pcd_clean)

    # 稀疏点云可视化（无 bbox）
    render_pointcloud_with_bbox(pcd_clean, None,
        str(dir_pc / "car_pointcloud_preview.png"))

    # ── Stage 5: OBB ─────────────────────────────────────
    LOG.info("=" * 50)
    LOG.info("Stage 5: 计算 OBB")
    if len(clean_pts) < 4:
        LOG.error("清理后点数不足 4，无法构建 OBB")
        return

    obb = pcd_clean.get_minimal_oriented_bounding_box()
    obb.color = (0.0, 1.0, 0.0)  # 绿色 OBB
    LOG.info("OBB center=%s  extent=%s",
             np.asarray(obb.center).round(4).tolist(),
             np.asarray(obb.extent).round(4).tolist())

    # 保存 bbox json
    bbox_info = {
        "center": np.asarray(obb.center, dtype=np.float32).tolist(),
        "extent": np.asarray(obb.extent, dtype=np.float32).tolist(),
        "R": np.asarray(obb.R, dtype=np.float32).tolist(),
        "corners": np.asarray(obb.get_box_points(), dtype=np.float32).tolist(),
    }
    with open(str(dir_boxes / "car_obb.json"), "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, indent=2, ensure_ascii=False)

    # 保存带绿色 OBB 边框的彩色点云 ply（方便全方位查看）
    line_set_ply = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    box_edge_pts = np.asarray(line_set_ply.points)
    box_edge_lines = np.asarray(line_set_ply.lines)
    # 沿每条边插值采样点，让 OBB 在 ply 中可见
    bbox_sample_pts = []
    for s, e in box_edge_lines:
        p0, p1 = box_edge_pts[s], box_edge_pts[e]
        for t in np.linspace(0, 1, 60):
            bbox_sample_pts.append(p0 * (1 - t) + p1 * t)
    bbox_sample_pts = np.array(bbox_sample_pts, dtype=np.float64)
    bbox_colors = np.tile([0.0, 1.0, 0.0], (len(bbox_sample_pts), 1))  # 绿色

    pcd_with_box = o3d.geometry.PointCloud()
    combined_pts = np.vstack([np.asarray(pcd_clean.points), bbox_sample_pts])
    pcd_with_box.points = o3d.utility.Vector3dVector(combined_pts)
    if len(pcd_clean.colors) == len(pcd_clean.points):
        combined_clrs = np.vstack([np.asarray(pcd_clean.colors), bbox_colors])
    else:
        combined_clrs = np.vstack([
            np.ones((len(pcd_clean.points), 3)) * 0.7,
            bbox_colors,
        ])
    pcd_with_box.colors = o3d.utility.Vector3dVector(combined_clrs)
    ply_path = str(dir_boxes / "car_with_obb.ply")
    o3d.io.write_point_cloud(ply_path, pcd_with_box)
    pcd_path = str(dir_boxes / "car_with_obb.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd_with_box, write_ascii=False)
    LOG.info("已保存带 OBB 的点云: %s / %s", ply_path, pcd_path)
    render_pointcloud_with_bbox(pcd_clean, obb,
        str(dir_boxes / "car_obb_result.png"))

    # 将 OBB 投影回每张原图
    LOG.info("将 OBB 投影回原图...")
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    box_world_pts = np.asarray(line_set.points, dtype=np.float32)
    box_lines = np.asarray(line_set.lines, dtype=np.int32)

    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]
        ext = geo["extrinsics"][i]
        intr = geo["intrinsics"][i]

        # 世界坐标 → 相机坐标 → 像素坐标 (518x518 空间)
        cam_pts = box_world_pts @ ext[:, :3].T + ext[:, 3]
        valid = np.isfinite(cam_pts).all(axis=1) & (cam_pts[:, 2] > 1e-6)
        px_518 = np.full((len(box_world_pts), 2), np.nan, dtype=np.float32)
        if valid.any():
            proj = cam_pts[valid] @ intr.T
            px_518[valid] = proj[:, :2] / proj[:, 2:3]

        # 518 空间 → 原图空间 (考虑 square padding)
        max_dim = max(h_orig, w_orig)
        scale = max_dim / res
        top_pad = (max_dim - h_orig) / 2.0
        left_pad = (max_dim - w_orig) / 2.0
        px_orig = px_518.copy()
        finite_mask = np.isfinite(px_orig).all(axis=1)
        px_orig[finite_mask, 0] = px_orig[finite_mask, 0] * scale - left_pad
        px_orig[finite_mask, 1] = px_orig[finite_mask, 1] * scale - top_pad

        overlay_bgr = img_bgr.copy()
        for s_idx, e_idx in box_lines:
            if not (valid[s_idx] and valid[e_idx]):
                continue
            pt1 = tuple(np.round(px_orig[s_idx]).astype(int).tolist())
            pt2 = tuple(np.round(px_orig[e_idx]).astype(int).tolist())
            cv2.line(overlay_bgr, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(str(dir_boxes / f"{name}_car_obb_proj.png"), overlay_bgr)

    elapsed = time.time() - t0
    LOG.info("=" * 50)
    LOG.info("Pipeline 完成! 总耗时 %.1f 秒", elapsed)
    LOG.info("输出目录: %s", OUTPUT_ROOT)


def _make_pcd(pts, clrs):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if clrs is not None and len(clrs) == len(pts):
        c = clrs.astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd


if __name__ == "__main__":
    main()
