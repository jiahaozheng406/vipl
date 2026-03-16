"""
all2asset_njupyvolvoscene.py
njuptVolvo 场景全物体 Real2Sim Pipeline — 单文件完整版

流程:
  Stage 1: SAM3 多类别实例分割 (car / tree / grass / pole / building / road)
  Stage 2: VGGT 联合几何推理 (depth + camera + world points, 只跑一次)
  Stage 3: 逐实例 mask → depth 反投影 → 融合点云 → 清理 → DBSCAN → OBB
  Stage 4: 有 UrbanVerse 对应类别的物体 → 比例匹配 3D 资产
  Stage 5: 汇总输出 (点云 / OBB / 资产匹配 / 预览图 / JSON)

输入: E:/vipl/VIPL/datadets_cityscapes/njuptVolvo/1.jpg~6.jpg
输出: E:/vipl/VIPL/repos/all2asset/outputs/njuptVolvo/
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

# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

IMAGE_DIR = r"E:\vipl\VIPL\datadets_cityscapes\njuptVolvo"
IMAGE_NAMES = [f"{i}.jpg" for i in range(1, 7)]
VGGT_MODEL_PATH = r"E:\vipl\VIPL\models\vggt\model.safetensors"
VGGT_CONFIG_PATH = r"E:\vipl\VIPL\models\vggt\config.json"
SAM3_MODEL_DIR = r"E:\vipl\VIPL\models\sam3"
OUTPUT_ROOT = Path(r"E:\vipl\VIPL\repos\all2asset\outputs\njuptVolvo")
URBANVERSE_CACHE = Path(r"E:\vipl\VIPL\repos\all2asset\outputs\urbanverse_cache")

# ── SAM3 分割参数 ──
SAM_SCORE_THRESH = 0.3
SAM_MASK_THRESH = 0.5
# ── 几何参数 ──
DEPTH_CONF_THRESH = 1.0
VOXEL_SIZE = 0.008
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 1.0
DBSCAN_EPS = 0.05
DBSCAN_MIN_POINTS = 100

# ── 目标物体定义 ──
# sam3_label: SAM3 text prompt
# uv_category: UrbanVerse L1 类别 (None = 不做资产匹配, 只输出点云+OBB)
# uv_query: UrbanVerse 语义查询文本
# multi_instance: 是否可能有多个实例 (True → 用 DBSCAN 分离)
# color: 叠加可视化颜色 (RGB)
TARGETS = [
    {
        "name": "car",
        "sam3_label": "car",
        "uv_category": "vehicle",
        "uv_query": "gray sedan car Volvo",
        "multi_instance": True,
        "min_mask_area": 2000,
        "color": [255, 64, 64],
    },
    {
        "name": "tree",
        "sam3_label": "tree",
        "uv_category": "nature",
        "uv_query": "tree",
        "multi_instance": True,
        "min_mask_area": 1500,
        "color": [64, 200, 64],
    },
    {
        "name": "grass",
        "sam3_label": "grass",
        "uv_category": "nature",
        "uv_query": "grass patch",
        "multi_instance": False,
        "min_mask_area": 3000,
        "color": [100, 255, 100],
    },
    {
        "name": "pole",
        "sam3_label": "utility pole",
        "uv_category": "amenity",
        "uv_query": "utility pole street light",
        "multi_instance": True,
        "min_mask_area": 500,
        "color": [200, 200, 64],
    },
    {
        "name": "building",
        "sam3_label": "building",
        "uv_category": "building",
        "uv_query": "building",
        "multi_instance": False,
        "min_mask_area": 5000,
        "color": [64, 128, 255],
    },
    {
        "name": "road",
        "sam3_label": "road",
        "uv_category": None,
        "uv_query": None,
        "multi_instance": False,
        "min_mask_area": 5000,
        "color": [180, 180, 180],
    },
]

URBANVERSE_CANDIDATE_K = 20
URBANVERSE_BEST_K = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("all2asset")

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

# ═══════════════════════════════════════════════════════════
#  Stage 1: SAM3 多类别分割
# ═══════════════════════════════════════════════════════════

def load_sam3(model_dir: str, device: torch.device):
    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
    model = Sam3Model.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    return processor, model


def segment_all_instances(
    image_path: str, label: str, processor, model, device: torch.device,
    min_area: int = 500,
) -> list[np.ndarray]:
    """用 SAM3 text prompt 分割图片，返回所有满足面积阈值的 mask 列表 (各为 H,W bool)"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=label, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM_SCORE_THRESH,
        mask_threshold=SAM_MASK_THRESH,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks_tensor = results.get("masks")
    if masks_tensor is None or (hasattr(masks_tensor, "__len__") and len(masks_tensor) == 0):
        return []
    masks_out = []
    n = masks_tensor.shape[0] if torch.is_tensor(masks_tensor) else len(masks_tensor)
    for i in range(n):
        m = masks_tensor[i].detach().cpu().numpy().astype(bool) if torch.is_tensor(masks_tensor) else np.asarray(masks_tensor[i], dtype=bool)
        if int(m.sum()) >= min_area:
            masks_out.append(m)
    # 按面积降序
    masks_out.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks_out


def merge_masks(masks: list[np.ndarray]) -> np.ndarray:
    """将多个 bool mask 合并为一个 (OR)"""
    if not masks:
        return None
    merged = masks[0].copy()
    for m in masks[1:]:
        merged |= m
    return merged

# ═══════════════════════════════════════════════════════════
#  Stage 2: VGGT 几何推理
# ═══════════════════════════════════════════════════════════

def load_vggt(model_path: str, config_path: str, device: torch.device):
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
        "model": model, "dtype": dtype,
        "load_images": load_and_preprocess_images_square,
        "pose_enc_to_extri_intri": pose_encoding_to_extri_intri,
        "unproject": unproject_depth_map_to_point_map,
        "inference_res": config.get("img_size", 518),
    }


def infer_geometry(image_paths: list[str], vggt_ctx: dict, device: torch.device):
    from contextlib import nullcontext
    model = vggt_ctx["model"]
    dtype = vggt_ctx["dtype"]
    load_images = vggt_ctx["load_images"]
    pose_enc_fn = vggt_ctx["pose_enc_to_extri_intri"]
    unproject_fn = vggt_ctx["unproject"]
    res = vggt_ctx["inference_res"]

    images_1024, _ = load_images(image_paths, target_size=1024)
    images_518 = F.interpolate(images_1024, size=(res, res), mode="bilinear", align_corners=False)
    input_rgb = (
        np.clip(images_518.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255)
        .round().astype(np.uint8)
    )
    images_518 = images_518.to(device)
    autocast = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    with torch.inference_mode(), autocast:
        batched = images_518.unsqueeze(0)
        agg_tokens, patch_idx = model.aggregator(batched)
        pose_enc = model.camera_head(agg_tokens)[-1]
        extrinsic, intrinsic = pose_enc_fn(pose_enc, batched.shape[-2:])
        depth_map, depth_conf = model.depth_head(
            agg_tokens, images=batched, patch_start_idx=patch_idx,
        )

    ext_np = extrinsic.squeeze(0).cpu().numpy().astype(np.float32)
    int_np = intrinsic.squeeze(0).cpu().numpy().astype(np.float32)
    depth_np = depth_map.squeeze(0).cpu().numpy().astype(np.float32)
    conf_np = depth_conf.squeeze(0).cpu().numpy().astype(np.float32)
    world_pts = unproject_fn(depth_np, ext_np, int_np).astype(np.float32)

    LOG.info("VGGT 推理完成: %d 帧, 分辨率 %d", len(image_paths), res)
    return {
        "extrinsics": ext_np, "intrinsics": int_np,
        "depth_maps": depth_np[..., 0],
        "depth_conf": conf_np,
        "world_points": world_pts,
        "input_rgb": input_rgb,
        "inference_res": res,
    }

# ═══════════════════════════════════════════════════════════
#  Stage 3: mask → 点云 → 清理 → OBB
# ═══════════════════════════════════════════════════════════

def resize_mask_to_square(mask: np.ndarray, target_size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    left = (max_dim - w) // 2
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    square[top:top + h, left:left + w] = (mask > 0).astype(np.uint8)
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    resized = cv2.erode(resized, kernel, iterations=1)
    return resized.astype(bool)


def extract_masked_points(
    mask_bool: np.ndarray, world_points: np.ndarray,
    depth_conf: np.ndarray, colors_rgb: np.ndarray, conf_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask_bool.copy()
    valid &= np.isfinite(depth_conf) & (depth_conf >= conf_thresh)
    valid &= np.isfinite(world_points).all(axis=-1)
    pts = world_points[valid].astype(np.float32)
    clrs = colors_rgb[valid].astype(np.uint8)
    return pts, clrs


def make_pcd(pts, clrs=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if clrs is not None and len(clrs) == len(pts):
        c = clrs.astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd


def clean_pointcloud(pts, clrs):
    pcd = make_pcd(pts, clrs)
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


def dbscan_largest_cluster(pcd):
    """DBSCAN 聚类，返回最大簇的 pcd"""
    labels = np.asarray(pcd.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False,
    ))
    if labels.size == 0 or labels.max() < 0:
        return pcd
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return pcd
    best_label = unique[np.argmax(counts)]
    mask = labels == best_label
    LOG.info("    DBSCAN: %d 簇, 最大簇 %d 点 (共 %d)", len(unique), mask.sum(), len(labels))
    return pcd.select_by_index(np.where(mask)[0])


def dbscan_split_instances(pcd, min_points=100):
    """DBSCAN 聚类，返回所有满足最小点数的簇 pcd 列表"""
    labels = np.asarray(pcd.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=min(DBSCAN_MIN_POINTS, min_points), print_progress=False,
    ))
    if labels.size == 0 or labels.max() < 0:
        return [pcd]
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    clusters = []
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if cnt >= min_points:
            idx = np.where(labels == lbl)[0]
            clusters.append(pcd.select_by_index(idx))
    LOG.info("    DBSCAN 分离: %d 簇 (>=%d 点)", len(clusters), min_points)
    return clusters if clusters else [pcd]


def fit_obb(pcd):
    """拟合 OBB，返回 (obb, obb_dict)"""
    try:
        obb = pcd.get_minimal_oriented_bounding_box()
    except Exception:
        obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    center = obb.center.tolist()
    extent = obb.extent.tolist()
    R = obb.R.tolist()
    corners = np.asarray(obb.get_box_points()).tolist()
    return obb, {
        "center": center, "extent": extent, "R": R, "corners": corners,
    }


def save_pcd_with_obb(pcd, obb, save_path):
    """保存带绿色 OBB 边框的点云文件"""
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    edge_pts = np.asarray(line_set.points)
    edge_lines = np.asarray(line_set.lines)
    bbox_sample = []
    for s, e in edge_lines:
        p0, p1 = edge_pts[s], edge_pts[e]
        for t in np.linspace(0, 1, 60):
            bbox_sample.append(p0 * (1 - t) + p1 * t)
    bbox_sample = np.array(bbox_sample, dtype=np.float64)
    bbox_colors = np.tile([0.0, 1.0, 0.0], (len(bbox_sample), 1))

    combined_pts = np.vstack([np.asarray(pcd.points), bbox_sample])
    combined_clrs = np.vstack([
        np.asarray(pcd.colors) if len(pcd.colors) == len(pcd.points)
        else np.ones((len(pcd.points), 3)) * 0.7,
        bbox_colors,
    ])
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(combined_pts)
    out_pcd.colors = o3d.utility.Vector3dVector(combined_clrs)
    o3d.io.write_point_cloud(str(save_path), out_pcd, write_ascii=False)

# ═══════════════════════════════════════════════════════════
#  Stage 4: UrbanVerse 资产匹配
# ═══════════════════════════════════════════════════════════

def extent_to_ratio(ext):
    ext = np.array(sorted(ext, reverse=True), dtype=np.float64)
    return ext / ext[0]


def ratio_distance(r1, r2):
    return float(np.linalg.norm(r1 - r2))


def glb_bbox_extent(glb_path):
    import trimesh
    scene = trimesh.load(str(glb_path), force='scene')
    return sorted(scene.bounding_box.extents.tolist(), reverse=True)


def match_urbanverse_asset(obb_dict, uv_category, uv_query, output_dir):
    """
    用 OBB 比例从 UrbanVerse 匹配最佳 3D 资产。
    返回匹配结果 dict 或 None。
    """
    import urbanverse_asset as uva
    uva.set(str(URBANVERSE_CACHE))

    obb_ratio = extent_to_ratio(obb_dict["extent"])
    LOG.info("    OBB 比例: %s", obb_ratio.round(3).tolist())

    # 语义查询候选
    try:
        candidate_uids = uva.object.get_uids_conditioned(
            categories=[uv_category],
            query=uv_query,
            top_k=URBANVERSE_CANDIDATE_K,
        )
    except Exception as e:
        LOG.warning("    UrbanVerse 查询失败: %s", e)
        return None

    if not candidate_uids:
        LOG.warning("    未找到候选资产 (category=%s, query=%s)", uv_category, uv_query)
        return None
    LOG.info("    语义候选: %d 个", len(candidate_uids))

    # 下载 GLB + annotation + thumbnail
    try:
        result = uva.object.load(candidate_uids, what=("std_glb", "std_annotation", "thumbnail"))
    except Exception as e:
        LOG.warning("    资产下载失败: %s", e)
        return None

    # 比例匹配排序
    scored = []
    for uid in candidate_uids:
        info = result.get(uid, {})
        glb_path = info.get("std_glb")
        if not glb_path or not Path(glb_path).exists():
            continue
        try:
            glb_ext = glb_bbox_extent(glb_path)
            glb_ratio = extent_to_ratio(glb_ext)
            dist = ratio_distance(obb_ratio, glb_ratio)
            scored.append((uid, dist, glb_ext, info))
        except Exception as e:
            LOG.warning("    %s: GLB 读取失败 - %s", uid[:8], e)

    if not scored:
        LOG.warning("    无有效 GLB 可比较")
        return None

    scored.sort(key=lambda x: x[1])
    best_k = scored[:URBANVERSE_BEST_K]

    # 保存最佳 thumbnail
    best_uid, best_dist, best_ext, best_info = best_k[0]
    thumb_path = best_info.get("thumbnail")
    if thumb_path and Path(thumb_path).exists():
        img = Image.open(thumb_path)
        img.save(str(output_dir / f"best_{best_uid[:12]}_thumb.png"))

    # Top-5 拼图
    thumbs = []
    for uid, dist, ext, info in best_k:
        tp = info.get("thumbnail")
        if tp and Path(tp).exists():
            thumbs.append(Image.open(tp))
    if thumbs:
        w = max(t.width for t in thumbs)
        h_total = sum(t.height for t in thumbs)
        grid = Image.new("RGB", (w, h_total), (255, 255, 255))
        y = 0
        for t in thumbs:
            grid.paste(t, (0, y))
            y += t.height
        grid.save(str(output_dir / "top5_candidates.png"))

    # 读取最佳 annotation
    best_ann = {}
    ann_path = best_info.get("std_annotation")
    if ann_path and Path(ann_path).exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            best_ann = json.load(f)

    return {
        "best_uid": best_uid,
        "glb_path": str(best_info.get("std_glb", "")),
        "annotation": best_ann,
        "obb_ratio": obb_ratio.tolist(),
        "query": uv_query,
        "ratio_distance": round(best_dist, 4),
        "ranking": [
            {"uid": uid, "ratio_dist": round(dist, 4),
             "bbox_meters": [round(x, 3) for x in ext]}
            for uid, dist, ext, _ in best_k
        ],
    }

# ═══════════════════════════════════════════════════════════
#  Stage 5: 可视化辅助
# ═══════════════════════════════════════════════════════════

def depth_preview(depth: np.ndarray) -> np.ndarray:
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


def make_overlay(image_rgb, mask_bool, color):
    overlay = image_rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * 0.45 + c * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("设备: %s", device)

    # 输出目录
    out_root = ensure_dir(OUTPUT_ROOT)
    dir_depth = ensure_dir(out_root / "depth")

    image_paths = [str(Path(IMAGE_DIR) / n) for n in IMAGE_NAMES]
    for p in image_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"输入图片不存在: {p}")

    # ── Stage 1: SAM3 多类别分割 ──────────────────────────
    LOG.info("=" * 60)
    LOG.info("Stage 1: SAM3 多类别实例分割")
    processor, sam_model = load_sam3(SAM3_MODEL_DIR, device)

    # {target_name: {img_idx: [mask1, mask2, ...]}}
    all_masks: dict[str, dict[int, list[np.ndarray]]] = {}

    for tgt in TARGETS:
        name = tgt["name"]
        label = tgt["sam3_label"]
        min_area = tgt["min_mask_area"]
        all_masks[name] = {}
        LOG.info("  分割类别: %s (prompt='%s')", name, label)

        for idx, img_path in enumerate(image_paths):
            masks = segment_all_instances(img_path, label, processor, sam_model, device, min_area)
            all_masks[name][idx] = masks
            total_area = sum(int(m.sum()) for m in masks)
            LOG.info("    帧 %d: %d 个实例, 总面积 %d", idx + 1, len(masks), total_area)

    del processor, sam_model
    release_gpu()
    LOG.info("SAM3 完成，释放显存")

    # 保存分割可视化
    for tgt in TARGETS:
        name = tgt["name"]
        color = tgt["color"]
        dir_masks = ensure_dir(out_root / name / "masks")
        dir_overlays = ensure_dir(out_root / name / "overlays")
        for idx, img_path in enumerate(image_paths):
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            merged = merge_masks(all_masks[name][idx])
            if merged is not None:
                cv2.imwrite(str(dir_masks / f"{idx+1}_mask.png"),
                            (merged.astype(np.uint8) * 255))
                overlay = make_overlay(img_rgb, merged, color)
                cv2.imwrite(str(dir_overlays / f"{idx+1}_overlay.png"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                # 空 mask
                h, w = img_rgb.shape[:2]
                cv2.imwrite(str(dir_masks / f"{idx+1}_mask.png"),
                            np.zeros((h, w), dtype=np.uint8))

    # ── Stage 2: VGGT 几何推理 (只跑一次) ────────────────
    LOG.info("=" * 60)
    LOG.info("Stage 2: VGGT 联合几何推理")
    vggt_ctx = load_vggt(VGGT_MODEL_PATH, VGGT_CONFIG_PATH, device)
    geo = infer_geometry(image_paths, vggt_ctx, device)

    for idx, img_path in enumerate(image_paths):
        d = geo["depth_maps"][idx]
        np.save(str(dir_depth / f"{idx+1}_depth.npy"), d)
        cv2.imwrite(str(dir_depth / f"{idx+1}_depth.png"), depth_preview(d))

    del vggt_ctx
    release_gpu()
    LOG.info("VGGT 完成，释放显存")

    # ── Stage 3 & 4: 逐类别提取点云 → OBB → 资产匹配 ──
    LOG.info("=" * 60)
    LOG.info("Stage 3-4: 逐类别点云提取 + OBB + 资产匹配")
    res = geo["inference_res"]
    scene_summary = {"scene": "njuptVolvo", "objects": []}

    for tgt in TARGETS:
        name = tgt["name"]
        multi_inst = tgt["multi_instance"]
        uv_cat = tgt["uv_category"]
        uv_query = tgt["uv_query"]

        LOG.info("-" * 50)
        LOG.info("处理类别: %s", name)
        dir_obj = ensure_dir(out_root / name)
        dir_pc = ensure_dir(dir_obj / "pointclouds")
        dir_boxes = ensure_dir(dir_obj / "boxes")

        # 合并所有帧的 mask → 提取点云
        all_pts, all_clrs = [], []
        for idx in range(len(image_paths)):
            merged = merge_masks(all_masks[name][idx])
            if merged is None:
                continue
            mask_sq = resize_mask_to_square(merged, res)
            pts, clrs = extract_masked_points(
                mask_sq, geo["world_points"][idx],
                geo["depth_conf"][idx], geo["input_rgb"][idx],
                DEPTH_CONF_THRESH,
            )
            LOG.info("  帧 %d: %d 点", idx + 1, len(pts))
            if len(pts) > 0:
                all_pts.append(pts)
                all_clrs.append(clrs)

        if not all_pts:
            LOG.warning("  %s: 所有帧均无有效点云，跳过", name)
            continue

        merged_pts = np.concatenate(all_pts)
        merged_clrs = np.concatenate(all_clrs)
        LOG.info("  融合点数: %d", len(merged_pts))

        # 清理
        pcd_clean, clean_pts, clean_clrs = clean_pointcloud(merged_pts, merged_clrs)
        LOG.info("  清理后: %d 点", len(clean_pts))

        if len(clean_pts) < 50:
            LOG.warning("  %s: 清理后点数过少，跳过", name)
            continue

        # 保存融合点云
        o3d.io.write_point_cloud(str(dir_pc / f"{name}_merged.ply"), pcd_clean)

        # 分实例 or 单实例
        if multi_inst:
            instances = dbscan_split_instances(pcd_clean, min_points=max(50, DBSCAN_MIN_POINTS // 2))
        else:
            # 单实例: DBSCAN 取最大簇
            pcd_main = dbscan_largest_cluster(pcd_clean)
            instances = [pcd_main]

        LOG.info("  实例数: %d", len(instances))

        for inst_idx, pcd_inst in enumerate(instances):
            inst_name = f"{name}_{inst_idx}" if len(instances) > 1 else name
            n_pts = len(pcd_inst.points)
            LOG.info("  实例 %s: %d 点", inst_name, n_pts)

            if n_pts < 30:
                continue

            # OBB
            obb, obb_dict = fit_obb(pcd_inst)
            obb_path = dir_boxes / f"{inst_name}_obb.json"
            with open(str(obb_path), "w", encoding="utf-8") as f:
                json.dump(obb_dict, f, indent=2, ensure_ascii=False)

            # 带 OBB 的点云
            save_pcd_with_obb(pcd_inst, obb, dir_boxes / f"{inst_name}_with_obb.pcd")
            o3d.io.write_point_cloud(str(dir_pc / f"{inst_name}.ply"), pcd_inst)

            LOG.info("    OBB extent: %s", [round(x, 4) for x in obb_dict["extent"]])

            # 资产匹配
            asset_result = None
            if uv_cat and uv_query:
                LOG.info("    UrbanVerse 匹配: category=%s, query='%s'", uv_cat, uv_query)
                dir_asset = ensure_dir(dir_obj / "asset")
                asset_result = match_urbanverse_asset(obb_dict, uv_cat, uv_query, dir_asset)
                if asset_result:
                    asset_json = dir_asset / f"{inst_name}_asset.json"
                    with open(str(asset_json), "w", encoding="utf-8") as f:
                        json.dump(asset_result, f, indent=2, ensure_ascii=False)
                    LOG.info("    最佳匹配: %s (dist=%.4f)",
                             asset_result["best_uid"][:12], asset_result["ratio_distance"])
                else:
                    LOG.info("    未匹配到合适资产")

            # 汇总
            obj_info = {
                "name": inst_name,
                "category": name,
                "point_count": n_pts,
                "obb": obb_dict,
                "obb_file": str(obb_path),
                "has_asset": asset_result is not None,
            }
            if asset_result:
                obj_info["asset_uid"] = asset_result["best_uid"]
                obj_info["asset_glb"] = asset_result["glb_path"]
                obj_info["asset_ratio_dist"] = asset_result["ratio_distance"]
            scene_summary["objects"].append(obj_info)

    # ── Stage 5: 汇总输出 ──────────────────────────────────
    LOG.info("=" * 60)
    LOG.info("Stage 5: 汇总输出")

    # 场景汇总 JSON
    summary_path = out_root / "scene_summary.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump(scene_summary, f, indent=2, ensure_ascii=False)
    LOG.info("场景汇总: %s", summary_path)

    # 全场景合并点云 (所有类别叠加, 各类别用不同颜色)
    scene_pts_list, scene_clrs_list = [], []
    color_map = {tgt["name"]: np.array(tgt["color"], dtype=np.float64) / 255.0 for tgt in TARGETS}
    for tgt in TARGETS:
        name = tgt["name"]
        merged_ply = out_root / name / "pointclouds" / f"{name}_merged.ply"
        if merged_ply.exists():
            pcd = o3d.io.read_point_cloud(str(merged_ply))
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                scene_pts_list.append(pts)
                # 用类别颜色着色
                clrs = np.tile(color_map[name], (len(pts), 1))
                scene_clrs_list.append(clrs)

    if scene_pts_list:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.vstack(scene_pts_list))
        scene_pcd.colors = o3d.utility.Vector3dVector(np.vstack(scene_clrs_list))
        scene_ply = out_root / "scene_all_objects.ply"
        o3d.io.write_point_cloud(str(scene_ply), scene_pcd)
        LOG.info("全场景点云: %s (%d 点)", scene_ply, len(scene_pcd.points))

    elapsed = time.time() - t0
    LOG.info("=" * 60)
    LOG.info("Pipeline 完成! 总耗时 %.1f 秒", elapsed)
    LOG.info("输出目录: %s", OUTPUT_ROOT)
    LOG.info("物体数: %d", len(scene_summary["objects"]))
    for obj in scene_summary["objects"]:
        asset_str = f" → asset={obj['asset_uid'][:12]}" if obj.get("asset_uid") else ""
        LOG.info("  %s: %d pts, extent=%s%s",
                 obj["name"], obj["point_count"],
                 [round(x, 3) for x in obj["obb"]["extent"]], asset_str)


if __name__ == "__main__":
    main()
