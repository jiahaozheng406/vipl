"""
Shared filesystem and artifact utilities for the modular Real2Sim pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_output_layout(root: str | Path) -> dict[str, Path]:
    root_path = ensure_dir(root)
    return {
        "root": root_path,
        "frames": ensure_dir(root_path / "frames"),
        "masks": ensure_dir(root_path / "masks"),
        "overlays": ensure_dir(root_path / "overlays"),
        "depth": ensure_dir(root_path / "depth"),
        "pointclouds": ensure_dir(root_path / "pointclouds"),
        "boxes": ensure_dir(root_path / "boxes"),
        "box_projections": ensure_dir(root_path / "box_projections"),
    }


def save_mask_image(mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == np.bool_ else mask.astype(np.uint8)
    cv2.imwrite(str(output_path), mask_uint8)
    return output_path


def save_overlay_image(overlay_rgb: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    overlay_bgr = cv2.cvtColor(overlay_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)
    return output_path


def save_depth_artifacts(
    depth_map: np.ndarray,
    output_npy_path: str | Path,
    output_preview_path: str | Path,
) -> tuple[Path, Path]:
    output_npy_path = Path(output_npy_path)
    output_preview_path = Path(output_preview_path)
    ensure_dir(output_npy_path.parent)
    ensure_dir(output_preview_path.parent)

    depth_array = np.asarray(depth_map, dtype=np.float32)
    np.save(output_npy_path, depth_array)
    cv2.imwrite(str(output_preview_path), _depth_preview(depth_array))
    return output_npy_path, output_preview_path


def save_pointcloud_ply(
    points_xyz: np.ndarray,
    output_path: str | Path,
    colors_rgb: np.ndarray | None = None,
) -> bool:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0 or len(points_xyz) == 0:
        LOGGER.warning("Skipping empty point cloud save for %s", output_path)
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if colors_rgb is not None and len(colors_rgb) == len(points_xyz):
        colors = np.asarray(colors_rgb, dtype=np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

    return bool(o3d.io.write_point_cloud(str(output_path), pcd))


def save_json(data: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return output_path


def _depth_preview(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    finite_mask = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(finite_mask):
        return preview

    finite_values = depth[finite_mask]
    low = float(np.percentile(finite_values, 1))
    high = float(np.percentile(finite_values, 99))
    if high <= low:
        high = low + 1e-6

    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    preview[finite_mask] = (normalized[finite_mask] * 255.0).astype(np.uint8)
    return preview
