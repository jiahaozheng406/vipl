"""
Geometry helpers for converting masks and model outputs into object-level 3D points.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def extract_object_points_from_geometry(
    mask: np.ndarray,
    geometry_result: dict[str, Any],
    frame_index: int,
    depth_conf_threshold: float,
) -> dict[str, Any]:
    inference_resolution = int(geometry_result["inference_resolution"])
    mask_resized = _resize_mask_to_square_grid(mask, inference_resolution)

    world_points = np.asarray(geometry_result["world_points_from_depth"][frame_index], dtype=np.float32)
    depth_conf = np.asarray(geometry_result["depth_conf"][frame_index], dtype=np.float32)
    input_images_rgb = geometry_result.get("input_images_rgb")
    colors_rgb = None
    if input_images_rgb is not None:
        colors_rgb = np.asarray(input_images_rgb[frame_index], dtype=np.uint8)

    valid = mask_resized.copy()
    valid &= np.isfinite(depth_conf)
    valid &= depth_conf >= float(depth_conf_threshold)
    valid &= np.isfinite(world_points).all(axis=-1)

    points_xyz = world_points[valid]
    selected_colors = colors_rgb[valid] if colors_rgb is not None else None
    return {
        "frame_index": frame_index,
        "num_points": int(points_xyz.shape[0]),
        "points_xyz": points_xyz.astype(np.float32),
        "colors_rgb": selected_colors.astype(np.uint8) if selected_colors is not None else None,
        "mask_resized": mask_resized,
        "mask_area": int(mask_resized.sum()),
        "valid_mask_area": int(valid.sum()),
        "used_depth_conf_threshold": float(depth_conf_threshold),
    }


def extract_object_points_with_support_filter(
    mask: np.ndarray,
    geometry_result: dict[str, Any],
    frame_index: int,
    *,
    support_depth_conf_threshold: float = 2.0,
    dense_depth_conf_threshold: float = 1.0,
    min_support_points: int = 32,
    min_dense_points: int = 256,
    xy_expand_ratio: float = 0.5,
    z_low_expand_ratio: float = 0.25,
    z_high_expand_ratio: float = 0.5,
) -> dict[str, Any]:
    support = extract_object_points_from_geometry(
        mask=mask,
        geometry_result=geometry_result,
        frame_index=frame_index,
        depth_conf_threshold=support_depth_conf_threshold,
    )
    dense = extract_object_points_from_geometry(
        mask=mask,
        geometry_result=geometry_result,
        frame_index=frame_index,
        depth_conf_threshold=dense_depth_conf_threshold,
    )

    support_points = np.asarray(support["points_xyz"], dtype=np.float32)
    dense_points = np.asarray(dense["points_xyz"], dtype=np.float32)
    if (
        support_points.shape[0] < int(min_support_points)
        or dense_points.shape[0] < int(min_dense_points)
        or dense_points.shape[0] <= support_points.shape[0]
    ):
        fallback = dense if dense["num_points"] > 0 else support
        fallback["used_support_guided_filter"] = False
        fallback["support_point_count"] = int(support_points.shape[0])
        fallback["dense_point_count"] = int(dense_points.shape[0])
        fallback["support_depth_conf_threshold"] = float(support_depth_conf_threshold)
        fallback["dense_depth_conf_threshold"] = float(dense_depth_conf_threshold)
        return fallback

    support_colors = support.get("colors_rgb")
    dense_colors = dense.get("colors_rgb")

    filtered_mask = _support_guided_keep_mask(
        support_points=support_points,
        dense_points=dense_points,
        xy_expand_ratio=xy_expand_ratio,
        z_low_expand_ratio=z_low_expand_ratio,
        z_high_expand_ratio=z_high_expand_ratio,
    )
    filtered_points = dense_points[filtered_mask]
    filtered_colors = dense_colors[filtered_mask] if dense_colors is not None else None

    if filtered_points.shape[0] < max(int(min_support_points), support_points.shape[0] // 2):
        fallback = support if support["num_points"] > 0 else dense
        fallback["used_support_guided_filter"] = False
        fallback["support_point_count"] = int(support_points.shape[0])
        fallback["dense_point_count"] = int(dense_points.shape[0])
        fallback["support_depth_conf_threshold"] = float(support_depth_conf_threshold)
        fallback["dense_depth_conf_threshold"] = float(dense_depth_conf_threshold)
        return fallback

    return {
        "frame_index": frame_index,
        "num_points": int(filtered_points.shape[0]),
        "points_xyz": filtered_points.astype(np.float32),
        "colors_rgb": filtered_colors.astype(np.uint8) if filtered_colors is not None else None,
        "mask_resized": dense["mask_resized"],
        "mask_area": dense["mask_area"],
        "valid_mask_area": int(filtered_points.shape[0]),
        "used_depth_conf_threshold": float(dense_depth_conf_threshold),
        "used_support_guided_filter": True,
        "support_point_count": int(support_points.shape[0]),
        "dense_point_count": int(dense_points.shape[0]),
        "support_depth_conf_threshold": float(support_depth_conf_threshold),
        "dense_depth_conf_threshold": float(dense_depth_conf_threshold),
    }


def depth_to_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)

    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]

    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    cam_points = np.stack([x, y, z], axis=-1).astype(np.float32)

    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_points = np.einsum("...i,ij->...j", cam_points - translation, rotation)

    finite_mask = np.isfinite(world_points).all(axis=-1) & np.isfinite(depth_map) & (depth_map > 0)
    if valid_mask is not None:
        finite_mask &= valid_mask.astype(bool)

    return {
        "world_points": world_points,
        "valid_mask": finite_mask,
        "points_xyz": world_points[finite_mask].astype(np.float32),
    }


def pointmap_to_points(
    point_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    point_map = np.asarray(point_map, dtype=np.float32)
    finite_mask = np.isfinite(point_map).all(axis=-1)
    if valid_mask is not None:
        finite_mask &= valid_mask.astype(bool)

    return {
        "world_points": point_map,
        "valid_mask": finite_mask,
        "points_xyz": point_map[finite_mask].astype(np.float32),
    }


def _resize_mask_to_square_grid(mask: np.ndarray, target_size: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask.shape}")

    mask_bool = mask > 0
    height, width = mask_bool.shape
    max_dim = max(height, width)
    top = (max_dim - height) // 2
    left = (max_dim - width) // 2

    square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
    square_mask[top : top + height, left : left + width] = mask_bool.astype(np.uint8)
    resized = cv2.resize(square_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _support_guided_keep_mask(
    support_points: np.ndarray,
    dense_points: np.ndarray,
    *,
    xy_expand_ratio: float,
    z_low_expand_ratio: float,
    z_high_expand_ratio: float,
) -> np.ndarray:
    support_points = np.asarray(support_points, dtype=np.float32)
    dense_points = np.asarray(dense_points, dtype=np.float32)

    support_mins = support_points.min(axis=0)
    support_maxs = support_points.max(axis=0)
    support_range = np.maximum(support_maxs - support_mins, 1e-6)

    keep_mask = np.ones(dense_points.shape[0], dtype=bool)
    for axis in (0, 1):
        lower = support_mins[axis] - xy_expand_ratio * support_range[axis]
        upper = support_maxs[axis] + xy_expand_ratio * support_range[axis]
        keep_mask &= dense_points[:, axis] >= lower
        keep_mask &= dense_points[:, axis] <= upper

    z_values = support_points[:, 2]
    z_q05, z_q25, z_q75, z_q95 = np.quantile(z_values, [0.05, 0.25, 0.75, 0.95])
    z_iqr = max(float(z_q75 - z_q25), 1e-6)
    z_lower = float(z_q05 - z_low_expand_ratio * z_iqr)
    z_upper = float(z_q95 + z_high_expand_ratio * z_iqr)
    keep_mask &= dense_points[:, 2] >= z_lower
    keep_mask &= dense_points[:, 2] <= z_upper

    return keep_mask
