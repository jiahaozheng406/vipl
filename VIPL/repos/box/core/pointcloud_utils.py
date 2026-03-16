"""
Point cloud merging and cleanup helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d


def merge_point_sets(point_sets: list[Any]) -> dict[str, Any]:
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    all_have_colors = True

    for item in point_sets:
        if item is None:
            continue
        if isinstance(item, dict):
            points_xyz = np.asarray(item.get("points_xyz", np.empty((0, 3), dtype=np.float32)), dtype=np.float32)
            colors_rgb = item.get("colors_rgb")
        else:
            points_xyz = np.asarray(item, dtype=np.float32)
            colors_rgb = None

        if points_xyz.size == 0 or len(points_xyz) == 0:
            continue

        point_chunks.append(points_xyz)
        if colors_rgb is not None and len(colors_rgb) == len(points_xyz):
            color_chunks.append(np.asarray(colors_rgb, dtype=np.uint8))
        else:
            all_have_colors = False

    merged_points = (
        np.concatenate(point_chunks, axis=0).astype(np.float32)
        if point_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    merged_colors = None
    if point_chunks and all_have_colors and len(color_chunks) == len(point_chunks):
        merged_colors = np.concatenate(color_chunks, axis=0).astype(np.uint8)

    return {
        "points_xyz": merged_points,
        "colors_rgb": merged_colors,
        "num_points": int(merged_points.shape[0]),
        "num_point_sets": len(point_chunks),
    }


def refine_masked_object_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
    *,
    extrinsic: np.ndarray | None = None,
    depth_quantile: float = 0.02,
    cluster_eps: float = 0.035,
    min_cluster_points: int = 64,
) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points_xyz.shape}")

    valid_mask = np.isfinite(points_xyz).all(axis=1)
    colors_filtered = None
    if colors_rgb is not None:
        colors_rgb = np.asarray(colors_rgb)
        if len(colors_rgb) == len(points_xyz):
            colors_filtered = colors_rgb[valid_mask]

    points_valid = points_xyz[valid_mask]
    if len(points_valid) == 0:
        return {
            "points_xyz": np.empty((0, 3), dtype=np.float32),
            "colors_rgb": None,
            "num_points_before": int(points_xyz.shape[0]),
            "num_points_after": 0,
            "num_points_after_depth_trim": 0,
            "num_clusters": 0,
            "selected_cluster_index": None,
        }

    if extrinsic is not None:
        points_cam = _world_to_camera_points(points_valid, extrinsic)
    else:
        points_cam = points_valid.copy()

    depth_keep_mask = _depth_trim_mask(points_cam[:, 2], quantile=depth_quantile)
    points_trimmed = points_valid[depth_keep_mask]
    points_trimmed_cam = points_cam[depth_keep_mask]
    colors_trimmed = colors_filtered[depth_keep_mask] if colors_filtered is not None else None

    if len(points_trimmed) == 0:
        points_trimmed = points_valid
        points_trimmed_cam = points_cam
        colors_trimmed = colors_filtered

    cluster_labels = _cluster_points_dbscan(
        points_xyz=points_trimmed_cam,
        eps=cluster_eps,
        min_cluster_points=min_cluster_points,
    )
    selected_cluster_index = None
    num_clusters = 0
    selected_mask = np.ones(len(points_trimmed), dtype=bool)
    if cluster_labels is not None:
        valid_cluster_labels = cluster_labels[cluster_labels >= 0]
        if valid_cluster_labels.size > 0:
            num_clusters = int(valid_cluster_labels.max()) + 1
            selected_cluster_index = _select_best_cluster(cluster_labels, points_trimmed_cam)
            selected_mask = cluster_labels == selected_cluster_index

    refined_points = points_trimmed[selected_mask]
    refined_colors = colors_trimmed[selected_mask] if colors_trimmed is not None else None
    return {
        "points_xyz": refined_points.astype(np.float32),
        "colors_rgb": refined_colors.astype(np.uint8) if refined_colors is not None else None,
        "num_points_before": int(points_xyz.shape[0]),
        "num_points_after": int(refined_points.shape[0]),
        "num_points_after_depth_trim": int(points_trimmed.shape[0]),
        "num_clusters": int(num_clusters),
        "selected_cluster_index": None if selected_cluster_index is None else int(selected_cluster_index),
    }


def clean_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
    voxel_size: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    valid_mask = np.isfinite(points_xyz).all(axis=-1)

    colors_filtered = None
    if colors_rgb is not None:
        colors_rgb = np.asarray(colors_rgb)
        if len(colors_rgb) == len(points_xyz):
            colors_filtered = colors_rgb[valid_mask]

    points_filtered = points_xyz[valid_mask]
    if len(points_filtered) == 0:
        empty_pcd = o3d.geometry.PointCloud()
        return {
            "points_xyz": np.empty((0, 3), dtype=np.float32),
            "colors_rgb": None,
            "o3d_pcd": empty_pcd,
            "num_points_before": int(points_xyz.shape[0]),
            "num_points_after": 0,
        }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float64))
    if colors_filtered is not None:
        colors_float = colors_filtered.astype(np.float32)
        if colors_float.max() > 1.0:
            colors_float /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float, 0.0, 1.0).astype(np.float64))

    if voxel_size > 0 and len(points_filtered) > 1:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    if len(pcd.points) >= max(5, nb_neighbors):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=float(std_ratio))

    cleaned_points = np.asarray(pcd.points, dtype=np.float32)
    cleaned_colors = None
    if len(pcd.colors) == len(cleaned_points):
        cleaned_colors = np.clip(np.asarray(pcd.colors) * 255.0, 0.0, 255.0).astype(np.uint8)

    return {
        "points_xyz": cleaned_points,
        "colors_rgb": cleaned_colors,
        "o3d_pcd": pcd,
        "num_points_before": int(points_xyz.shape[0]),
        "num_points_after": int(cleaned_points.shape[0]),
    }


def _world_to_camera_points(points_xyz: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected extrinsic shape (3, 4), got {extrinsic.shape}")
    return points_xyz @ extrinsic[:, :3].T + extrinsic[:, 3]


def _depth_trim_mask(depth_values: np.ndarray, *, quantile: float) -> np.ndarray:
    depth_values = np.asarray(depth_values, dtype=np.float32)
    finite_mask = np.isfinite(depth_values)
    if finite_mask.sum() < 8:
        return finite_mask

    low_q = float(np.clip(quantile, 0.0, 0.2))
    high_q = 1.0 - low_q
    lower = float(np.quantile(depth_values[finite_mask], low_q))
    upper = float(np.quantile(depth_values[finite_mask], high_q))
    return finite_mask & (depth_values >= lower) & (depth_values <= upper)


def _cluster_points_dbscan(
    *,
    points_xyz: np.ndarray,
    eps: float,
    min_cluster_points: int,
) -> np.ndarray | None:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if len(points_xyz) < max(8, min_cluster_points):
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(max(eps, 1e-4)),
            min_points=int(max(min_cluster_points, 3)),
            print_progress=False,
        ),
        dtype=np.int32,
    )
    return labels if labels.size == len(points_xyz) else None


def _select_best_cluster(cluster_labels: np.ndarray, points_xyz: np.ndarray) -> int:
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    best_label = -1
    best_score = float("-inf")
    for label in np.unique(cluster_labels):
        if label < 0:
            continue
        cluster_points = points_xyz[cluster_labels == label]
        if len(cluster_points) == 0:
            continue
        mean_depth = float(np.mean(cluster_points[:, 2]))
        score = float(len(cluster_points)) / max(mean_depth, 1e-3)
        if score > best_score:
            best_score = score
            best_label = int(label)

    return 0 if best_label < 0 else best_label
