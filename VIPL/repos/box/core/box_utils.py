"""
Open3D bounding box and visualization helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from core.io_utils import ensure_dir, save_json


LOGGER = logging.getLogger(__name__)


def build_oriented_bbox(points_xyz: np.ndarray, prefer_minimal: bool = False) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points_xyz.shape}")
    if points_xyz.shape[0] < 4:
        raise ValueError("At least 4 points are required to build an oriented bounding box")

    pcd = _build_point_cloud(points_xyz)
    methods = ["minimal", "obb"] if prefer_minimal else ["obb", "minimal"]
    errors: list[str] = []

    for method in methods:
        try:
            bbox = (
                pcd.get_minimal_oriented_bounding_box()
                if method == "minimal"
                else pcd.get_oriented_bounding_box()
            )
            extent = np.asarray(bbox.extent, dtype=np.float32)
            if not np.isfinite(extent).all() or np.any(extent <= 0):
                raise ValueError(f"Degenerate bbox extent: {extent.tolist()}")
            bbox.color = (1.0, 0.0, 0.0)
            return {
                "bbox": bbox,
                "method": method,
                "point_count": int(points_xyz.shape[0]),
            }
        except Exception as exc:
            errors.append(f"{method}: {exc}")

    raise RuntimeError("Failed to build oriented bounding box. " + " | ".join(errors))


def build_single_view_dynamic_bbox(
    points_xyz: np.ndarray,
    extrinsic: np.ndarray,
    *,
    depth_to_width_ratio: float = 1.8,
    min_depth_extent: float = 0.04,
    width_expand_ratio: float = 1.05,
    height_expand_ratio: float = 1.05,
    quantile: float = 0.02,
) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points_xyz.shape}")
    if points_xyz.shape[0] < 4:
        raise ValueError("At least 4 points are required to build a dynamic single-view bounding box")
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected extrinsic shape (3, 4), got {extrinsic.shape}")

    points_cam = _world_to_camera_points(points_xyz, extrinsic)
    basis_cam = _estimate_single_view_basis(points_cam)
    local_points = points_cam @ basis_cam

    low_q = float(np.clip(quantile, 0.0, 0.49))
    high_q = 1.0 - low_q
    mins = np.quantile(local_points, low_q, axis=0).astype(np.float32)
    maxs = np.quantile(local_points, high_q, axis=0).astype(np.float32)

    width_extent = max(float(maxs[0] - mins[0]) * float(width_expand_ratio), 1e-4)
    height_extent = max(float(maxs[1] - mins[1]) * float(height_expand_ratio), 1e-4)
    visible_depth_extent = max(float(maxs[2] - mins[2]), 1e-4)
    target_depth_extent = max(
        float(min_depth_extent),
        visible_depth_extent,
        width_extent * float(depth_to_width_ratio),
    )

    front_plane = float(mins[2] - 0.05 * visible_depth_extent)
    back_plane = float(front_plane + target_depth_extent)
    center_local = np.array(
        [
            0.5 * float(mins[0] + maxs[0]),
            0.5 * float(mins[1] + maxs[1]),
            0.5 * (front_plane + back_plane),
        ],
        dtype=np.float32,
    )

    center_cam = basis_cam @ center_local
    center_world = _camera_to_world_points(center_cam[None, :], extrinsic)[0]
    rotation_world = extrinsic[:, :3].T @ basis_cam

    bbox = o3d.geometry.OrientedBoundingBox(
        center_world.astype(np.float64),
        rotation_world.astype(np.float64),
        np.array([width_extent, height_extent, target_depth_extent], dtype=np.float64),
    )
    bbox.color = (1.0, 0.0, 0.0)
    return {
        "bbox": bbox,
        "method": "single_view_dynamic",
        "point_count": int(points_xyz.shape[0]),
        "width_extent": float(width_extent),
        "height_extent": float(height_extent),
        "visible_depth_extent": float(visible_depth_extent),
        "target_depth_extent": float(target_depth_extent),
        "front_plane_cam": float(front_plane),
        "back_plane_cam": float(back_plane),
    }


def build_single_view_masked_bbox(
    points_xyz: np.ndarray,
    extrinsic: np.ndarray,
    *,
    quantile: float = 0.02,
    axis_padding_ratio: float = 0.03,
    min_extent: float = 0.01,
) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points_xyz.shape}")
    if points_xyz.shape[0] < 4:
        raise ValueError("At least 4 points are required to build a masked single-view bounding box")
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected extrinsic shape (3, 4), got {extrinsic.shape}")

    points_cam = _world_to_camera_points(points_xyz, extrinsic)
    basis_cam = _estimate_single_view_basis(points_cam)
    local_points = points_cam @ basis_cam

    low_q = float(np.clip(quantile, 0.0, 0.2))
    high_q = 1.0 - low_q
    mins = np.quantile(local_points, low_q, axis=0).astype(np.float32)
    maxs = np.quantile(local_points, high_q, axis=0).astype(np.float32)
    extent = np.maximum(maxs - mins, float(min_extent)).astype(np.float32)
    extent *= 1.0 + float(max(axis_padding_ratio, 0.0))

    center_local = (0.5 * (mins + maxs)).astype(np.float32)
    center_cam = basis_cam @ center_local
    center_world = _camera_to_world_points(center_cam[None, :], extrinsic)[0]
    rotation_world = extrinsic[:, :3].T @ basis_cam

    bbox = o3d.geometry.OrientedBoundingBox(
        center_world.astype(np.float64),
        rotation_world.astype(np.float64),
        extent.astype(np.float64),
    )
    bbox.color = (1.0, 0.0, 0.0)
    return {
        "bbox": bbox,
        "method": "single_view_masked_tight",
        "point_count": int(points_xyz.shape[0]),
        "width_extent": float(extent[0]),
        "height_extent": float(extent[1]),
        "visible_depth_extent": float(extent[2]),
    }


def build_urbanverse_vehicle_cuboid_bbox(
    points_xyz: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    mask_resized: np.ndarray,
    *,
    quantile: float = 0.04,
    side_padding_ratio: float = 0.04,
    depth_front_padding_ratio: float = 0.03,
    depth_back_padding_ratio: float = 0.08,
    bottom_padding_ratio: float = 0.04,
    top_padding_ratio: float = 0.10,
    min_height_to_width_ratio: float = 0.78,
    target_height_to_width_ratio: float = 0.86,
    min_depth_to_width_ratio: float = 2.00,
    target_depth_to_width_ratio: float = 2.35,
    max_depth_to_width_ratio: float = 2.80,
    min_extent: float = 0.01,
) -> dict[str, Any]:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    mask_resized = np.asarray(mask_resized)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points_xyz.shape}")
    if points_xyz.shape[0] < 4:
        raise ValueError("At least 4 points are required to build a vehicle cuboid")
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected extrinsic shape (3, 4), got {extrinsic.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {intrinsic.shape}")
    if mask_resized.ndim != 2:
        raise ValueError(f"Expected 2D mask_resized, got {mask_resized.shape}")

    points_cam = _world_to_camera_points(points_xyz, extrinsic)
    basis_cam = _estimate_primary_vehicle_cuboid_basis(points_cam)
    local_points = points_cam @ basis_cam

    low_q = float(np.clip(quantile, 0.0, 0.2))
    high_q = 1.0 - low_q
    mins = np.quantile(local_points, low_q, axis=0).astype(np.float32)
    maxs = np.quantile(local_points, high_q, axis=0).astype(np.float32)
    visible_extent = np.maximum(maxs - mins, float(min_extent)).astype(np.float32)

    visible_width = float(visible_extent[0])
    visible_height = float(visible_extent[1])
    visible_depth = float(visible_extent[2])

    mask_box_xyxy = _mask_to_box_xyxy(mask_resized > 0)
    if mask_box_xyxy is None:
        raise ValueError("mask_resized is empty; cannot estimate vehicle cuboid scale")

    mask_width_px = max(float(mask_box_xyxy[2] - mask_box_xyxy[0] + 1.0), 1.0)
    mask_height_px = max(float(mask_box_xyxy[3] - mask_box_xyxy[1] + 1.0), 1.0)
    median_depth = float(np.median(points_cam[:, 2]))
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    width_from_mask = max(mask_width_px * median_depth / fx, float(min_extent))
    height_from_mask = max(mask_height_px * median_depth / fy, float(min_extent))

    width_extent = max(
        visible_width * (1.0 + float(side_padding_ratio) * 2.0),
        width_from_mask * 0.92,
        float(min_extent),
    )
    height_extent = max(
        visible_height * (1.0 + float(bottom_padding_ratio) + float(top_padding_ratio)),
        height_from_mask * 0.94,
        width_extent * float(min_height_to_width_ratio),
        width_extent * float(target_height_to_width_ratio),
        float(min_extent),
    )

    depth_from_prior = width_extent * float(target_depth_to_width_ratio)
    depth_extent = max(
        visible_depth * (1.0 + float(depth_front_padding_ratio) + float(depth_back_padding_ratio)),
        depth_from_prior,
        width_extent * float(min_depth_to_width_ratio),
        float(min_extent),
    )
    depth_extent = min(depth_extent, width_extent * float(max_depth_to_width_ratio))

    u_center = 0.5 * float(mask_box_xyxy[0] + mask_box_xyxy[2])
    v_bottom = float(mask_box_xyxy[3])
    center_depth = max(median_depth + 0.18 * depth_extent, float(min_extent))
    x_center_cam = (u_center - float(intrinsic[0, 2])) * center_depth / fx
    y_bottom_cam = (v_bottom - float(intrinsic[1, 2])) * center_depth / fy
    center_cam = np.array(
        [
            x_center_cam,
            y_bottom_cam - 0.5 * height_extent,
            center_depth,
        ],
        dtype=np.float32,
    )
    center_local = center_cam @ basis_cam
    z_front = float(center_local[2] - 0.5 * depth_extent)
    z_back = float(center_local[2] + 0.5 * depth_extent)
    extent = np.array(
        [
            max(width_extent, float(min_extent)),
            max(height_extent, float(min_extent)),
            max(depth_extent, float(min_extent)),
        ],
        dtype=np.float32,
    )

    center_world = _camera_to_world_points(center_cam[None, :], extrinsic)[0]
    rotation_world = extrinsic[:, :3].T @ basis_cam

    bbox = o3d.geometry.OrientedBoundingBox(
        center_world.astype(np.float64),
        rotation_world.astype(np.float64),
        extent.astype(np.float64),
    )
    bbox.color = (1.0, 0.0, 0.0)
    return {
        "bbox": bbox,
        "method": "urbanverse_vehicle_cuboid",
        "point_count": int(points_xyz.shape[0]),
        "width_extent": float(extent[0]),
        "height_extent": float(extent[1]),
        "visible_depth_extent": float(visible_depth),
        "target_depth_extent": float(extent[2]),
        "front_plane_cam": float(z_front),
        "back_plane_cam": float(z_back),
        "mask_box_xyxy": [float(v) for v in mask_box_xyxy],
        "estimated_width_from_mask": float(width_from_mask),
        "estimated_height_from_mask": float(height_from_mask),
        "estimated_depth_from_prior": float(depth_from_prior),
    }


def save_bbox_info(bbox: dict[str, Any], save_path: str | Path) -> Path:
    bbox_geom = bbox["bbox"] if isinstance(bbox, dict) else bbox
    method = bbox.get("method") if isinstance(bbox, dict) else "unknown"
    point_count = bbox.get("point_count") if isinstance(bbox, dict) else None

    center = np.asarray(bbox_geom.center, dtype=np.float32)
    extent = np.asarray(bbox_geom.extent, dtype=np.float32)
    rotation = np.asarray(bbox_geom.R, dtype=np.float32)
    corners = np.asarray(bbox_geom.get_box_points(), dtype=np.float32)
    payload = {
        "center": center.tolist(),
        "extent": extent.tolist(),
        "R": rotation.tolist(),
        "corner_points": corners.tolist(),
        "volume": float(np.prod(extent)),
        "method": method,
        "point_count": point_count,
    }
    for key in (
        "width_extent",
        "height_extent",
        "visible_depth_extent",
        "target_depth_extent",
        "front_plane_cam",
        "back_plane_cam",
        "frame_name",
        "motion_mode",
        "bbox_source",
        "mask_box_xyxy",
        "estimated_width_from_mask",
        "estimated_height_from_mask",
        "estimated_depth_from_prior",
    ):
        if isinstance(bbox, dict) and key in bbox:
            payload[key] = bbox[key]
    save_json(payload, save_path)
    return Path(save_path)


def project_bbox_to_image(
    bbox: dict[str, Any] | o3d.geometry.OrientedBoundingBox,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size_hw: tuple[int, int],
    inference_resolution: int,
) -> dict[str, Any]:
    bbox_geom = bbox["bbox"] if isinstance(bbox, dict) else bbox
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Expected extrinsic shape (3, 4), got {extrinsic.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {intrinsic.shape}")

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox_geom)
    world_points = np.asarray(line_set.points, dtype=np.float32)
    lines = np.asarray(line_set.lines, dtype=np.int32)
    camera_points = world_points @ extrinsic[:, :3].T + extrinsic[:, 3]

    valid_mask = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    projected_points_square = np.full((world_points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid_mask):
        pixel_h = camera_points[valid_mask] @ intrinsic.T
        projected_points_square[valid_mask] = pixel_h[:, :2] / pixel_h[:, 2:3]

    bbox_center_world = np.asarray(bbox_geom.center, dtype=np.float32)[None, :]
    bbox_center_camera = bbox_center_world @ extrinsic[:, :3].T + extrinsic[:, 3]
    bbox_center_square = np.full((1, 2), np.nan, dtype=np.float32)
    bbox_center_valid = bool(np.isfinite(bbox_center_camera).all() and bbox_center_camera[0, 2] > 1e-6)
    if bbox_center_valid:
        bbox_center_pixel_h = bbox_center_camera @ intrinsic.T
        bbox_center_square[0] = (bbox_center_pixel_h[:, :2] / bbox_center_pixel_h[:, 2:3])[0]

    projected_points_xy = _map_square_points_to_original_image(
        projected_points_square,
        image_size_hw=image_size_hw,
        square_size=int(inference_resolution),
    )
    bbox_center_xy = _map_square_points_to_original_image(
        bbox_center_square,
        image_size_hw=image_size_hw,
        square_size=int(inference_resolution),
    )[0]
    image_visible_mask = _compute_image_visible_mask(projected_points_xy, image_size_hw) & valid_mask
    bbox_center_visible = bool(
        _compute_image_visible_mask(bbox_center_xy[None, :], image_size_hw)[0] and bbox_center_valid
    )

    return {
        "world_points": world_points,
        "camera_points": camera_points,
        "lines": lines,
        "valid_mask": valid_mask,
        "image_visible_mask": image_visible_mask,
        "projected_points_square": projected_points_square,
        "projected_points_xy": projected_points_xy,
        "bbox_center_camera": bbox_center_camera[0],
        "bbox_center_xy": bbox_center_xy,
        "bbox_center_visible": bbox_center_visible,
    }


def draw_projected_bbox_on_image(
    image_path: str | Path,
    bbox: dict[str, Any] | o3d.geometry.OrientedBoundingBox,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size_hw: tuple[int, int],
    inference_resolution: int,
    output_image_path: str | Path | None = None,
    label: str | None = None,
    line_color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image for bbox projection: {image_path}")

    projection = project_bbox_to_image(
        bbox=bbox,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_size_hw=image_size_hw,
        inference_resolution=inference_resolution,
    )

    overlay = image_bgr.copy()
    image_rect = (0, 0, int(image_bgr.shape[1]), int(image_bgr.shape[0]))
    drawn_lines = 0
    for start_idx, end_idx in projection["lines"]:
        if not (projection["valid_mask"][start_idx] and projection["valid_mask"][end_idx]):
            continue
        pt1 = tuple(np.round(projection["projected_points_xy"][start_idx]).astype(np.int32).tolist())
        pt2 = tuple(np.round(projection["projected_points_xy"][end_idx]).astype(np.int32).tolist())
        clipped, clipped_pt1, clipped_pt2 = cv2.clipLine(image_rect, pt1, pt2)
        if not clipped:
            continue
        cv2.line(overlay, clipped_pt1, clipped_pt2, line_color_bgr, thickness, cv2.LINE_AA)
        drawn_lines += 1

    visible_corners = 0

    saved = False
    output_path = None
    if output_image_path is not None:
        output_path = Path(output_image_path)
        ensure_dir(output_path.parent)
        saved = bool(cv2.imwrite(str(output_path), overlay))

    return {
        "saved": saved,
        "output_path": str(output_path) if output_path is not None else None,
        "num_lines_drawn": int(drawn_lines),
        "num_visible_corners": int(visible_corners),
        "projection": projection,
    }


def visualize_pointcloud_and_bbox(
    points_xyz: np.ndarray,
    bbox: dict[str, Any] | o3d.geometry.OrientedBoundingBox | None = None,
    output_image_path: str | Path | None = None,
    colors_rgb: np.ndarray | None = None,
    show_viewer: bool = False,
    width: int = 1280,
    height: int = 720,
) -> bool:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        LOGGER.warning("Skipping visualization because the point cloud is empty")
        return False

    pcd = _build_point_cloud(points_xyz, colors_rgb=colors_rgb)
    bbox_geom = bbox["bbox"] if isinstance(bbox, dict) else bbox

    visualizer = o3d.visualization.Visualizer()
    try:
        visualizer.create_window(
            window_name="sam_vggt_box",
            width=width,
            height=height,
            visible=show_viewer,
        )
        visualizer.add_geometry(pcd)
        if bbox_geom is not None:
            visualizer.add_geometry(bbox_geom)

        render_option = visualizer.get_render_option()
        render_option.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        render_option.point_size = 2.0

        visualizer.poll_events()
        visualizer.update_renderer()
        if show_viewer:
            visualizer.run()

        if output_image_path is not None:
            output_image_path = Path(output_image_path)
            ensure_dir(output_image_path.parent)
            visualizer.capture_screen_image(str(output_image_path), do_render=True)

        return True
    except Exception as exc:
        LOGGER.warning("Open3D visualization failed: %s", exc)
        return False
    finally:
        visualizer.destroy_window()


def _build_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    if colors_rgb is not None and len(colors_rgb) == len(points_xyz):
        colors = np.asarray(colors_rgb, dtype=np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    return pcd


def _world_to_camera_points(points_xyz: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    return points_xyz @ extrinsic[:, :3].T + extrinsic[:, 3]


def _camera_to_world_points(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    return (points_cam - extrinsic[:, 3]) @ extrinsic[:, :3]


def _estimate_single_view_basis(points_cam: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float32)
    centered = points_cam - np.median(points_cam, axis=0, keepdims=True)
    xz = centered[:, [0, 2]]

    if xz.shape[0] < 2 or not np.isfinite(xz).all():
        side_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        covariance = np.cov(xz.T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        candidate_axes = [
            np.array([eigvecs[0, 0], 0.0, eigvecs[1, 0]], dtype=np.float32),
            np.array([eigvecs[0, 1], 0.0, eigvecs[1, 1]], dtype=np.float32),
        ]
        candidate_axes = [axis / max(float(np.linalg.norm(axis)), 1e-8) for axis in candidate_axes]
        side_axis = max(candidate_axes, key=lambda axis: abs(float(axis[0])))
        if abs(float(side_axis[0])) < 1e-4:
            side_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if float(side_axis[0]) < 0.0:
        side_axis = -side_axis

    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    depth_axis = np.cross(side_axis, up_axis).astype(np.float32)
    depth_norm = float(np.linalg.norm(depth_axis))
    if depth_norm < 1e-8:
        depth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        depth_axis /= depth_norm

    if float(depth_axis[2]) < 0.0:
        depth_axis = -depth_axis
        side_axis = -side_axis

    basis_cam = np.stack([side_axis, up_axis, depth_axis], axis=1).astype(np.float32)
    return basis_cam


def _estimate_vehicle_cuboid_basis(points_cam: np.ndarray) -> np.ndarray:
    basis_cam = _estimate_single_view_basis(points_cam)
    local_points = np.asarray(points_cam, dtype=np.float32) @ basis_cam
    x_extent = float(np.quantile(local_points[:, 0], 0.95) - np.quantile(local_points[:, 0], 0.05))
    z_extent = float(np.quantile(local_points[:, 2], 0.95) - np.quantile(local_points[:, 2], 0.05))

    if z_extent > x_extent:
        side_axis = basis_cam[:, 2].copy()
        depth_axis = basis_cam[:, 0].copy()
    else:
        side_axis = basis_cam[:, 0].copy()
        depth_axis = basis_cam[:, 2].copy()

    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-8)
    depth_axis = np.cross(side_axis, up_axis).astype(np.float32)
    depth_axis = depth_axis / max(float(np.linalg.norm(depth_axis)), 1e-8)
    if float(depth_axis[2]) < 0.0:
        depth_axis = -depth_axis
        side_axis = -side_axis

    return np.stack([side_axis, up_axis, depth_axis], axis=1).astype(np.float32)


def _estimate_primary_vehicle_cuboid_basis(points_cam: np.ndarray) -> np.ndarray:
    pca_basis = _estimate_vehicle_cuboid_basis(points_cam)
    pca_side = np.asarray(pca_basis[:, 0], dtype=np.float32)

    side_axis = np.array(
        [
            0.85 + 0.15 * float(pca_side[0]),
            0.0,
            0.15 * float(pca_side[2]),
        ],
        dtype=np.float32,
    )
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-8)
    if float(side_axis[0]) < 0.0:
        side_axis = -side_axis

    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    depth_axis = np.cross(side_axis, up_axis).astype(np.float32)
    depth_axis = depth_axis / max(float(np.linalg.norm(depth_axis)), 1e-8)
    if float(depth_axis[2]) < 0.0:
        depth_axis = -depth_axis
        side_axis = -side_axis

    return np.stack([side_axis, up_axis, depth_axis], axis=1).astype(np.float32)


def _mask_to_box_xyxy(mask_bool: np.ndarray) -> list[float] | None:
    mask_bool = np.asarray(mask_bool, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    ]


def _map_square_points_to_original_image(
    points_xy: np.ndarray,
    image_size_hw: tuple[int, int],
    square_size: int,
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    height, width = [int(value) for value in image_size_hw]
    max_dim = max(height, width)
    top = (max_dim - height) / 2.0
    left = (max_dim - width) / 2.0
    scale = float(max_dim) / float(square_size)

    mapped = np.full_like(points_xy, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(points_xy).all(axis=1)
    if not np.any(finite_mask):
        return mapped

    mapped[finite_mask] = points_xy[finite_mask] * scale
    mapped[finite_mask, 0] -= left
    mapped[finite_mask, 1] -= top
    return mapped


def _compute_image_visible_mask(
    points_xy: np.ndarray,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    height, width = [int(value) for value in image_size_hw]
    finite_mask = np.isfinite(points_xy).all(axis=1)
    return (
        finite_mask
        & (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= max(width - 1, 0))
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= max(height - 1, 0))
    )


def _compute_label_anchor(
    points_xy: np.ndarray,
    image_size_hw: tuple[int, int],
) -> tuple[int, int] | None:
    finite_mask = np.isfinite(points_xy).all(axis=1)
    if not np.any(finite_mask):
        return None

    valid_points = points_xy[finite_mask]
    height, width = [int(value) for value in image_size_hw]
    x = int(np.clip(valid_points[:, 0].min(), 0, max(width - 1, 0)))
    y = int(np.clip(valid_points[:, 1].min() - 8, 16, max(height - 1, 16)))
    return x, y


def _build_face_polygon(points_xy: np.ndarray, face_indices: np.ndarray) -> np.ndarray | None:
    face_points = np.asarray(points_xy[np.asarray(face_indices, dtype=np.int32)], dtype=np.float32)
    if face_points.shape != (4, 2) or not np.isfinite(face_points).all():
        return None
    centroid = np.mean(face_points, axis=0)
    angles = np.arctan2(face_points[:, 1] - centroid[1], face_points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return np.round(face_points[order]).astype(np.int32)


def _blend_polygon(
    image_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    polygon_xy = np.asarray(polygon_xy, dtype=np.int32)
    if polygon_xy.ndim != 2 or polygon_xy.shape[0] < 3:
        return image_bgr
    fill_layer = image_bgr.copy()
    cv2.fillConvexPoly(fill_layer, polygon_xy, color_bgr, lineType=cv2.LINE_AA)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(fill_layer, alpha_clamped, image_bgr, 1.0 - alpha_clamped, 0.0)


def _draw_stroked_line(
    image_bgr: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.line(
        image_bgr,
        pt1,
        pt2,
        (12, 24, 24),
        max(1, int(thickness) + 2),
        cv2.LINE_AA,
    )
    cv2.line(
        image_bgr,
        pt1,
        pt2,
        color_bgr,
        max(1, int(thickness)),
        cv2.LINE_AA,
    )


def _draw_stroked_circle(
    image_bgr: np.ndarray,
    *,
    center: tuple[int, int],
    radius: int,
    color_bgr: tuple[int, int, int],
) -> None:
    cv2.circle(
        image_bgr,
        center,
        max(1, int(radius) + 2),
        (12, 24, 24),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        image_bgr,
        center,
        max(1, int(radius)),
        color_bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def _draw_label(
    image_bgr: np.ndarray,
    *,
    text: str,
    anchor_xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    x, y = anchor_xy
    top_left = (x, max(0, y - text_height - baseline - 8))
    bottom_right = (x + text_width + 12, y + 4)
    label_layer = image_bgr.copy()
    cv2.rectangle(label_layer, top_left, bottom_right, (18, 30, 30), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(label_layer, top_left, bottom_right, color_bgr, thickness=1, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(label_layer, 0.78, image_bgr, 0.22, 0.0)
    image_bgr[:] = blended
    cv2.putText(
        image_bgr,
        text,
        (x + 6, y - 4),
        font,
        font_scale,
        color_bgr,
        text_thickness,
        cv2.LINE_AA,
    )
