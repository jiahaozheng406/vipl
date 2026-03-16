"""
Top-level Real2Sim pipeline for single-object 3D extraction from a video.
Dynamic objects use per-frame bbox. Static objects can use merged bbox.
"""

from __future__ import annotations

import gc
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.box_utils import (
    build_oriented_bbox,
    build_urbanverse_vehicle_cuboid_bbox,
    draw_projected_bbox_on_image,
    save_bbox_info,
    visualize_pointcloud_and_bbox,
)
from core.geometry_utils import (
    extract_object_points_from_geometry,
    extract_object_points_with_support_filter,
)
from core.io_utils import build_output_layout, save_depth_artifacts, save_json, save_pointcloud_ply
from core.pointcloud_utils import clean_point_cloud, merge_point_sets, refine_masked_object_point_cloud
from core.sam3_infer import build_sam3_model, segment_target_in_image
from core.vggt_infer import build_vggt_model, infer_vggt_geometry
from core.video_utils import sample_video_frames

VIDEO_PATH = r"E:\vipl\VIPL\datadets_cityscapes\Open_video_page.mp4"
VGGT_MODEL_PATH = r"E:\vipl\VIPL\models\vggt\model.safetensors"
VGGT_CONFIG_PATH = r"E:\vipl\VIPL\models\vggt\config.json"
SAM3_MODEL_PATH = r"E:\vipl\VIPL\models\sam3\sam3.pt"
SAM3_CONFIG_PATH = r"E:\vipl\VIPL\models\sam3\config.json"
SAM3_REFERENCE_SCRIPT = r"E:\vipl\VIPL\repos\parser\run_cityscapes_ground_sam3.py"
OUTPUT_ROOT = r"E:\vipl\VIPL\repos\box\outputs\sam_vggt_box_car"
TARGET_LABEL = "car"
NUM_FRAMES = 5
OBJECT_MOTION_MODE = "dynamic"  # dynamic / static

FRAME_START_SECONDS = 0.5
FRAME_MAX_SECONDS = 0.8
SAM_SCORE_THRESH = 0.5
SAM_MASK_THRESH = 0.75
SAM_INSTANCE_SELECTION_MODE = "urbanverse_primary"
SAM_MAX_INSTANCES = 1
SAM_MIN_INSTANCE_AREA_PX = 96
SAM_MERGE_SECONDARY_MIN_AREA_RATIO = 0.08
SAM_MERGE_MAX_CENTER_DISTANCE_RATIO = 0.28
DEPTH_CONF_THRESH = 5.0
SUPPORT_DEPTH_CONF_THRESH = 2.0
DENSE_DEPTH_CONF_THRESH = 1.0
USE_SUPPORT_GUIDED_FILTER = True
MIN_SUPPORT_POINTS = 32
MIN_DENSE_POINTS = 256
XY_EXPAND_RATIO = 0.2
Z_LOW_EXPAND_RATIO = 0.1
Z_HIGH_EXPAND_RATIO = 0.2
VOXEL_SIZE = 0.005
REFINE_DEPTH_QUANTILE = 0.02
REFINE_CLUSTER_EPS = 0.035
REFINE_MIN_CLUSTER_POINTS = 64
SHOW_VIEWER = False
PREFER_MINIMAL_BOX = False
SAVE_RGB_BBOX_OVERLAYS = True
VEHICLE_CUBOID_QUANTILE = 0.04
VEHICLE_CUBOID_SIDE_PADDING_RATIO = 0.05
VEHICLE_CUBOID_FRONT_PADDING_RATIO = 0.04
VEHICLE_CUBOID_BACK_PADDING_RATIO = 0.10
VEHICLE_CUBOID_BOTTOM_PADDING_RATIO = 0.04
VEHICLE_CUBOID_TOP_PADDING_RATIO = 0.10
VEHICLE_CUBOID_MIN_HEIGHT_TO_WIDTH_RATIO = 0.78
VEHICLE_CUBOID_TARGET_HEIGHT_TO_WIDTH_RATIO = 0.86
VEHICLE_CUBOID_MIN_DEPTH_TO_WIDTH_RATIO = 2.00
VEHICLE_CUBOID_TARGET_DEPTH_TO_WIDTH_RATIO = 2.35
VEHICLE_CUBOID_MAX_DEPTH_TO_WIDTH_RATIO = 2.80

LOGGER = logging.getLogger("sam_vggt_box_car")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_required_paths() -> None:
    required_paths = {
        "VIDEO_PATH": VIDEO_PATH,
        "VGGT_MODEL_PATH": VGGT_MODEL_PATH,
        "VGGT_CONFIG_PATH": VGGT_CONFIG_PATH,
        "SAM3_MODEL_PATH": SAM3_MODEL_PATH,
        "SAM3_CONFIG_PATH": SAM3_CONFIG_PATH,
        "SAM3_REFERENCE_SCRIPT": SAM3_REFERENCE_SCRIPT,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def log_runtime_context() -> None:
    LOGGER.info("Python executable: %s", sys.executable)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
        LOGGER.info("CUDA capability: %s", torch.cuda.get_device_capability(0))


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_with_threshold_fallback(
    *,
    mask: np.ndarray,
    geometry_result: dict[str, Any],
    frame_index: int,
    base_threshold: float,
) -> dict[str, Any]:
    thresholds = [float(base_threshold)]
    for candidate in (2.0, 1.0, 0.5, 0.1):
        if candidate < base_threshold and candidate not in thresholds:
            thresholds.append(candidate)

    last_extraction = None
    for threshold in thresholds:
        extraction = extract_object_points_from_geometry(
            mask=mask,
            geometry_result=geometry_result,
            frame_index=frame_index,
            depth_conf_threshold=threshold,
        )
        last_extraction = extraction
        if extraction["num_points"] > 0:
            return extraction
    return last_extraction


def extract_object_points_for_box(
    *,
    mask: np.ndarray,
    geometry_result: dict[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    if USE_SUPPORT_GUIDED_FILTER:
        extraction = extract_object_points_with_support_filter(
            mask=mask,
            geometry_result=geometry_result,
            frame_index=frame_index,
            support_depth_conf_threshold=SUPPORT_DEPTH_CONF_THRESH,
            dense_depth_conf_threshold=DENSE_DEPTH_CONF_THRESH,
            min_support_points=MIN_SUPPORT_POINTS,
            min_dense_points=MIN_DENSE_POINTS,
            xy_expand_ratio=XY_EXPAND_RATIO,
            z_low_expand_ratio=Z_LOW_EXPAND_RATIO,
            z_high_expand_ratio=Z_HIGH_EXPAND_RATIO,
        )
        if extraction["num_points"] > 0:
            return extraction

    return extract_with_threshold_fallback(
        mask=mask,
        geometry_result=geometry_result,
        frame_index=frame_index,
        base_threshold=DEPTH_CONF_THRESH,
    )


def build_base_summary() -> dict[str, Any]:
    output_root = Path(OUTPUT_ROOT)
    return {
        "status": "running",
        "input_mode": "video",
        "target_label": TARGET_LABEL,
        "timings": {},
        "config": {
            "video_path": VIDEO_PATH,
            "output_root": OUTPUT_ROOT,
            "num_frames": NUM_FRAMES,
            "object_motion_mode": OBJECT_MOTION_MODE,
            "frame_start_seconds": FRAME_START_SECONDS,
            "frame_max_seconds": FRAME_MAX_SECONDS,
            "sam_score_thresh": SAM_SCORE_THRESH,
            "sam_mask_thresh": SAM_MASK_THRESH,
            "sam_instance_selection_mode": SAM_INSTANCE_SELECTION_MODE,
            "sam_max_instances": SAM_MAX_INSTANCES,
            "sam_min_instance_area_px": SAM_MIN_INSTANCE_AREA_PX,
            "sam_merge_secondary_min_area_ratio": SAM_MERGE_SECONDARY_MIN_AREA_RATIO,
            "sam_merge_max_center_distance_ratio": SAM_MERGE_MAX_CENTER_DISTANCE_RATIO,
            "depth_conf_thresh": DEPTH_CONF_THRESH,
            "support_depth_conf_thresh": SUPPORT_DEPTH_CONF_THRESH,
            "dense_depth_conf_thresh": DENSE_DEPTH_CONF_THRESH,
            "use_support_guided_filter": USE_SUPPORT_GUIDED_FILTER,
            "voxel_size": VOXEL_SIZE,
            "refine_depth_quantile": REFINE_DEPTH_QUANTILE,
            "refine_cluster_eps": REFINE_CLUSTER_EPS,
            "refine_min_cluster_points": REFINE_MIN_CLUSTER_POINTS,
            "vehicle_cuboid_quantile": VEHICLE_CUBOID_QUANTILE,
            "vehicle_cuboid_side_padding_ratio": VEHICLE_CUBOID_SIDE_PADDING_RATIO,
            "vehicle_cuboid_front_padding_ratio": VEHICLE_CUBOID_FRONT_PADDING_RATIO,
            "vehicle_cuboid_back_padding_ratio": VEHICLE_CUBOID_BACK_PADDING_RATIO,
            "vehicle_cuboid_bottom_padding_ratio": VEHICLE_CUBOID_BOTTOM_PADDING_RATIO,
            "vehicle_cuboid_top_padding_ratio": VEHICLE_CUBOID_TOP_PADDING_RATIO,
            "vehicle_cuboid_min_height_to_width_ratio": VEHICLE_CUBOID_MIN_HEIGHT_TO_WIDTH_RATIO,
            "vehicle_cuboid_target_height_to_width_ratio": VEHICLE_CUBOID_TARGET_HEIGHT_TO_WIDTH_RATIO,
            "vehicle_cuboid_min_depth_to_width_ratio": VEHICLE_CUBOID_MIN_DEPTH_TO_WIDTH_RATIO,
            "vehicle_cuboid_target_depth_to_width_ratio": VEHICLE_CUBOID_TARGET_DEPTH_TO_WIDTH_RATIO,
            "vehicle_cuboid_max_depth_to_width_ratio": VEHICLE_CUBOID_MAX_DEPTH_TO_WIDTH_RATIO,
        },
        "selected_inputs": [],
        "sam3": {"frames": [], "valid_mask_frames": 0},
        "geometry": {},
        "pointcloud": {"frames": []},
        "bbox": {},
        "artifacts": {
            "output_root": str(output_root),
            "run_summary_path": str(output_root / "run_summary.json"),
        },
    }


def summarize_bbox_result(bbox_result: dict[str, Any]) -> dict[str, Any]:
    bbox_geom = bbox_result["bbox"]
    summary = {
        "status": "completed",
        "method": bbox_result["method"],
        "point_count": bbox_result["point_count"],
        "center": np.asarray(bbox_geom.center, dtype=np.float32).tolist(),
        "extent": np.asarray(bbox_geom.extent, dtype=np.float32).tolist(),
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
        if key in bbox_result:
            summary[key] = bbox_result[key]
    return summary


def build_dynamic_frame_bbox(
    *,
    fit_points_xyz: np.ndarray,
    mask_resized: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    frame_name: str,
) -> dict[str, Any]:
    bbox_result = build_urbanverse_vehicle_cuboid_bbox(
        points_xyz=fit_points_xyz,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        mask_resized=mask_resized,
        quantile=VEHICLE_CUBOID_QUANTILE,
        side_padding_ratio=VEHICLE_CUBOID_SIDE_PADDING_RATIO,
        depth_front_padding_ratio=VEHICLE_CUBOID_FRONT_PADDING_RATIO,
        depth_back_padding_ratio=VEHICLE_CUBOID_BACK_PADDING_RATIO,
        bottom_padding_ratio=VEHICLE_CUBOID_BOTTOM_PADDING_RATIO,
        top_padding_ratio=VEHICLE_CUBOID_TOP_PADDING_RATIO,
        min_height_to_width_ratio=VEHICLE_CUBOID_MIN_HEIGHT_TO_WIDTH_RATIO,
        target_height_to_width_ratio=VEHICLE_CUBOID_TARGET_HEIGHT_TO_WIDTH_RATIO,
        min_depth_to_width_ratio=VEHICLE_CUBOID_MIN_DEPTH_TO_WIDTH_RATIO,
        target_depth_to_width_ratio=VEHICLE_CUBOID_TARGET_DEPTH_TO_WIDTH_RATIO,
        max_depth_to_width_ratio=VEHICLE_CUBOID_MAX_DEPTH_TO_WIDTH_RATIO,
    )
    bbox_result["frame_name"] = frame_name
    bbox_result["motion_mode"] = OBJECT_MOTION_MODE
    bbox_result["bbox_source"] = "urbanverse_inspired_vehicle_cuboid"
    return bbox_result


def copy_best_frame_bbox_artifacts(*, best_json_path: Path, best_projection_path: Path | None, canonical_json_path: Path, canonical_png_path: Path) -> None:
    shutil.copy2(best_json_path, canonical_json_path)
    if best_projection_path is not None and best_projection_path.exists():
        shutil.copy2(best_projection_path, canonical_png_path)


def build_dynamic_bboxes(
    *,
    frame_paths: list[Path],
    per_frame_extractions: list[dict[str, Any]],
    per_frame_refined_sets: list[dict[str, Any] | None],
    per_frame_cleaned_sets: list[dict[str, Any] | None],
    geometry_result: dict[str, Any],
    output_layout: dict[str, Path],
    summary: dict[str, Any],
    bbox_json_path: Path,
    bbox_png_path: Path,
) -> None:
    frame_bbox_entries: list[dict[str, Any]] = []
    best_frame_entry: dict[str, Any] | None = None
    best_frame_score = -1

    for frame_index, frame_path in enumerate(frame_paths):
        frame_stem = Path(frame_path).stem
        extraction = per_frame_extractions[frame_index]
        refined_frame = per_frame_refined_sets[frame_index]
        cleaned_frame = per_frame_cleaned_sets[frame_index]
        fit_frame = (
            refined_frame
            if refined_frame is not None and refined_frame["num_points_after"] >= 4
            else cleaned_frame
        )
        if fit_frame is None or fit_frame["num_points_after"] < 4:
            frame_bbox_entries.append(
                {
                    "frame_name": frame_stem,
                    "status": "failed",
                    "image_path": str(frame_path),
                    "error_message": "Not enough cleaned 3D points for dynamic single-frame bbox",
                }
            )
            continue

        bbox_result = build_dynamic_frame_bbox(
            fit_points_xyz=fit_frame["points_xyz"],
            mask_resized=extraction["mask_resized"],
            extrinsic=geometry_result["extrinsics"][frame_index],
            intrinsic=geometry_result["intrinsics"][frame_index],
            frame_name=frame_stem,
        )
        frame_json_path = output_layout["boxes"] / f"{frame_stem}_{TARGET_LABEL}_bbox.json"
        save_bbox_info(bbox=bbox_result, save_path=frame_json_path)

        frame_bbox_preview = output_layout["boxes"] / f"{frame_stem}_{TARGET_LABEL}_bbox.png"
        visualize_pointcloud_and_bbox(
            points_xyz=fit_frame["points_xyz"],
            bbox=bbox_result,
            output_image_path=frame_bbox_preview,
            colors_rgb=fit_frame.get("colors_rgb"),
            show_viewer=SHOW_VIEWER,
        )

        projection_path = None
        projection_result = None
        if SAVE_RGB_BBOX_OVERLAYS:
            projection_path = output_layout["box_projections"] / f"{frame_stem}_{TARGET_LABEL}_bbox3d.png"
            projection_result = draw_projected_bbox_on_image(
                image_path=frame_path,
                bbox=bbox_result,
                extrinsic=geometry_result["extrinsics"][frame_index],
                intrinsic=geometry_result["intrinsics"][frame_index],
                image_size_hw=geometry_result["image_sizes_hw"][frame_index],
                inference_resolution=geometry_result["inference_resolution"],
                output_image_path=projection_path,
                label=f"{TARGET_LABEL} 3D box",
            )

        frame_entry = summarize_bbox_result(bbox_result)
        frame_entry.update(
            {
                "frame_name": frame_stem,
                "status": "completed",
                "image_path": str(frame_path),
                "json_path": str(frame_json_path),
                "bbox_preview_path": str(frame_bbox_preview),
                "projection_path": str(projection_path) if projection_path is not None else None,
                "fit_num_points": int(fit_frame["num_points_after"]),
                "cleaned_num_points": int(cleaned_frame["num_points_after"]),
                "num_lines_drawn": projection_result["num_lines_drawn"] if projection_result is not None else None,
                "num_visible_corners": projection_result["num_visible_corners"] if projection_result is not None else None,
            }
        )
        frame_bbox_entries.append(frame_entry)

        if int(cleaned_frame["num_points_after"]) > best_frame_score:
            best_frame_score = int(cleaned_frame["num_points_after"])
            best_frame_entry = frame_entry

    if best_frame_entry is None:
        raise RuntimeError("Dynamic motion mode did not produce any valid per-frame bounding boxes.")

    copy_best_frame_bbox_artifacts(
        best_json_path=Path(best_frame_entry["json_path"]),
        best_projection_path=Path(best_frame_entry["projection_path"]) if best_frame_entry["projection_path"] else None,
        canonical_json_path=bbox_json_path,
        canonical_png_path=bbox_png_path,
    )

    summary["bbox"] = {
        "status": "completed",
        "mode": "per_frame_dynamic",
        "best_frame_name": best_frame_entry["frame_name"],
        "best_frame_cleaned_points": best_frame_score,
        "json_path": str(bbox_json_path),
        "image_path": str(bbox_png_path),
        "rgb_projection_frames": frame_bbox_entries,
    }


def build_static_bbox(
    *,
    per_frame_point_sets: list[dict[str, Any]],
    frame_paths: list[Path],
    geometry_result: dict[str, Any],
    output_layout: dict[str, Path],
    summary: dict[str, Any],
    bbox_json_path: Path,
    bbox_png_path: Path,
) -> None:
    merged = merge_point_sets(per_frame_point_sets)
    summary["pointcloud"]["merged_points"] = merged["num_points"]
    if merged["num_points"] < 1:
        raise RuntimeError(f"Object masks were found for '{TARGET_LABEL}', but all extracted point clouds were empty.")

    merged_raw_path = output_layout["pointclouds"] / f"{TARGET_LABEL}_merged_raw.ply"
    save_pointcloud_ply(points_xyz=merged["points_xyz"], output_path=merged_raw_path, colors_rgb=merged["colors_rgb"])

    cleaned = clean_point_cloud(points_xyz=merged["points_xyz"], colors_rgb=merged["colors_rgb"], voxel_size=VOXEL_SIZE)
    summary["pointcloud"]["cleaned_points"] = cleaned["num_points_after"]
    if cleaned["num_points_after"] < 1:
        raise RuntimeError("Point cloud cleaning removed all points. Raw merged point cloud was kept.")

    cleaned_path = output_layout["pointclouds"] / f"{TARGET_LABEL}_merged_cleaned.ply"
    save_pointcloud_ply(points_xyz=cleaned["points_xyz"], output_path=cleaned_path, colors_rgb=cleaned["colors_rgb"])

    bbox_result = build_oriented_bbox(points_xyz=cleaned["points_xyz"], prefer_minimal=PREFER_MINIMAL_BOX)
    save_bbox_info(bbox=bbox_result, save_path=bbox_json_path)
    visualize_pointcloud_and_bbox(
        points_xyz=cleaned["points_xyz"],
        bbox=bbox_result,
        output_image_path=bbox_png_path,
        colors_rgb=cleaned["colors_rgb"],
        show_viewer=SHOW_VIEWER,
    )

    summary["bbox"] = summarize_bbox_result(bbox_result)
    summary["bbox"]["json_path"] = str(bbox_json_path)
    summary["bbox"]["image_path"] = str(bbox_png_path)
    summary["bbox"]["rgb_projection_frames"] = []

    if SAVE_RGB_BBOX_OVERLAYS:
        for frame_index, frame_path in enumerate(frame_paths):
            frame_stem = Path(frame_path).stem
            projection_path = output_layout["box_projections"] / f"{frame_stem}_{TARGET_LABEL}_bbox3d.png"
            try:
                projection_result = draw_projected_bbox_on_image(
                    image_path=frame_path,
                    bbox=bbox_result,
                    extrinsic=geometry_result["extrinsics"][frame_index],
                    intrinsic=geometry_result["intrinsics"][frame_index],
                    image_size_hw=geometry_result["image_sizes_hw"][frame_index],
                    inference_resolution=geometry_result["inference_resolution"],
                    output_image_path=projection_path,
                    label=f"{TARGET_LABEL} 3D box",
                )
                summary["bbox"]["rgb_projection_frames"].append(
                    {
                        "frame_name": frame_stem,
                        "image_path": str(frame_path),
                        "projection_path": projection_result["output_path"],
                        "num_lines_drawn": projection_result["num_lines_drawn"],
                        "num_visible_corners": projection_result["num_visible_corners"],
                    }
                )
            except Exception as projection_exc:
                summary["bbox"]["rgb_projection_frames"].append(
                    {
                        "frame_name": frame_stem,
                        "image_path": str(frame_path),
                        "projection_path": None,
                        "error_type": type(projection_exc).__name__,
                        "error_message": str(projection_exc),
                    }
                )


def run_pipeline() -> None:
    summary = build_base_summary()
    output_layout = build_output_layout(OUTPUT_ROOT)
    summary_path = output_layout["root"] / "run_summary.json"
    pipeline_start_time = time.time()

    try:
        validate_required_paths()
        log_runtime_context()

        stage_start_time = time.time()
        frame_paths, video_metadata = sample_video_frames(
            video_path=VIDEO_PATH,
            output_dir=output_layout["frames"],
            num_frames=NUM_FRAMES,
            start_seconds=FRAME_START_SECONDS,
            max_seconds=FRAME_MAX_SECONDS,
        )
        summary["timings"]["sample_frames_seconds"] = round(time.time() - stage_start_time, 3)
        summary["selected_inputs"] = [
            {
                "source_type": "video_frame",
                "source_path": VIDEO_PATH,
                "source_frame_index": int(source_index),
                "prepared_path": str(frame_path),
            }
            for source_index, frame_path in zip(video_metadata["sampled_frame_indices"], frame_paths)
        ]
        summary["geometry"]["video_metadata"] = video_metadata

        stage_start_time = time.time()
        sam3_context = build_sam3_model(
            model_path=SAM3_MODEL_PATH,
            config_path=SAM3_CONFIG_PATH,
            reference_script_path=SAM3_REFERENCE_SCRIPT,
        )
        sam3_context["score_threshold"] = SAM_SCORE_THRESH
        sam3_context["mask_threshold"] = SAM_MASK_THRESH
        sam3_context["instance_selection_mode"] = SAM_INSTANCE_SELECTION_MODE
        sam3_context["max_instances"] = SAM_MAX_INSTANCES
        sam3_context["min_instance_area_px"] = SAM_MIN_INSTANCE_AREA_PX
        sam3_context["merge_secondary_min_area_ratio"] = SAM_MERGE_SECONDARY_MIN_AREA_RATIO
        sam3_context["merge_max_center_distance_ratio"] = SAM_MERGE_MAX_CENTER_DISTANCE_RATIO

        segmentation_results = []
        valid_mask_frames = 0
        for frame_path in frame_paths:
            frame_start_time = time.time()
            frame_stem = Path(frame_path).stem
            mask_path = output_layout["masks"] / f"{frame_stem}_{TARGET_LABEL}.png"
            overlay_path = output_layout["overlays"] / f"{frame_stem}_{TARGET_LABEL}_overlay.png"
            result = segment_target_in_image(
                image_path=frame_path,
                target_label=TARGET_LABEL,
                sam3_context=sam3_context,
                output_mask_path=mask_path,
                output_overlay_path=overlay_path,
            )
            segmentation_results.append(result)
            summary["sam3"]["frames"].append(
                {
                    "frame_name": frame_stem,
                    "image_path": str(frame_path),
                    "found": result["found"],
                    "area_px": result["area_px"],
                    "score": result["score"],
                    "box_xyxy": result["box_xyxy"],
                    "num_instances": result["num_instances"],
                    "selected_instance_indices": result["selected_instance_indices"],
                    "selection_mode": result["selection_mode"],
                    "elapsed_seconds": round(time.time() - frame_start_time, 3),
                    "mask_path": result["mask_path"],
                    "overlay_path": result["overlay_path"],
                }
            )
            if result["found"]:
                valid_mask_frames += 1

        summary["sam3"]["valid_mask_frames"] = valid_mask_frames
        summary["timings"]["sam3_total_seconds"] = round(time.time() - stage_start_time, 3)
        if valid_mask_frames < 1:
            raise RuntimeError(f"No valid '{TARGET_LABEL}' masks were found in the sampled frames.")

        del sam3_context
        release_cuda_memory()

        stage_start_time = time.time()
        vggt_context = build_vggt_model(
            model_path=VGGT_MODEL_PATH,
            config_path=VGGT_CONFIG_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        geometry_result = infer_vggt_geometry(frame_paths, vggt_context)
        summary["timings"]["vggt_total_seconds"] = round(time.time() - stage_start_time, 3)
        summary["geometry"].update(
            {
                "num_frames": len(frame_paths),
                "inference_resolution": int(geometry_result["inference_resolution"]),
                "depth_artifacts": [],
            }
        )
        for frame_index, frame_path in enumerate(frame_paths):
            frame_stem = Path(frame_path).stem
            depth_npy_path = output_layout["depth"] / f"{frame_stem}_depth.npy"
            depth_preview_path = output_layout["depth"] / f"{frame_stem}_depth.png"
            save_depth_artifacts(
                depth_map=geometry_result["depth_maps"][frame_index],
                output_npy_path=depth_npy_path,
                output_preview_path=depth_preview_path,
            )
            summary["geometry"]["depth_artifacts"].append(
                {
                    "frame_name": frame_stem,
                    "depth_npy_path": str(depth_npy_path),
                    "depth_preview_path": str(depth_preview_path),
                }
            )

        del vggt_context
        release_cuda_memory()

        stage_start_time = time.time()
        per_frame_point_sets = []
        per_frame_extractions: list[dict[str, Any]] = []
        per_frame_refined_sets: list[dict[str, Any] | None] = []
        per_frame_cleaned_sets: list[dict[str, Any] | None] = []
        for frame_index, frame_path in enumerate(frame_paths):
            frame_stem = Path(frame_path).stem
            extraction = extract_object_points_for_box(
                mask=segmentation_results[frame_index]["mask_uint8"],
                geometry_result=geometry_result,
                frame_index=frame_index,
            )
            per_frame_point_sets.append(extraction)
            per_frame_extractions.append(extraction)
            cleaned_frame = None
            refined_frame = None

            pointcloud_path = output_layout["pointclouds"] / f"{frame_stem}_{TARGET_LABEL}.ply"
            refined_pointcloud_path = output_layout["pointclouds"] / f"{frame_stem}_{TARGET_LABEL}_refined.ply"
            cleaned_pointcloud_path = output_layout["pointclouds"] / f"{frame_stem}_{TARGET_LABEL}_cleaned.ply"
            raw_saved = False
            refined_saved = False
            cleaned_saved = False

            if extraction["num_points"] > 0:
                raw_saved = save_pointcloud_ply(
                    points_xyz=extraction["points_xyz"],
                    output_path=pointcloud_path,
                    colors_rgb=extraction["colors_rgb"],
                )
                refined_frame = refine_masked_object_point_cloud(
                    points_xyz=extraction["points_xyz"],
                    colors_rgb=extraction["colors_rgb"],
                    extrinsic=geometry_result["extrinsics"][frame_index],
                    depth_quantile=REFINE_DEPTH_QUANTILE,
                    cluster_eps=REFINE_CLUSTER_EPS,
                    min_cluster_points=REFINE_MIN_CLUSTER_POINTS,
                )
                if refined_frame["num_points_after"] > 0:
                    refined_saved = save_pointcloud_ply(
                        points_xyz=refined_frame["points_xyz"],
                        output_path=refined_pointcloud_path,
                        colors_rgb=refined_frame["colors_rgb"],
                    )
                    cleaned_frame = clean_point_cloud(
                        points_xyz=refined_frame["points_xyz"],
                        colors_rgb=refined_frame["colors_rgb"],
                        voxel_size=VOXEL_SIZE,
                    )
                    if cleaned_frame["num_points_after"] > 0:
                        cleaned_saved = save_pointcloud_ply(
                            points_xyz=cleaned_frame["points_xyz"],
                            output_path=cleaned_pointcloud_path,
                            colors_rgb=cleaned_frame["colors_rgb"],
                        )
                else:
                    cleaned_frame = clean_point_cloud(
                        points_xyz=extraction["points_xyz"],
                        colors_rgb=extraction["colors_rgb"],
                        voxel_size=VOXEL_SIZE,
                    )
                    if cleaned_frame["num_points_after"] > 0:
                        cleaned_saved = save_pointcloud_ply(
                            points_xyz=cleaned_frame["points_xyz"],
                            output_path=cleaned_pointcloud_path,
                            colors_rgb=cleaned_frame["colors_rgb"],
                        )

            per_frame_refined_sets.append(refined_frame)
            per_frame_cleaned_sets.append(cleaned_frame)
            summary["pointcloud"]["frames"].append(
                {
                    "frame_name": frame_stem,
                    "num_points": extraction["num_points"],
                    "mask_area": extraction["mask_area"],
                    "valid_mask_area": extraction["valid_mask_area"],
                    "used_depth_conf_threshold": extraction["used_depth_conf_threshold"],
                    "used_support_guided_filter": extraction.get("used_support_guided_filter", False),
                    "support_point_count": extraction.get("support_point_count"),
                    "dense_point_count": extraction.get("dense_point_count"),
                    "refined_num_points": refined_frame["num_points_after"] if refined_frame is not None else 0,
                    "refined_num_points_after_depth_trim": refined_frame["num_points_after_depth_trim"] if refined_frame is not None else 0,
                    "refined_num_clusters": refined_frame["num_clusters"] if refined_frame is not None else 0,
                    "refined_selected_cluster_index": refined_frame["selected_cluster_index"] if refined_frame is not None else None,
                    "cleaned_num_points": cleaned_frame["num_points_after"] if cleaned_frame is not None else 0,
                    "pointcloud_path": str(pointcloud_path) if raw_saved else None,
                    "refined_pointcloud_path": str(refined_pointcloud_path) if refined_saved else None,
                    "cleaned_pointcloud_path": str(cleaned_pointcloud_path) if cleaned_saved else None,
                }
            )
        summary["timings"]["pointcloud_refine_seconds"] = round(time.time() - stage_start_time, 3)

        bbox_png_path = output_layout["boxes"] / f"{TARGET_LABEL}_bbox.png"
        bbox_json_path = output_layout["boxes"] / f"{TARGET_LABEL}_bbox.json"

        stage_start_time = time.time()
        if OBJECT_MOTION_MODE == "dynamic":
            build_dynamic_bboxes(
                frame_paths=frame_paths,
                per_frame_extractions=per_frame_extractions,
                per_frame_refined_sets=per_frame_refined_sets,
                per_frame_cleaned_sets=per_frame_cleaned_sets,
                geometry_result=geometry_result,
                output_layout=output_layout,
                summary=summary,
                bbox_json_path=bbox_json_path,
                bbox_png_path=bbox_png_path,
            )
        else:
            build_static_bbox(
                per_frame_point_sets=per_frame_point_sets,
                frame_paths=frame_paths,
                geometry_result=geometry_result,
                output_layout=output_layout,
                summary=summary,
                bbox_json_path=bbox_json_path,
                bbox_png_path=bbox_png_path,
            )

        summary["timings"]["bbox_seconds"] = round(time.time() - stage_start_time, 3)
        summary["status"] = "completed"
        summary["timings"]["total_seconds"] = round(time.time() - pipeline_start_time, 3)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary["timings"]["total_seconds"] = round(time.time() - pipeline_start_time, 3)
        raise
    finally:
        save_json(summary, summary_path)


def main() -> int:
    configure_logging()
    try:
        run_pipeline()
        return 0
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
