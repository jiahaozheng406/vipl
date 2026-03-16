"""
SAM3 image segmentation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from core.io_utils import save_mask_image, save_overlay_image


def build_sam3_model(
    model_path: str | Path,
    config_path: str | Path,
    reference_script_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    config_path = Path(config_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SAM3 model file does not exist: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"SAM3 config file does not exist: {config_path}")

    model_dir = model_path.parent if model_path.is_file() else model_path
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = Sam3Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Sam3Model.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to(resolved_device)
    model.eval()

    return {
        "model_dir": model_dir,
        "config_path": config_path,
        "reference_script_path": Path(reference_script_path) if reference_script_path else None,
        "device": resolved_device,
        "processor": processor,
        "model": model,
        "score_threshold": 0.5,
        "mask_threshold": 0.5,
    }


def segment_target_in_image(
    image_path: str | Path,
    target_label: str,
    sam3_context: dict[str, Any],
    output_mask_path: str | Path | None = None,
    output_overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)

    processor = sam3_context["processor"]
    model = sam3_context["model"]
    device = sam3_context["device"]
    score_threshold = float(sam3_context.get("score_threshold", 0.5))
    mask_threshold = float(sam3_context.get("mask_threshold", 0.5))
    instance_selection_mode = str(sam3_context.get("instance_selection_mode", "largest")).lower()
    max_instances = max(int(sam3_context.get("max_instances", 1)), 1)
    min_instance_area_px = max(int(sam3_context.get("min_instance_area_px", 0)), 0)
    merge_secondary_min_area_ratio = float(sam3_context.get("merge_secondary_min_area_ratio", 0.12))
    merge_max_center_distance_ratio = float(sam3_context.get("merge_max_center_distance_ratio", 0.35))

    inputs = processor(images=image, text=target_label, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes.tolist(),
    )
    result = processed[0] if processed else {}

    mask_uint8 = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    box_xyxy = None
    score = None
    area_px = 0
    found = False
    selected_instance_indices: list[int] = []

    masks = _to_indexable(result.get("masks"))
    scores = _to_indexable(result.get("scores"))
    boxes = _to_indexable(result.get("boxes"))
    instances = _collect_instances(masks=masks, scores=scores, boxes=boxes)
    selected_instances = _select_instances(
        instances=instances,
        image_shape_hw=image_rgb.shape[:2],
        selection_mode=instance_selection_mode,
        max_instances=max_instances,
        min_instance_area_px=min_instance_area_px,
        merge_secondary_min_area_ratio=merge_secondary_min_area_ratio,
        merge_max_center_distance_ratio=merge_max_center_distance_ratio,
    )
    if selected_instances:
        selected_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        selected_scores: list[float] = []
        selected_instance_indices = [int(item["index"]) for item in selected_instances]
        for item in selected_instances:
            selected_mask |= item["mask_bool"]
            if item["score"] is not None:
                selected_scores.append(float(item["score"]))

        mask_uint8 = (selected_mask.astype(np.uint8) * 255).astype(np.uint8)
        area_px = int(selected_mask.sum())
        score = float(np.mean(selected_scores)) if selected_scores else None
        box_xyxy = _mask_to_box_xyxy(selected_mask)
        found = area_px > 0

    overlay_rgb = _make_overlay(image_rgb, mask_uint8)
    if output_mask_path is not None:
        save_mask_image(mask_uint8, output_mask_path)
    if output_overlay_path is not None:
        save_overlay_image(overlay_rgb, output_overlay_path)

    return {
        "image_path": str(image_path),
        "target_label": target_label,
        "found": found,
        "mask_uint8": mask_uint8,
        "area_px": area_px,
        "score": score,
        "box_xyxy": box_xyxy,
        "overlay_path": str(output_overlay_path) if output_overlay_path is not None else None,
        "mask_path": str(output_mask_path) if output_mask_path is not None else None,
        "num_instances": int(len(instances)),
        "selected_instance_indices": selected_instance_indices,
        "selection_mode": instance_selection_mode,
        "instances": [
            {
                "index": int(item["index"]),
                "area_px": int(item["area_px"]),
                "score": item["score"],
                "box_xyxy": item["box_xyxy"],
                "centroid_xy": item["centroid_xy"],
            }
            for item in instances
        ],
    }


def _collect_instances(
    *,
    masks: list[Any],
    scores: list[Any],
    boxes: list[Any],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        mask_bool = np.asarray(_tensor_to_numpy(mask)).astype(bool)
        area_px = int(mask_bool.sum())
        if area_px < 1:
            continue

        ys, xs = np.nonzero(mask_bool)
        centroid_xy = [float(xs.mean()), float(ys.mean())]
        score = float(_tensor_to_numpy(scores[index])) if index < len(scores) else None
        box_xyxy = None
        if index < len(boxes):
            box_xyxy = np.asarray(_tensor_to_numpy(boxes[index]), dtype=np.float32).tolist()
        if box_xyxy is None:
            box_xyxy = _mask_to_box_xyxy(mask_bool)

        instances.append(
            {
                "index": index,
                "mask_bool": mask_bool,
                "area_px": area_px,
                "score": score,
                "box_xyxy": box_xyxy,
                "centroid_xy": centroid_xy,
            }
        )

    instances.sort(
        key=lambda item: (
            int(item["area_px"]),
            float(item["score"]) if item["score"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return instances


def _select_instances(
    *,
    instances: list[dict[str, Any]],
    image_shape_hw: tuple[int, int],
    selection_mode: str,
    max_instances: int,
    min_instance_area_px: int,
    merge_secondary_min_area_ratio: float,
    merge_max_center_distance_ratio: float,
) -> list[dict[str, Any]]:
    if not instances:
        return []

    eligible = [item for item in instances if int(item["area_px"]) >= int(min_instance_area_px)]
    if not eligible:
        eligible = instances

    if selection_mode in {"urbanverse_primary", "primary"}:
        primary = max(
            eligible,
            key=lambda item: _score_primary_instance(item=item, image_shape_hw=image_shape_hw),
        )
        return [primary]

    primary = eligible[0]
    if selection_mode not in {"merge", "union"} or max_instances == 1:
        return [primary]

    image_scale = float(max(image_shape_hw))
    max_center_distance = max(image_scale * float(merge_max_center_distance_ratio), 1.0)
    min_secondary_area = max(
        int(min_instance_area_px),
        int(round(float(primary["area_px"]) * float(merge_secondary_min_area_ratio))),
    )

    primary_center = np.asarray(primary["centroid_xy"], dtype=np.float32)
    selected: list[dict[str, Any]] = []
    for item in eligible:
        if len(selected) >= int(max_instances):
            break
        if item["index"] != primary["index"] and int(item["area_px"]) < min_secondary_area:
            continue

        center = np.asarray(item["centroid_xy"], dtype=np.float32)
        center_distance = float(np.linalg.norm(center - primary_center))
        if item["index"] != primary["index"] and center_distance > max_center_distance:
            continue
        selected.append(item)

    return selected or [primary]


def _score_primary_instance(
    *,
    item: dict[str, Any],
    image_shape_hw: tuple[int, int],
) -> float:
    height, width = [int(v) for v in image_shape_hw]
    area_norm = float(item["area_px"]) / max(float(height * width), 1.0)
    det_score = float(item["score"]) if item["score"] is not None else 0.0
    cx, cy = [float(v) for v in item["centroid_xy"]]

    target_x = 0.5 * float(width)
    target_y = 0.72 * float(height)
    dx = abs(cx - target_x) / max(0.5 * float(width), 1.0)
    dy = abs(cy - target_y) / max(0.72 * float(height), 1.0)
    center_bias = max(0.0, 1.0 - (0.65 * dx + 0.35 * dy))

    return 0.55 * area_norm + 0.25 * det_score + 0.20 * center_bias


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


def _to_indexable(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value]
        return [value[index] for index in range(value.shape[0])]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value]
        return [value[index] for index in range(value.shape[0])]
    return list(value)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_overlay(image_rgb: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    mask = mask_uint8 > 0
    if np.any(mask):
        color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)
