"""
VGGT model loading and geometry inference helpers.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


def build_vggt_model(
    model_path: str | Path,
    config_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    model_path = Path(model_path)
    config_path = Path(config_path)
    if not model_path.exists():
        raise FileNotFoundError(f"VGGT model file does not exist: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"VGGT config file does not exist: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_root = model_path.parent
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if resolved_device.type == "cuda":
        capability = torch.cuda.get_device_capability(resolved_device)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16

    model = VGGT(
        img_size=int(config.get("img_size", 518)),
        patch_size=int(config.get("patch_size", 14)),
        embed_dim=int(config.get("embed_dim", 1024)),
    )
    state_dict = load_file(str(model_path))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"VGGT state_dict mismatch. missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )
    model = model.to(resolved_device)
    model.eval()

    return {
        "model_root": model_root,
        "config": config,
        "device": resolved_device,
        "dtype": dtype,
        "model": model,
        "load_and_preprocess_images_square": load_and_preprocess_images_square,
        "pose_encoding_to_extri_intri": pose_encoding_to_extri_intri,
        "unproject_depth_map_to_point_map": unproject_depth_map_to_point_map,
        "img_load_resolution": 1024,
        "inference_resolution": int(config.get("img_size", 518)),
    }


def infer_vggt_geometry(
    image_paths: list[str | Path],
    vggt_context: dict[str, Any],
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("infer_vggt_geometry requires at least one image path")

    image_paths = [str(Path(path)) for path in image_paths]
    load_images = vggt_context["load_and_preprocess_images_square"]
    model = vggt_context["model"]
    device = vggt_context["device"]
    dtype = vggt_context["dtype"]
    inference_resolution = int(vggt_context["inference_resolution"])
    img_load_resolution = int(vggt_context["img_load_resolution"])
    pose_encoding_to_extri_intri = vggt_context["pose_encoding_to_extri_intri"]
    unproject_depth_map_to_point_map = vggt_context["unproject_depth_map_to_point_map"]

    images_1024, _ = load_images(image_paths, target_size=img_load_resolution)
    image_sizes_hw = [_read_image_size_hw(path) for path in image_paths]
    images_for_color = F.interpolate(
        images_1024,
        size=(inference_resolution, inference_resolution),
        mode="bilinear",
        align_corners=False,
    )
    input_images_rgb = (
        np.clip(images_for_color.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0.0, 255.0)
        .round()
        .astype(np.uint8)
    )

    images_device = images_1024.to(device)
    images_518_device = F.interpolate(
        images_device,
        size=(inference_resolution, inference_resolution),
        mode="bilinear",
        align_corners=False,
    )

    autocast_context = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    with torch.inference_mode():
        with autocast_context:
            batched_images = images_518_device.unsqueeze(0)
            aggregated_tokens_list, patch_start_idx = model.aggregator(batched_images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched_images.shape[-2:])
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list,
                images=batched_images,
                patch_start_idx=patch_start_idx,
            )

    extrinsics_np = extrinsic.squeeze(0).detach().cpu().numpy().astype(np.float32)
    intrinsics_np = intrinsic.squeeze(0).detach().cpu().numpy().astype(np.float32)
    depth_map_np = depth_map.squeeze(0).detach().cpu().numpy().astype(np.float32)
    depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy().astype(np.float32)
    world_points_np = unproject_depth_map_to_point_map(depth_map_np, extrinsics_np, intrinsics_np).astype(np.float32)

    return {
        "image_paths": image_paths,
        "image_sizes_hw": image_sizes_hw,
        "extrinsics": extrinsics_np,
        "intrinsics": intrinsics_np,
        "depth_maps": depth_map_np[..., 0],
        "depth_conf": depth_conf_np,
        "world_points_from_depth": world_points_np,
        "inference_resolution": inference_resolution,
        "input_images_rgb": input_images_rgb,
    }


def _read_image_size_hw(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width
