"""
Helpers for preparing a flat image directory as pipeline input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from core.io_utils import ensure_dir


def collect_image_paths(
    image_dir: str | Path,
    output_dir: str | Path,
    max_images: int,
    start_index: int = 0,
    stride: int = 1,
    extensions: Iterable[str] | None = None,
) -> tuple[list[Path], dict[str, object]]:
    image_dir = Path(image_dir)
    output_dir = ensure_dir(output_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {image_dir}")
    if max_images < 1:
        raise ValueError("max_images must be at least 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    normalized_extensions = _normalize_extensions(extensions)
    candidate_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ],
        key=lambda path: (path.name.lower(), path.name),
    )
    if not candidate_paths:
        raise RuntimeError(
            f"No image files found in {image_dir} with extensions {sorted(normalized_extensions)}"
        )

    selected_source_paths = candidate_paths[start_index::stride][:max_images]
    if not selected_source_paths:
        raise RuntimeError(
            f"No images selected from {image_dir} using start_index={start_index}, "
            f"stride={stride}, max_images={max_images}"
        )

    prepared_paths: list[Path] = []
    for index, source_path in enumerate(selected_source_paths):
        prepared_path = output_dir / f"frame_{index:03d}.png"
        with Image.open(source_path) as image:
            image.convert("RGB").save(prepared_path)
        prepared_paths.append(prepared_path)

    metadata = {
        "image_dir": str(image_dir),
        "total_images": len(candidate_paths),
        "selected_count": len(selected_source_paths),
        "selected_source_paths": [str(path) for path in selected_source_paths],
        "prepared_paths": [str(path) for path in prepared_paths],
        "start_index": start_index,
        "stride": stride,
        "max_images": max_images,
        "extensions": sorted(normalized_extensions),
    }
    return prepared_paths, metadata


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    default_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if extensions is None:
        return default_extensions

    normalized = set()
    for extension in extensions:
        extension = extension.strip().lower()
        if not extension:
            continue
        normalized.add(extension if extension.startswith(".") else f".{extension}")
    return normalized or default_extensions
