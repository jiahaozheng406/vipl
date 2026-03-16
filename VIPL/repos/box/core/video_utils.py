"""
Video frame sampling utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from core.io_utils import ensure_dir


LOGGER = logging.getLogger(__name__)


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 3,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> tuple[list[Path], dict[str, object]]:
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = (total_frames / fps) if fps > 0 else 0.0
        if total_frames <= 0:
            raise RuntimeError(f"Video reports no frames: {video_path}")

        quarter_span = duration_seconds * 0.25 if duration_seconds > 0 else None
        span_candidates = [value for value in (max_seconds, quarter_span) if value is not None and value > 0]
        effective_span = min(span_candidates) if span_candidates else duration_seconds

        effective_start = max(0.0, start_seconds)
        effective_end = min(duration_seconds, effective_start + effective_span) if duration_seconds > 0 else 0.0
        if effective_end <= effective_start:
            effective_end = duration_seconds

        start_frame = min(total_frames - 1, max(0, int(round(effective_start * fps)))) if fps > 0 else 0
        end_frame = min(total_frames - 1, max(start_frame, int(round(effective_end * fps)))) if fps > 0 else total_frames - 1

        sampled_indices = _pick_evenly_spaced_indices(start_frame, end_frame, num_frames)
        frame_paths: list[Path] = []

        for local_index, source_index in enumerate(sampled_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to decode frame {source_index} from {video_path}")
            output_path = output_dir / f"frame_{local_index:03d}.png"
            cv2.imwrite(str(output_path), frame_bgr)
            frame_paths.append(output_path)

        metadata = {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "effective_start_seconds": effective_start,
            "effective_end_seconds": effective_end,
            "sampled_frame_indices": sampled_indices,
        }
        LOGGER.info("Sampled frames from %s: %s", video_path, sampled_indices)
        return frame_paths, metadata
    finally:
        cap.release()


def _pick_evenly_spaced_indices(start_frame: int, end_frame: int, count: int) -> list[int]:
    if count <= 1 or start_frame >= end_frame:
        return [start_frame]

    available = end_frame - start_frame + 1
    if available <= count:
        return list(range(start_frame, end_frame + 1))

    raw_positions = np.linspace(start_frame, end_frame, num=count)
    unique_indices: list[int] = []
    used = set()
    for value in raw_positions:
        index = int(round(float(value)))
        index = min(end_frame, max(start_frame, index))
        if index not in used:
            used.add(index)
            unique_indices.append(index)

    if len(unique_indices) < count:
        for index in range(start_frame, end_frame + 1):
            if index not in used:
                used.add(index)
                unique_indices.append(index)
                if len(unique_indices) == count:
                    break

    unique_indices.sort()
    return unique_indices[:count]
