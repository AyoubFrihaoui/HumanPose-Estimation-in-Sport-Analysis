"""SAHI sliced inference for the five ground calibration markers."""

from pathlib import Path
from typing import Any

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def read_first_frame(video_path: str | Path) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read the first frame: {video_path}")
    return frame


def detect_ground_markers(
    video_path: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.3,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(Path(model_path).resolve()),
        confidence_threshold=confidence_threshold,
        device=device,
    )
    result = get_sliced_prediction(
        read_first_frame(video_path),
        detection_model,
        slice_width=1024,
        slice_height=1024,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    )
    return [
        {"id": index, "bbox": annotation["bbox"]}
        for index, annotation in enumerate(result.to_coco_annotations())
    ]


def assign_marker_positions(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign marker roles from right to left, as in the original pipeline."""
    remaining = list(detections)
    positions = [
        "Bottom-right",
        "Top-right",
        "Midpoint",
        "Top-left",
        "Bottom-left",
    ]
    assigned = []

    for position in positions:
        if not remaining:
            break
        detection = max(remaining, key=lambda item: item["bbox"][0])
        assigned.append({**detection, "position": position})
        remaining.remove(detection)

    return assigned

