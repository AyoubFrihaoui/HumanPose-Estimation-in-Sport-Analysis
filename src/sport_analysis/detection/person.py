"""YOLOv8 person detection used by the athlete tracker."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO


def load_person_detector(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def detect_persons(
    frame: Any, model: YOLO, confidence: float = 0.1
) -> list[tuple[int, int, int, int]]:
    """Return person boxes as ``(x, y, width, height)``."""
    results = model(frame, conf=confidence)
    person_boxes = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        if int(result.cls) == 0:
            person_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return person_boxes

