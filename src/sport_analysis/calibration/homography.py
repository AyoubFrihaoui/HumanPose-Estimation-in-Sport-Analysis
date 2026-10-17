"""Map image points to the measured 10 m by 1.22 m ground plane."""

from typing import Any

import cv2
import numpy as np


REAL_WORLD_POINTS = np.array(
    [
        [0, 0, 0],
        [10, 0, 0],
        [10, 1.22, 0],
        [0, 1.22, 0],
        [5, 0, 0],
    ],
    dtype=np.float32,
)


def get_calibration_points(
    positioned_markers: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    def center(position: str) -> list[float] | None:
        for marker in positioned_markers:
            if marker["position"] == position:
                x, y, width, height = marker["bbox"]
                return [x + width / 2, y + height / 2]
        return None

    image_points = np.array(
        [
            center("Top-left"),
            center("Top-right"),
            center("Bottom-right"),
            center("Bottom-left"),
            center("Midpoint"),
        ],
        dtype=np.float32,
    )
    return REAL_WORLD_POINTS.copy(), image_points


def calibrate_camera(
    real_world_points: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute the original planar homography; ``image_size`` is retained for API compatibility."""
    homography, _ = cv2.findHomography(image_points, real_world_points[:, :2])
    return homography


def apply_homography(homography: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    homogeneous_point = np.array([point[0], point[1], 1.0])
    world_point = np.dot(homography, homogeneous_point)
    world_point /= world_point[2]
    return world_point[:2]

