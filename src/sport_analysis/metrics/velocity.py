"""Hip trajectory extraction and instantaneous velocity calculation."""

from typing import Any

import numpy as np

from sport_analysis.calibration.homography import apply_homography


def detect_hip_joint(pose_result: Any, fixed_y: float = 1060.0) -> tuple[float, float]:
    """Return the left-hip X and the prototype's fixed ground-plane Y value."""
    import mediapipe as mp

    x = pose_result.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.LEFT_HIP
    ].x
    return x, fixed_y


def calculate_speed(positions: list[np.ndarray], fps: float) -> list[float]:
    velocities = []
    for index in range(1, len(positions)):
        distance = np.linalg.norm(
            np.array(positions[index]) - np.array(positions[index - 1])
        )
        velocities.append(distance / (1 / fps))
    return velocities


def process_instant_velocity(
    homography: np.ndarray,
    fps: float,
    pose_results: list[Any],
    fixed_hip_y: float = 1060.0,
) -> list[float]:
    positions = []
    for result in pose_results:
        if not result.pose_landmarks:
            break
        hip_joint = detect_hip_joint(result, fixed_hip_y)
        positions.append(apply_homography(homography, hip_joint))
    return calculate_speed(positions, fps)


def moving_average(values: list[float] | np.ndarray, window_size: int) -> np.ndarray:
    moving_averages = np.zeros(len(values) - window_size + 1)
    for index in range(len(moving_averages)):
        moving_averages[index] = np.mean(values[index : index + window_size])
    return moving_averages
