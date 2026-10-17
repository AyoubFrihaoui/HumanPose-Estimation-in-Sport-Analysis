"""The prototype's proximity-based multi-person tracker."""

from pathlib import Path
from typing import Any

import cv2

from sport_analysis.detection.person import detect_persons

BoundingBox = tuple[int, int, int, int]
Track = tuple[float, float, list[list[int]]]


def calculate_horizontal_velocity(
    current_positions: dict[int, tuple[float, BoundingBox]],
    previous_positions: dict[int, tuple[float, BoundingBox]],
) -> dict[int, float]:
    velocities = {}
    for track_id in current_positions:
        if track_id in previous_positions:
            velocities[track_id] = abs(
                current_positions[track_id][0] - previous_positions[track_id][0]
            )
    return velocities


def process_video(
    video_path: str | Path,
    model: Any,
    confidence: float = 0.1,
    display: bool = True,
) -> tuple[BoundingBox | None, int | None, dict[int, Track]]:
    """Detect and associate people using the original overlap heuristic."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    previous_positions = {}
    max_velocity = 0.0
    fastest_box = None
    fastest_track_id = None
    tracks: dict[int, Track] = {}
    next_track_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        current_positions = {}
        for box in detect_persons(frame, model, confidence):
            x, y, width, height = box
            center_x = x + width / 2
            center_y = y + height / 2

            for track_id, (previous_x, previous_y, history) in tracks.items():
                if (
                    abs(center_x - previous_x) < width
                    and abs(center_y - previous_y) < height
                ):
                    current_positions[track_id] = (center_x, box)
                    tracks[track_id] = (
                        center_x,
                        center_y,
                        history + [[x, y, width, height]],
                    )
                    break
            else:
                tracks[next_track_id] = (
                    center_x,
                    center_y,
                    [[x, y, width, height]],
                )
                current_positions[next_track_id] = (center_x, box)
                next_track_id += 1

        for track_id, velocity in calculate_horizontal_velocity(
            current_positions, previous_positions
        ).items():
            if velocity > max_velocity:
                max_velocity = velocity
                fastest_track_id = track_id
                fastest_box = current_positions[track_id][1]

        previous_positions = current_positions

        if display:
            for track_id, (_, box) in current_positions.items():
                x, y, width, height = box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
            cv2.imshow("Frame", cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()
    return fastest_box, fastest_track_id, tracks


def calculate_max_horizontal_velocity(
    boxes: list[list[int]],
) -> tuple[list[int], int]:
    x_coordinates = [box[0] for box in boxes]
    maximum_x = max(x_coordinates)
    minimum_x = min(x_coordinates)
    maximum_x_box = next(box for box in boxes if box[0] == maximum_x)
    return maximum_x_box, maximum_x - minimum_x


def find_fastest_track(
    velocities: dict[int, float],
) -> tuple[int, float]:
    track_id = max(velocities, key=velocities.get)
    return track_id, velocities[track_id]

