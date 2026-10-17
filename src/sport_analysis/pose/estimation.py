"""MediaPipe pose estimation on athlete-centred crops."""

import copy
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    if 0 <= normalized_x <= 1 and 0 <= normalized_y <= 1:
        return (
            min(int(normalized_x * image_width), image_width - 1),
            min(int(normalized_y * image_height), image_height - 1),
        )
    return None


def estimate_pose_for_track(
    tracks: dict[int, tuple[float, float, list[list[int]]]],
    track_id: int,
    video_path: str | Path,
    crop_left: int = 300,
    crop_right: int = 200,
    crop_vertical: int = 200,
    display: bool = True,
) -> tuple[list[Any], list[Any]]:
    """Estimate pose in each tracked crop and restore full-frame coordinates."""
    pose_results_cropped = []
    pose_results_full = []
    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    pose = pose_module.Pose(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=2,
    )
    cap = cv2.VideoCapture(str(video_path))

    with pose:
        frame_index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            try:
                x, y, width, height = tracks[track_id][2][frame_index]
                x_start = max(x - crop_left, 0)
                x_end = min(x + width + crop_right, frame.shape[1])
                y_start = max(y - crop_vertical, 0)
                y_end = min(y + height + crop_vertical, frame.shape[0])
                cropped_frame = frame[y_start:y_end, x_start:x_end]
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)
                pose_results_cropped.append(result)
                drawing.draw_landmarks(
                    frame_rgb, result.pose_landmarks, pose_module.POSE_CONNECTIONS
                )
                frame[y_start:y_end, x_start:x_end] = cv2.cvtColor(
                    frame_rgb, cv2.COLOR_RGB2BGR
                )

                if result.pose_landmarks:
                    full_result = copy.deepcopy(result)
                    for index, landmark in enumerate(result.pose_landmarks.landmark):
                        pixel = normalized_to_pixel_coordinates(
                            landmark.x,
                            landmark.y,
                            cropped_frame.shape[1],
                            cropped_frame.shape[0],
                        )
                        if pixel:
                            full_result.pose_landmarks.landmark[index].x = x_start + pixel[0]
                            full_result.pose_landmarks.landmark[index].y = y_start + pixel[1]
                            full_result.pose_landmarks.landmark[index].z = landmark.z
                    pose_results_full.append(full_result)

                frame_index += 1
                if display:
                    cv2.imshow("Output", cv2.resize(frame, (1280, 720)))
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
                    if key == ord("p"):
                        cv2.waitKey(-1)
            except (IndexError, KeyError, cv2.error):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()
    return pose_results_cropped, pose_results_full

