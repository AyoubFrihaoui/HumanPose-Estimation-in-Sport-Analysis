"""Orchestrate the original sprint-analysis stages."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from sport_analysis.calibration.homography import calibrate_camera, get_calibration_points
from sport_analysis.config import PipelineSettings
from sport_analysis.detection.markers import assign_marker_positions, detect_ground_markers
from sport_analysis.detection.person import load_person_detector
from sport_analysis.metrics.velocity import process_instant_velocity
from sport_analysis.pose.estimation import estimate_pose_for_track
from sport_analysis.tracking.athlete import (
    calculate_max_horizontal_velocity,
    find_fastest_track,
    process_video,
)
from sport_analysis.visualization.velocity_plot import plot_velocity


def run_pipeline(
    video_path: str | Path,
    settings: PipelineSettings | None = None,
) -> tuple[str, pd.DataFrame]:
    """Run detection, pose, calibration, and velocity estimation."""
    settings = settings or PipelineSettings()
    video_path = Path(video_path)

    person_detector = load_person_detector(settings.person_model)
    _, _, tracks = process_video(
        video_path,
        person_detector,
        confidence=settings.person_confidence,
        display=settings.display,
    )
    if not tracks:
        raise RuntimeError("No athlete tracks were detected.")

    track_movements = {
        track_id: calculate_max_horizontal_velocity(track[2])[1]
        for track_id, track in tracks.items()
        if track[2]
    }
    fastest_track_id, _ = find_fastest_track(track_movements)

    _, full_frame_poses = estimate_pose_for_track(
        tracks,
        fastest_track_id,
        video_path,
        crop_left=settings.crop_left,
        crop_right=settings.crop_right,
        crop_vertical=settings.crop_vertical,
        display=settings.display,
    )

    marker_detections = detect_ground_markers(
        video_path,
        settings.marker_model,
        confidence_threshold=settings.marker_confidence,
        device=settings.marker_device,
    )
    positioned_markers = assign_marker_positions(marker_detections)
    if len(positioned_markers) < 5:
        raise RuntimeError(
            f"Expected five calibration markers, detected {len(positioned_markers)}."
        )

    real_world_points, image_points = get_calibration_points(positioned_markers)
    homography = calibrate_camera(real_world_points, image_points, (3840, 2160))
    velocities = process_instant_velocity(
        homography,
        settings.fps,
        full_frame_poses,
        fixed_hip_y=settings.fixed_hip_y,
    )
    time_steps = np.arange(len(velocities)) / settings.fps
    dataframe = pd.DataFrame({"velocity": velocities, "time_steps": time_steps})

    # Preserve the LOWESS calculation performed by the prototype.
    if not dataframe.empty:
        lowess(dataframe["velocity"], dataframe["time_steps"], frac=0.095, it=1)

    if settings.show_plot:
        plot_velocity(time_steps, velocities)

    json_data = dataframe.to_json(orient="records")
    settings.output_json.parent.mkdir(parents=True, exist_ok=True)
    settings.output_json.write_text(json_data, encoding="utf-8")
    return json_data, dataframe

