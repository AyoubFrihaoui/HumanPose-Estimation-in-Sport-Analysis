# Codebase Recovery Guide

This guide separates the maintained pipeline from historical research material.

## Where execution starts

- `src/sport_analysis/__main__.py`: parses CLI arguments.
- `src/sport_analysis/processing/pipeline.py`: `run_pipeline` is the maintained
  application entry point.
- `scripts/run_pipeline.py`: thin compatibility launcher.

There is no active web API. The original FastAPI file was a disconnected dummy
prototype and is preserved under `experiments/legacy/fastapi_prototype.py`.

## Maintained implementation

| File | Important functions | Used by |
|---|---|---|
| `config.py` | `PipelineSettings` | CLI and pipeline |
| `detection/person.py` | `load_person_detector`, `detect_persons` | tracker |
| `tracking/athlete.py` | `process_video`, `calculate_max_horizontal_velocity`, `find_fastest_track` | pipeline |
| `pose/estimation.py` | `estimate_pose_for_track`, `normalized_to_pixel_coordinates` | pipeline |
| `detection/markers.py` | `detect_ground_markers`, `assign_marker_positions` | pipeline |
| `calibration/homography.py` | `get_calibration_points`, `calibrate_camera`, `apply_homography` | pipeline and metrics |
| `metrics/velocity.py` | `detect_hip_joint`, `process_instant_velocity`, `calculate_speed`, `moving_average` | pipeline and plot |
| `visualization/velocity_plot.py` | `plot_velocity` | pipeline |
| `processing/pipeline.py` | `run_pipeline` | CLI |

## Metric locations

- Hip extraction: `metrics/velocity.py::detect_hip_joint`
- Hip trajectory conversion: `metrics/velocity.py::process_instant_velocity`
- Instantaneous velocity: `metrics/velocity.py::calculate_speed`
- Moving-average smoothing: `metrics/velocity.py::moving_average`
- LOWESS calculation: `processing/pipeline.py::run_pipeline`
- JSON time-series construction: `processing/pipeline.py::run_pipeline`

Velocity is the only complete biomechanical metric in the active code. Stride
length appears only as a future heuristic image. Joint-angle extraction is
described by the project objective but is not present as a completed calculation.

## Calibration locations

- Ground-marker inference: `detection/markers.py::detect_ground_markers`
- Marker role assignment: `detection/markers.py::assign_marker_positions`
- Real/image point pairing: `calibration/homography.py::get_calibration_points`
- Homography calculation: `calibration/homography.py::calibrate_camera`
- Pixel-to-world conversion: `calibration/homography.py::apply_homography`

## Historical and experimental files

| File | Original name | Purpose/status |
|---|---|---|
| `experiments/legacy/final_pipeline_prototype.py` | `finalpipe3.py` | Source of the maintained pipeline; preserved verbatim for comparison |
| `experiments/legacy/marker_detection_prototype.py` | `g9.py` | SAHI marker detection; contained a missing hard-coded image override |
| `experiments/legacy/person_tracking_prototype.py` | `GetMertics2.py` | Earlier tracker; processes a `None` path during import |
| `experiments/legacy/fastapi_prototype.py` | `fastAPI4.py` | Dummy endpoints; never called the real pipeline |
| `experiments/legacy/yolo_segmentation_prototype.py` | `yolo_segmentation.py` | Incomplete segmentation wrapper with an incorrectly named constructor |
| `experiments/legacy/run_pipeline_prototype.py` | `testt.py` | Broken runner importing the missing `finalpipe` module |
| `experiments/legacy/sliced_marker_detection_class.py` | `pipleline_classes/PlatessDetection.py` | Alternative marker detector with an invalid `subfolder` path |

These files are references, not supported entry points. Their broken imports and
paths have intentionally not been rewritten, so they continue to document the
state of the original experiments.

## Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `pose_and_marker_detection_experiments.ipynb` | Early MediaPipe, EfficientDet, and SAHI experiments | Historical; contains machine-specific paths |
| `end_to_end_pipeline_development.ipynb` | Development of tracking, pose, calibration, metrics, and API ideas | Main research history; contains duplicated cells and old paths |
| `person_tracking_experiment.ipynb` | Small YOLO person-tracking experiment | Historical |
| `tracking_debug.ipynb` | Minimal import/run debugging for the old metrics script | Historical and incomplete |

Notebook code and outputs were preserved unchanged. Their references to
`Data/Videos`, local Windows directories, `finalpipe2`, and old root-level scripts
are stale. Use the maintained package for new work and consult notebooks only for
research history.

## Assets, models, data, and outputs

- `models/`: all unique trained/downloaded model files.
- `data/samples/`: four original videos. No labeled training dataset is present.
- `configs/dataset.yaml`: historical YOLO config pointing to an absent `/Data`
  dataset; retained because changing it would invent an unknown dataset layout.
- `outputs/example_velocity.json`: velocity output from an earlier run.
- `outputs/prediction_visual.png`: preserved generated detection visualization.
- `docs/assets/`: README figures and demonstration animation.
- `docs/RunRightAI_ProjectReport.pdf`: original research report.

