# Architecture

## Processing pipeline

```text
Input video
  -> YOLOv8 person detection
  -> proximity-based track association
  -> athlete selection by maximum horizontal displacement
  -> athlete-centred crop per frame
  -> MediaPipe pose estimation
  -> full-frame landmark coordinates
  -> custom YOLOv8 + SAHI marker detection on the first frame
  -> five marker roles assigned from right to left
  -> planar homography (image pixels to metres)
  -> left-hip ground-plane positions
  -> frame-to-frame Euclidean distance / elapsed time
  -> velocity JSON and optional plot
```

The command-line entry point is `sport_analysis.__main__`. It creates a
`PipelineSettings` instance and calls
`sport_analysis.processing.pipeline.run_pipeline`.

## Module responsibilities

### `config.py`

Defines repository-relative paths and exposes the experimental parameters in
`PipelineSettings`. Values that were formerly hard-coded in the monolithic script
are visible here but retain their original defaults.

### `detection/person.py`

Loads `models/yolov8s.pt` through Ultralytics and returns class-0 detections as
`(x, y, width, height)` boxes.

### `tracking/athlete.py`

Associates a detection with the first prior track whose centre difference is
smaller than the current box dimensions. After the video is scanned, the track
with the largest X-coordinate range is selected as the athlete.

### `pose/estimation.py`

Replays the video, crops around the selected track, and runs MediaPipe Pose.
Landmarks normalized to each crop are converted into full-frame pixel positions.

### `detection/markers.py`

Runs SAHI sliced inference with `models/ground_marker_yolov8.pt` on the first
video frame. Five detections are assigned the original semantic roles by sorting
from right to left.

### `calibration/homography.py`

Pairs the detected marker centres with the original measured ground coordinates
and calls `cv2.findHomography`. `apply_homography` maps image points into metres.

### `metrics/velocity.py`

Extracts the left hip X coordinate, combines it with the preserved fixed Y value,
maps it through the homography, and calculates Euclidean displacement divided by
`1 / fps`.

### `visualization/velocity_plot.py`

Plots raw velocity and, when at least 20 samples exist, the original 20-sample
moving average.

### `processing/pipeline.py`

Coordinates all stages, performs the preserved LOWESS calculation, writes JSON,
and returns both the JSON text and pandas DataFrame.

## Model dependencies

| Model | Purpose | Active? |
|---|---|---|
| `models/yolov8s.pt` | General COCO person detection | Yes |
| `models/ground_marker_yolov8.pt` | Custom ground-marker detection | Yes |
| `models/experimental_best2.pt` | Unidentified historical training result | No; preserved |
| `models/efficientdet_lite2.tflite` | Early EfficientDet experiment | No; preserved |
| MediaPipe Pose complexity 2 | Human pose landmarks | Yes |

## Data flow and units

Person detections and tracks use pixel coordinates. MediaPipe landmarks are first
normalized within a crop, then changed to full-frame pixel coordinates. Marker
centres and ground measurements produce a planar transformation from pixels to
metres. Consecutive transformed hip locations are therefore separated in metres;
multiplying displacement by FPS produces metres per second.

The current implementation assumes 60 FPS rather than reading video metadata.
It also replaces the hip Y landmark with a fixed value of 1060. Both behaviors
come from the original prototype and remain intentionally unchanged.

