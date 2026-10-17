"""Project paths and parameters preserved from the research prototype."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings corresponding to the original hard-coded values."""

    person_model: Path = MODELS_DIR / "yolov8s.pt"
    marker_model: Path = MODELS_DIR / "ground_marker_yolov8.pt"
    output_json: Path = OUTPUTS_DIR / "velocity.json"
    person_confidence: float = 0.1
    marker_confidence: float = 0.3
    marker_device: str = "cpu"
    fps: float = 60.0
    crop_left: int = 300
    crop_right: int = 200
    crop_vertical: int = 200
    fixed_hip_y: float = 1060.0
    display: bool = True
    show_plot: bool = True

