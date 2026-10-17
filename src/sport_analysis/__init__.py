"""Sprint video analysis package."""

from pathlib import Path
from typing import Any


def run_pipeline(video_path: str | Path, settings: Any = None) -> Any:
    """Load and invoke the heavy computer-vision pipeline lazily."""
    from sport_analysis.processing.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(video_path, settings)


__all__ = ["run_pipeline"]
