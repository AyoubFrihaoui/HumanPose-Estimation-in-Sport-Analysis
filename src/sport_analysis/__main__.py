"""Command-line entry point for sprint analysis."""

import argparse
from dataclasses import replace
from pathlib import Path

from sport_analysis.config import PipelineSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a sprint video.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Velocity JSON output path")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview windows and the Matplotlib plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sport_analysis.processing.pipeline import run_pipeline

    settings = PipelineSettings()
    if args.output:
        settings = replace(settings, output_json=args.output)
    if args.no_display:
        settings = replace(settings, display=False, show_plot=False)
    _, dataframe = run_pipeline(args.video, settings)
    print(dataframe)
    print(f"Saved velocity data to {settings.output_json}")


if __name__ == "__main__":
    main()
