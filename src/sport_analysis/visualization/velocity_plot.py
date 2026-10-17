"""Velocity time-series plotting."""

import matplotlib.pyplot as plt
import numpy as np

from sport_analysis.metrics.velocity import moving_average


def plot_velocity(time_steps: np.ndarray, velocities: list[float]) -> None:
    plt.plot(time_steps, velocities)
    if len(velocities) >= 20:
        plt.plot(
            moving_average(time_steps, 20),
            moving_average(velocities, 20),
        )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Athlete Velocity Over Time")
    plt.show()

