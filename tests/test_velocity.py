import numpy as np

from sport_analysis.metrics.velocity import calculate_speed, moving_average


def test_calculate_speed_preserves_prototype_formula() -> None:
    positions = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
    assert calculate_speed(positions, fps=2.0) == [10.0]


def test_moving_average() -> None:
    np.testing.assert_allclose(moving_average([1.0, 2.0, 3.0], 2), [1.5, 2.5])

