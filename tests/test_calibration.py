import numpy as np

from sport_analysis.calibration.homography import apply_homography


def test_identity_homography() -> None:
    np.testing.assert_allclose(
        apply_homography(np.identity(3), (4.0, 7.0)),
        [4.0, 7.0],
    )

