import numpy as np

from rdk_generator.sampling import rand_points_in_circle


def test_rand_points_in_circle_within_radius():
    rng = np.random.default_rng(0)
    pts = rand_points_in_circle(10_000, radius=5.0, rng=rng)
    r2 = (pts[:, 0] ** 2 + pts[:, 1] ** 2)
    assert np.all(r2 <= 5.0 * 5.0 + 1e-6)
