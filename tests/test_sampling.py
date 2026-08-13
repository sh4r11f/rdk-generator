import numpy as np

from rdk_generator.sampling import (
    poisson_points_in_circle,
    rand_points_in_circle,
    rand_unit_vectors,
)


def min_pairwise_distance(xys: np.ndarray) -> float:
    dist = np.linalg.norm(xys[:, None, :] - xys[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))


def test_rand_points_in_circle_within_radius():
    rng = np.random.default_rng(0)
    pts = rand_points_in_circle(10_000, radius=5.0, rng=rng)
    r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    assert np.all(r2 <= 5.0 * 5.0 + 1e-6)


def test_rand_points_in_circle_is_uniform_over_area():
    # Uniform over area means half the points fall inside r = R/sqrt(2).
    rng = np.random.default_rng(1)
    radius = 5.0
    pts = rand_points_in_circle(40_000, radius=radius, rng=rng)
    inner = np.linalg.norm(pts, axis=1) < radius / np.sqrt(2.0)
    assert abs(inner.mean() - 0.5) < 0.01


def test_rand_unit_vectors_are_unit_length_and_spread():
    rng = np.random.default_rng(2)
    vecs = rand_unit_vectors(5_000, rng)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    # Directions cover the circle, so the mean vector is near zero.
    assert np.linalg.norm(vecs.mean(axis=0)) < 0.05


def test_poisson_points_respect_min_separation():
    rng = np.random.default_rng(3)
    pts, achieved = poisson_points_in_circle(120, radius=100.0, min_sep_px=5.0, rng=rng)
    assert pts.shape == (120, 2)
    assert achieved == 5.0
    assert min_pairwise_distance(pts) >= 5.0 - 1e-4
    assert np.all(np.linalg.norm(pts, axis=1) <= 100.0 + 1e-4)


def test_poisson_points_clear_existing_points():
    rng = np.random.default_rng(4)
    existing = np.array([[0.0, 0.0], [10.0, 0.0], [-10.0, 5.0]], dtype=np.float32)
    pts, _ = poisson_points_in_circle(40, radius=60.0, min_sep_px=6.0, rng=rng, existing=existing)
    both = np.vstack([existing, pts])
    assert min_pairwise_distance(both) >= 6.0 - 1e-4


def test_poisson_relaxes_when_density_is_impossible():
    # 30 dots each needing 20 px of clearance cannot fit in a 40 px radius circle.
    rng = np.random.default_rng(5)
    pts, achieved = poisson_points_in_circle(
        30, radius=40.0, min_sep_px=20.0, rng=rng, max_attempts=400
    )
    assert pts.shape == (30, 2)
    assert achieved < 20.0
    assert min_pairwise_distance(pts) >= achieved - 1e-4


def test_poisson_handles_zero_points():
    rng = np.random.default_rng(6)
    pts, achieved = poisson_points_in_circle(0, radius=10.0, min_sep_px=2.0, rng=rng)
    assert pts.shape == (0, 2)
    assert achieved == 2.0
