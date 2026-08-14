from __future__ import annotations

import numpy as np


def rand_points_in_circle(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Sample `n` points uniformly over the area of a circle.

    Returns an array of shape (n, 2) with (x, y) points.
    """
    n = int(max(0, n))
    radius = float(radius)
    r = radius * np.sqrt(rng.random(n))
    t = 2.0 * np.pi * rng.random(n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y]).astype(np.float32)


def rand_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `n` unit vectors with uniformly distributed directions.

    Returns an array of shape (n, 2) with (dx, dy) components.
    """
    n = int(max(0, n))
    t = 2.0 * np.pi * rng.random(n)
    return np.column_stack([np.cos(t), np.sin(t)]).astype(np.float32)


def poisson_points_in_circle(
    n: int,
    radius: float,
    min_sep_px: float,
    rng: np.random.Generator,
    *,
    existing: np.ndarray | None = None,
    max_attempts: int = 20000,
    relax_factor: float = 0.95,
) -> tuple[np.ndarray, float]:
    """Place `n` points in a circle keeping every pair at least `min_sep_px` apart.

    Uses rejection sampling against `existing` points (which are left untouched) and
    against points accepted so far. If the requested density is unachievable within
    `max_attempts`, the separation is relaxed by `relax_factor` and the batch restarts.

    Args:
      n: Number of new points to place.
      radius: Field radius in pixels.
      min_sep_px: Requested minimum centre-to-centre distance.
      rng: Seeded generator.
      existing: Optional (m, 2) array of fixed points new points must clear.
      max_attempts: Rejection budget before relaxing the constraint.
      relax_factor: Multiplier applied to `min_sep_px` when the budget is exhausted.

    Returns:
      A tuple of the (n, 2) placed points and the separation actually achieved, which
      is `min_sep_px` unless relaxation was needed.
    """
    n = int(max(0, n))
    sep = float(min_sep_px)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), sep

    fixed = (
        np.zeros((0, 2), dtype=np.float32)
        if existing is None or len(existing) == 0
        else np.asarray(existing, dtype=np.float32)
    )

    while True:
        sep_sq = sep * sep
        pts = np.empty((n, 2), dtype=np.float32)
        placed = 0
        attempts = 0
        relax = False

        while placed < n:
            if attempts > max_attempts:
                relax = True
                break

            # Draw a small batch at a time: cheaper than one point per RNG call.
            cand = rand_points_in_circle(min(64, (n - placed) * 8), radius, rng)
            for c in cand:
                if placed >= n:
                    break
                if fixed.shape[0]:
                    d = fixed - c
                    if np.any(np.einsum("ij,ij->i", d, d) < sep_sq):
                        attempts += 1
                        continue
                if placed:
                    d = pts[:placed] - c
                    if np.any(np.einsum("ij,ij->i", d, d) < sep_sq):
                        attempts += 1
                        continue
                pts[placed] = c
                placed += 1

        if not relax:
            return pts, sep
        sep *= relax_factor
