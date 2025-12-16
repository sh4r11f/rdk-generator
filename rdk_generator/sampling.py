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
