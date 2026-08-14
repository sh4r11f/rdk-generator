from __future__ import annotations

import numpy as np


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def resolve_dot_luminances(
    *,
    background_lum: float,
    dot_contrast: float | None,
    dot_low_lum: float | None,
    dot_high_lum: float | None,
) -> tuple[float, float]:
    """Resolve the low/high dot luminance values.

    If explicit `dot_low_lum` and `dot_high_lum` are provided they are used.
    Otherwise, `dot_contrast` defines two luminances around background.
    """
    bg = clip01(background_lum)
    if dot_low_lum is not None and dot_high_lum is not None:
        lo = clip01(dot_low_lum)
        hi = clip01(dot_high_lum)
        if hi == lo:
            raise ValueError("dot_high_lum and dot_low_lum must differ")
        return (lo, hi)

    if dot_contrast is None:
        raise ValueError("Provide either dot_contrast or both dot_low_lum and dot_high_lum")

    c = float(dot_contrast)
    # Interpreted as absolute luminance span; clipped to displayable range.
    hi = clip01(bg + c / 2.0)
    lo = clip01(bg - c / 2.0)
    if hi == lo:
        # With extreme clipping you can end up equal; fall back to max span.
        hi = 1.0
        lo = 0.0
    return (lo, hi)


def uniform_dot_luminance(
    *,
    background_lum: float,
    dot_contrast: float | None,
    dot_high_lum: float | None = None,
) -> float:
    """Resolve the single luminance used when every dot is drawn the same shade.

    This is the classic arrangement: bright dots on a darker background. `dot_contrast`
    is an absolute luminance span, so the dot sits `dot_contrast / 2` above the
    background (clipped to the displayable range). To get white dots on black, pair
    `background_lum=0.0` with `dot_contrast=2.0`.

    An explicit `dot_high_lum` overrides the contrast-derived value.
    """
    if dot_high_lum is not None:
        return clip01(dot_high_lum)
    if dot_contrast is None:
        raise ValueError("Provide either dot_contrast or dot_high_lum")

    bg = clip01(background_lum)
    lum = clip01(bg + float(dot_contrast) / 2.0)
    if lum == bg:
        # Zero span (or clipping collapsed it): fall back to maximum visibility.
        lum = 1.0 if bg < 0.5 else 0.0
    return lum


def white_fraction_for_mean(background_lum: float, low_lum: float, high_lum: float) -> float:
    """Fraction of high-luminance dots needed so mean equals background.

    Solve p*high + (1-p)*low = background => p = (bg - low) / (high - low)
    Result is clipped to [0,1].
    """
    bg = clip01(background_lum)
    lo = clip01(low_lum)
    hi = clip01(high_lum)
    if hi == lo:
        return 0.5
    p = (bg - lo) / (hi - lo)
    return float(np.clip(p, 0.0, 1.0))


def assign_binary_luminances(
    n: int,
    *,
    background_lum: float,
    low_lum: float,
    high_lum: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign per-dot luminance values (low/high) with mean equal to background."""
    n = int(max(0, n))
    p_high = white_fraction_for_mean(background_lum, low_lum, high_lum)
    # Exact count to reduce sampling noise in mean luminance
    n_high = int(round(n * p_high))
    lum = np.full(n, float(low_lum), dtype=np.float32)
    if n_high > 0:
        idx = np.arange(n)
        rng.shuffle(idx)
        lum[idx[:n_high]] = float(high_lum)
    return lum
