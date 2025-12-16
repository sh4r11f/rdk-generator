import numpy as np

from rdk_generator.luminance import (
    assign_binary_luminances,
    resolve_dot_luminances,
    white_fraction_for_mean,
)


def test_white_fraction_balances_mean():
    bg = 0.3
    lo, hi = 0.0, 1.0
    p = white_fraction_for_mean(bg, lo, hi)
    assert abs(p - bg) < 1e-9


def test_assign_binary_luminances_exact_mean_close():
    rng = np.random.default_rng(123)
    bg = 0.37
    lo, hi = 0.0, 1.0
    lum = assign_binary_luminances(1000, background_lum=bg, low_lum=lo, high_lum=hi, rng=rng)
    assert lum.min() >= 0.0 and lum.max() <= 1.0
    # With rounding, mean should be close.
    assert abs(lum.mean() - bg) < 0.01


def test_resolve_dot_luminances_from_contrast():
    lo, hi = resolve_dot_luminances(background_lum=0.5, dot_contrast=1.0, dot_low_lum=None, dot_high_lum=None)
    assert lo < hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
