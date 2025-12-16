from __future__ import annotations

"""Optional PsychoPy integration.

This module is designed to be importable even when PsychoPy is not installed.
Callers should handle `ImportError` if they need PsychoPy-only functionality.

This module provides a PsychoPy implementation for gaussian-masked motion and a
luminance-balanced variant that assigns black/white dots so the *mean* dot
luminance matches the background.
"""

import numpy as np

from .luminance import assign_binary_luminances, resolve_dot_luminances


def _require_psychopy():
    try:
        from psychopy import visual  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PsychoPy is required for rdk_generator.psychopy_adapter; "
            "install extras: pip install 'rdk-generator[psychopy]'"
        ) from e


class BalancedGaussianRDK:  # pragma: no cover (requires PsychoPy runtime)
    """Gaussian RDK with luminance-balanced black/white dots.

    The motion/life logic mirrors a classic Gaussian-masked RDK, but dot colors are
    assigned per-element.

    Color model:
      - `background_rgb` is expected in PsychoPy `rgb` space [-1, 1] and should be gray.
      - Dots are assigned two luminance levels ("low" and "high") and mixed to ensure
        the mean equals the background.

    Notes:
      - This balancing is done in linear RGB space. Exact photometric luminance depends
        on your monitor calibration/gamma setup.
    """

    def __init__(
        self,
        *,
        win,
        n_dots: int,
        dot_size: float,
        speed: float,
        dot_life: int,
        direction: float,
        coherence: float,
        field_pos: tuple[float, float],
        field_size: float,
        gauss_sigma: float | None = None,
        background_lum: float = 0.5,
        dot_contrast: float | None = 1.0,
        dot_low_lum: float | None = None,
        dot_high_lum: float | None = None,
        reassign_life: bool = True,
        seed: int | None = None,
    ) -> None:
        _require_psychopy()
        from psychopy import visual  # type: ignore[import-not-found]

        self.win = win
        self.n = int(max(1, n_dots))
        self.dot_size = float(dot_size)
        self.speed = float(speed)
        self.dot_life = int(max(1, dot_life))
        self.direction = float(direction)
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self.field_pos = np.array(field_pos, dtype=float)
        self.field_diam = float(field_size)
        self.radius = self.field_diam / 2.0
        self.gauss_sigma = float(gauss_sigma) if gauss_sigma is not None else (self.radius / 2.0)
        self.reassign_life = bool(reassign_life)

        self.rng = np.random.default_rng(seed)

        self.xys = self._rand_points_in_circle(self.n, self.radius)
        self.life = self.rng.integers(1, self.dot_life + 1, size=self.n, dtype=np.int32)
        self._assign_membership()

        self._frame_dur = getattr(self.win, "monitorFramePeriod", None) or (1.0 / 60.0)

        lo, hi = resolve_dot_luminances(
            background_lum=background_lum,
            dot_contrast=dot_contrast,
            dot_low_lum=dot_low_lum,
            dot_high_lum=dot_high_lum,
        )
        dot_lum = assign_binary_luminances(
            self.n,
            background_lum=background_lum,
            low_lum=lo,
            high_lum=hi,
            rng=self.rng,
        )
        # PsychoPy rgb space expects [-1,1]
        dot_rgb = (dot_lum * 2.0 - 1.0).astype(np.float32)
        self.colors = np.column_stack([dot_rgb, dot_rgb, dot_rgb])

        theta = np.deg2rad(self.direction)
        self.signal_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)

        self.stim = visual.ElementArrayStim(
            self.win,
            nElements=self.n,
            sizes=self.dot_size,
            xys=self._apply_offset(self.xys),
            colors=self.colors,
            colorSpace="rgb",
            elementTex=None,
            elementMask="circle",
            opacities=self._compute_opacity(self.xys),
            interpolate=False,
            autoLog=False,
        )

    def draw(self):
        self._step()
        self.stim.xys = self._apply_offset(self.xys)
        self.stim.opacities = self._compute_opacity(self.xys)
        self.stim.draw()

    def _assign_membership(self):
        n_sig = int(round(self.n * self.coherence))
        idx = np.arange(self.n)
        self.rng.shuffle(idx)
        self.is_signal = np.zeros(self.n, dtype=bool)
        if n_sig > 0:
            self.is_signal[idx[:n_sig]] = True

    @staticmethod
    def _rand_points_in_circle(n, radius):
        r = radius * np.sqrt(np.random.rand(n))
        t = 2 * np.pi * np.random.rand(n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y]).astype(float)

    def _apply_offset(self, xys):
        return xys + self.field_pos[None, :]

    def _compute_opacity(self, xys):
        r2 = np.sum(xys**2, axis=1)
        sigma2 = (self.gauss_sigma**2)
        if sigma2 <= 0:
            return np.ones(self.n, dtype=float)
        alpha = np.exp(-0.5 * r2 / sigma2)
        return np.clip(alpha, 0.0, 1.0)

    def _step(self):
        self.life -= 1
        dead = self.life <= 0
        if np.any(dead):
            self.xys[dead] = self._rand_points_in_circle(int(np.sum(dead)), self.radius)
            self.life[dead] = self.dot_life
            if self.reassign_life:
                self._assign_membership()

        step_scale = self.speed * self._frame_dur
        steps = np.empty_like(self.xys)
        steps[self.is_signal] = self.signal_vec[None, :] * step_scale
        n_noise = np.count_nonzero(~self.is_signal)
        if n_noise > 0:
            thetas = 2 * np.pi * np.random.rand(n_noise)
            rand_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
            steps[~self.is_signal] = rand_vecs * step_scale
        self.xys += steps

        outside = np.sum(self.xys**2, axis=1) > (self.radius * self.radius)
        if np.any(outside):
            self.xys[outside] = self._rand_points_in_circle(int(np.sum(outside)), self.radius)
