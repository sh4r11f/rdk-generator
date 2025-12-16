from __future__ import annotations

import math

import numpy as np

from .luminance import assign_binary_luminances, resolve_dot_luminances
from .sampling import rand_points_in_circle


class RDKEngine:
    """Headless RDK simulation (positions, membership, dot life).

    This is intentionally PsychoPy-free so it can be tested and used on servers.

    Coordinates:
      - Dot positions are stored relative to the field center (0,0) in pixels.
    """

    def __init__(
        self,
        *,
        n_dots: int,
        dot_life_frames: int,
        coherence: float,
        direction_deg: float,
        speed_px_per_s: float,
        field_diam_px: float,
        gauss_sigma_px: float | None,
        reassign_life: bool,
        fps: int,
        seed: int | None,
        background_lum: float,
        dot_contrast: float | None,
        dot_low_lum: float | None,
        dot_high_lum: float | None,
    ) -> None:
        self.n = int(max(1, n_dots))
        self.dot_life = int(max(1, dot_life_frames))
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self.direction_deg = float(direction_deg)
        self.speed_px_per_s = float(speed_px_per_s)
        self.field_diam = float(field_diam_px)
        self.radius = self.field_diam / 2.0
        self.gauss_sigma = float(gauss_sigma_px) if gauss_sigma_px is not None else (self.radius / 2.0)
        self.reassign_life = bool(reassign_life)
        self.fps = int(max(1, fps))

        self.rng = np.random.default_rng(seed)

        self.xys = rand_points_in_circle(self.n, self.radius, self.rng)
        self.life = self.rng.integers(1, self.dot_life + 1, size=self.n, dtype=np.int32)
        self.is_signal = np.zeros(self.n, dtype=bool)
        self._assign_membership()

        lo, hi = resolve_dot_luminances(
            background_lum=background_lum,
            dot_contrast=dot_contrast,
            dot_low_lum=dot_low_lum,
            dot_high_lum=dot_high_lum,
        )
        self.dot_low_lum = float(lo)
        self.dot_high_lum = float(hi)
        self.dot_lum = assign_binary_luminances(
            self.n,
            background_lum=background_lum,
            low_lum=self.dot_low_lum,
            high_lum=self.dot_high_lum,
            rng=self.rng,
        )

        theta = math.radians(self.direction_deg)
        self.signal_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

    def _assign_membership(self) -> None:
        n_sig = int(round(self.n * self.coherence))
        idx = np.arange(self.n)
        self.rng.shuffle(idx)
        self.is_signal[:] = False
        if n_sig > 0:
            self.is_signal[idx[:n_sig]] = True

    def compute_opacity(self) -> np.ndarray:
        """Gaussian radial mask opacity per dot in [0,1]."""
        if self.gauss_sigma <= 0:
            return np.ones(self.n, dtype=np.float32)
        r2 = np.sum(self.xys * self.xys, axis=1)
        sigma2 = float(self.gauss_sigma) ** 2
        alpha = np.exp(-0.5 * r2 / sigma2)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def step(self) -> None:
        """Advance simulation by one frame."""
        self.life -= 1
        dead = self.life <= 0
        if np.any(dead):
            self.xys[dead] = rand_points_in_circle(int(np.sum(dead)), self.radius, self.rng)
            self.life[dead] = self.dot_life
            if self.reassign_life:
                self._assign_membership()

        step_scale = self.speed_px_per_s / float(self.fps)
        steps = np.empty_like(self.xys)

        steps[self.is_signal] = self.signal_vec[None, :] * step_scale
        n_noise = int(np.count_nonzero(~self.is_signal))
        if n_noise:
            thetas = 2.0 * np.pi * self.rng.random(n_noise)
            rand_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)]).astype(np.float32)
            steps[~self.is_signal] = rand_vecs * step_scale

        self.xys += steps

        outside = np.sum(self.xys * self.xys, axis=1) > (self.radius * self.radius)
        if np.any(outside):
            self.xys[outside] = rand_points_in_circle(int(np.sum(outside)), self.radius, self.rng)


class FrameRenderer:
    """Render RDK frames into grayscale images (uint8)."""

    def __init__(
        self,
        *,
        width_px: int,
        height_px: int,
        background_lum: float,
        field_center_xy_px: tuple[float, float],
        dot_size_px: int,
    ) -> None:
        self.width = int(width_px)
        self.height = int(height_px)
        self.bg = float(np.clip(background_lum, 0.0, 1.0))
        self.cx = float(field_center_xy_px[0])
        self.cy = float(field_center_xy_px[1])
        self.dot_size_px = int(max(1, dot_size_px))

        self._disk = self._make_disk_mask(self.dot_size_px)

    @staticmethod
    def _make_disk_mask(dot_size_px: int) -> np.ndarray:
        r = max(0.5, dot_size_px / 2.0)
        rad = int(math.ceil(r))
        ys, xs = np.mgrid[-rad: rad + 1, -rad: rad + 1]
        mask = (xs * xs + ys * ys) <= (r * r)
        return mask

    def render(self, xys: np.ndarray, dot_lum: np.ndarray, opacity: np.ndarray) -> np.ndarray:
        """Render a single frame.

        Args:
          xys: (n,2) positions relative to field center (0,0).
          dot_lum: (n,) per-dot luminance in [0,1].
          opacity: (n,) per-dot alpha in [0,1].

        Returns:
          uint8 image of shape (height, width), grayscale.
        """
        img = np.full((self.height, self.width), self.bg, dtype=np.float32)

        rad_y, rad_x = (self._disk.shape[0] // 2, self._disk.shape[1] // 2)
        # Center of the image is (width/2, height/2) in pixel coordinates.
        ox = self.width / 2.0 + self.cx
        oy = self.height / 2.0 - self.cy

        for (x, y), lum, a in zip(xys, dot_lum, opacity, strict=False):
            px = int(round(ox + float(x)))
            py = int(round(oy - float(y)))
            x0 = px - rad_x
            x1 = px + rad_x + 1
            y0 = py - rad_y
            y1 = py + rad_y + 1
            if x1 <= 0 or y1 <= 0 or x0 >= self.width or y0 >= self.height:
                continue

            sx0 = max(0, x0)
            sy0 = max(0, y0)
            sx1 = min(self.width, x1)
            sy1 = min(self.height, y1)

            mx0 = sx0 - x0
            my0 = sy0 - y0
            mx1 = mx0 + (sx1 - sx0)
            my1 = my0 + (sy1 - sy0)

            m = self._disk[my0:my1, mx0:mx1]
            if not np.any(m):
                continue

            a = float(np.clip(a, 0.0, 1.0))
            v = self.bg * (1.0 - a) + float(lum) * a

            patch = img[sy0:sy1, sx0:sx1]
            patch[m] = v

        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
