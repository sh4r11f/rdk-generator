from __future__ import annotations

import math

import numpy as np

from .luminance import (
    assign_binary_luminances,
    resolve_dot_luminances,
    uniform_dot_luminance,
)
from .sampling import poisson_points_in_circle, rand_points_in_circle, rand_unit_vectors

SIGNAL_RULES = ("different", "same")
LUMINANCE_MODES = ("uniform", "balanced")


class BaseRDKEngine:
    """Shared scaffolding for every RDK algorithm (see docs/methods.md).

    Subclasses supply the noise rule — how non-signal dots move — and may override
    placement, per-dot spawn state, and the default Gaussian envelope width.

    This layer is intentionally PsychoPy-free so it can be tested and used on servers.

    Coordinates:
      - Dot positions are stored relative to the field center (0, 0) in pixels,
        with y increasing upwards.

    Conventions:
      - `dot_life_frames <= 0` disables ageing; dots then persist until they exit
        the aperture.
      - `signal_rule="different"` redraws the signal subset every frame;
        `"same"` fixes it for the whole run.
    """

    method_id = "base"

    def __init__(
        self,
        *,
        n_dots: int,
        coherence: float,
        direction_deg: float,
        speed_px_per_s: float,
        field_diam_px: float,
        fps: int,
        seed: int | None = None,
        dot_life_frames: int = 0,
        signal_rule: str = "different",
        gauss_sigma_px: float | None = None,
        background_lum: float = 0.5,
        dot_contrast: float | None = 1.0,
        dot_low_lum: float | None = None,
        dot_high_lum: float | None = None,
        luminance_mode: str = "uniform",
    ) -> None:
        if signal_rule not in SIGNAL_RULES:
            raise ValueError(f"signal_rule must be one of {SIGNAL_RULES}, got {signal_rule!r}")
        if luminance_mode not in LUMINANCE_MODES:
            raise ValueError(
                f"luminance_mode must be one of {LUMINANCE_MODES}, got {luminance_mode!r}"
            )

        self.n = int(max(1, n_dots))
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self.direction_deg = float(direction_deg)
        self.speed_px_per_s = float(speed_px_per_s)
        self.field_diam = float(field_diam_px)
        self.radius = self.field_diam / 2.0
        self.fps = int(max(1, fps))
        self.dot_life = int(dot_life_frames)
        self.signal_rule = signal_rule
        self.luminance_mode = luminance_mode

        self.rng = np.random.default_rng(seed)
        self._init_state()

        sigma = gauss_sigma_px if gauss_sigma_px is not None else self._default_gauss_sigma()
        self.gauss_sigma = float(sigma) if sigma is not None else 0.0

        self.xys = self._initial_positions(self.n)
        self.life = (
            self.rng.integers(1, self.dot_life + 1, size=self.n, dtype=np.int32)
            if self.dot_life > 0
            else np.zeros(self.n, dtype=np.int32)
        )
        self.is_signal = np.zeros(self.n, dtype=bool)
        self._assign_membership()
        self._on_spawn(np.ones(self.n, dtype=bool))

        self.dot_low_lum, self.dot_high_lum, self.dot_lum = self._resolve_luminance(
            background_lum=background_lum,
            dot_contrast=dot_contrast,
            dot_low_lum=dot_low_lum,
            dot_high_lum=dot_high_lum,
        )

        theta = math.radians(self.direction_deg)
        self.signal_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

    # -- hooks -------------------------------------------------------------

    def _init_state(self) -> None:
        """Allocate subclass-specific per-dot arrays. Runs before dot placement."""

    def _default_gauss_sigma(self) -> float | None:
        """Envelope width used when the caller does not supply one. None = hard aperture."""
        return None

    def _initial_positions(self, n: int) -> np.ndarray:
        return rand_points_in_circle(n, self.radius, self.rng)

    def _on_spawn(self, mask: np.ndarray) -> None:
        """Refresh per-dot state for newly (re)placed dots."""

    def _move_noise(self, noise: np.ndarray, step_scale: float) -> None:
        raise NotImplementedError("subclasses define the noise rule")

    # -- shared mechanics --------------------------------------------------

    def _assign_membership(self) -> None:
        """Choose the signal subset as an exact count of `coherence * n_dots`."""
        n_sig = int(round(self.n * self.coherence))
        is_signal = np.zeros(self.n, dtype=bool)
        if n_sig > 0:
            is_signal[self.rng.permutation(self.n)[:n_sig]] = True
        self.is_signal = is_signal

    def _resolve_luminance(
        self,
        *,
        background_lum: float,
        dot_contrast: float | None,
        dot_low_lum: float | None,
        dot_high_lum: float | None,
    ) -> tuple[float, float, np.ndarray]:
        if self.luminance_mode == "balanced":
            lo, hi = resolve_dot_luminances(
                background_lum=background_lum,
                dot_contrast=dot_contrast,
                dot_low_lum=dot_low_lum,
                dot_high_lum=dot_high_lum,
            )
            lum = assign_binary_luminances(
                self.n,
                background_lum=background_lum,
                low_lum=lo,
                high_lum=hi,
                rng=self.rng,
            )
            return float(lo), float(hi), lum

        value = uniform_dot_luminance(
            background_lum=background_lum,
            dot_contrast=dot_contrast,
            dot_high_lum=dot_high_lum,
        )
        return value, value, np.full(self.n, value, dtype=np.float32)

    def _respawn(self, mask: np.ndarray) -> None:
        count = int(np.count_nonzero(mask))
        if not count:
            return
        self.xys[mask] = rand_points_in_circle(count, self.radius, self.rng)
        self._on_spawn(mask)

    def _age_dots(self) -> None:
        if self.dot_life <= 0:
            return
        self.life -= 1
        dead = self.life <= 0
        if np.any(dead):
            self.life[dead] = self.dot_life
            self._respawn(dead)

    def _move(self) -> None:
        step_scale = self.speed_px_per_s / float(self.fps)
        signal = self.is_signal
        if np.any(signal):
            self.xys[signal] += self.signal_vec[None, :] * step_scale
        noise = ~signal
        if np.any(noise):
            self._move_noise(noise, step_scale)

    def _handle_exits(self) -> None:
        r2 = np.einsum("ij,ij->i", self.xys, self.xys)
        outside = r2 > (self.radius * self.radius)
        if np.any(outside):
            self._respawn(outside)

    def compute_opacity(self) -> np.ndarray:
        """Gaussian radial mask opacity per dot in [0, 1]."""
        if self.gauss_sigma <= 0:
            return np.ones(self.n, dtype=np.float32)
        r2 = np.einsum("ij,ij->i", self.xys, self.xys)
        alpha = np.exp(-0.5 * r2 / (float(self.gauss_sigma) ** 2))
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def step(self) -> None:
        """Advance the simulation by one frame."""
        self._age_dots()
        if self.signal_rule == "different":
            self._assign_membership()
        self._move()
        self._handle_exits()


class BrownianRDKEngine(BaseRDKEngine):
    """Random-walk noise: each noise dot steps at signal speed in a fresh random direction.

    Scase, Braddick & Raymond (1996) "random walk"; the BM algorithm of Pilly & Seitz (2009).
    """

    method_id = "brownian"

    def _move_noise(self, noise: np.ndarray, step_scale: float) -> None:
        count = int(np.count_nonzero(noise))
        self.xys[noise] += rand_unit_vectors(count, self.rng) * step_scale


class WhiteNoiseRDKEngine(BaseRDKEngine):
    """Random-position noise: each noise dot is relocated anywhere in the aperture each frame.

    Scase et al. (1996) "random position"; the WN algorithm of Pilly & Seitz (2009).
    """

    method_id = "white_noise"

    def _move_noise(self, noise: np.ndarray, step_scale: float) -> None:
        count = int(np.count_nonzero(noise))
        self.xys[noise] = rand_points_in_circle(count, self.radius, self.rng)


class RandomDirectionRDKEngine(BaseRDKEngine):
    """Random-direction noise: each noise dot keeps one random heading for its lifetime.

    Scase et al. (1996) "random direction". Because a heading is only redrawn when the dot
    respawns, `dot_life_frames` controls how quickly the noise directions refresh.
    """

    method_id = "random_direction"

    def _init_state(self) -> None:
        self.dot_dir = np.zeros((self.n, 2), dtype=np.float32)

    def _on_spawn(self, mask: np.ndarray) -> None:
        count = int(np.count_nonzero(mask))
        if count:
            self.dot_dir[mask] = rand_unit_vectors(count, self.rng)

    def _move_noise(self, noise: np.ndarray, step_scale: float) -> None:
        self.xys[noise] += self.dot_dir[noise] * step_scale


class MovshonNewsomeRDKEngine(BaseRDKEngine):
    """Three-sequence interleaved algorithm of Newsome & Paré (1988) / Britten et al. (1992).

    Dots are split into `n_sequences` interleaved groups; one group is redrawn per frame, so
    each dot updates every `n_sequences` frames and its coherent displacement is scaled
    accordingly. On update, each dot independently carries the signal with probability
    `coherence` or is relocated to a random position. Coherence is a per-dot probability
    here, not an exact count, which makes the stimulus inherently `different`-rule.
    """

    method_id = "movshon_newsome"

    def __init__(self, *, n_sequences: int = 3, **kwargs) -> None:
        self.n_sequences = int(max(1, n_sequences))
        kwargs["signal_rule"] = "different"
        super().__init__(**kwargs)

    def _init_state(self) -> None:
        self.sequence = self.rng.integers(0, self.n_sequences, size=self.n)
        self.frame_index = 0

    def _assign_membership(self) -> None:
        # Coherence is a per-dot probability in this algorithm, not an exact count.
        self.is_signal = self.rng.random(self.n) < self.coherence

    def step(self) -> None:
        self._age_dots()

        active = np.flatnonzero(self.sequence == (self.frame_index % self.n_sequences))
        self.is_signal = np.zeros(self.n, dtype=bool)
        if active.size:
            # A dot only updates every `n_sequences` frames, so it travels that much
            # further when it does.
            step_scale = self.speed_px_per_s / float(self.fps) * float(self.n_sequences)
            carries_signal = self.rng.random(active.size) < self.coherence
            signal_idx = active[carries_signal]
            noise_idx = active[~carries_signal]

            if signal_idx.size:
                self.is_signal[signal_idx] = True
                self.xys[signal_idx] += self.signal_vec[None, :] * step_scale
            if noise_idx.size:
                relocated = np.zeros(self.n, dtype=bool)
                relocated[noise_idx] = True
                self._respawn(relocated)

        self.frame_index += 1
        self._handle_exits()


class GaussianNonOverlapRDKEngine(BrownianRDKEngine):
    """The Brownian method plus a Gaussian envelope and a minimum dot separation.

    Motion is inherited from `BrownianRDKEngine` unchanged, so coherence stays comparable
    with that method. The additions are presentational: a soft (Gaussian) aperture instead
    of a hard-edged one, and blue-noise dot placement so no dot is hidden behind another.

    Separation is enforced by rejection sampling at spawn and by replanting violators after
    a motion step, which can displace a signal dot mid-trajectory; at the densities this is
    intended for that is rare, but it does mean effective coherence sits marginally below
    nominal. See docs/methods.md.
    """

    method_id = "gaussian_nonoverlap"

    def __init__(
        self,
        *,
        min_sep_px: float,
        max_reject_attempts: int = 20000,
        relax_factor: float = 0.95,
        **kwargs,
    ) -> None:
        min_sep = float(min_sep_px)
        if min_sep <= 0:
            raise ValueError("min_sep_px must be > 0")
        if not (0.0 < float(relax_factor) < 1.0):
            raise ValueError("relax_factor must be in (0, 1)")
        self.min_sep_px = min_sep
        self.max_reject_attempts = int(max(1, max_reject_attempts))
        self.relax_factor = float(relax_factor)
        super().__init__(**kwargs)

    def _default_gauss_sigma(self) -> float | None:
        return self.radius / 2.0

    def _place(self, count: int, existing: np.ndarray | None) -> np.ndarray:
        pts, achieved = poisson_points_in_circle(
            count,
            self.radius,
            self.min_sep_px,
            self.rng,
            existing=existing,
            max_attempts=self.max_reject_attempts,
            relax_factor=self.relax_factor,
        )
        # The field may be too dense for the requested spacing; keep whatever the
        # sampler could actually achieve so later frames stay self-consistent.
        self.min_sep_px = achieved
        return pts

    def _initial_positions(self, n: int) -> np.ndarray:
        return self._place(n, None)

    def _respawn(self, mask: np.ndarray) -> None:
        count = int(np.count_nonzero(mask))
        if not count:
            return
        self.xys[mask] = self._place(count, self.xys[~mask])
        self._on_spawn(mask)

    def _resolve_violations(self, max_iters: int = 3) -> None:
        for _ in range(int(max_iters)):
            diff = self.xys[:, None, :] - self.xys[None, :, :]
            dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
            np.fill_diagonal(dist_sq, np.inf)
            violating = np.any(dist_sq < self.min_sep_px * self.min_sep_px, axis=1)
            if not np.any(violating):
                return
            self._respawn(violating)

    def step(self) -> None:
        super().step()
        self._resolve_violations()


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
        ys, xs = np.mgrid[-rad : rad + 1, -rad : rad + 1]
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
