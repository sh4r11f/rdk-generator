from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RDKParams:
    """Parameters that define the dot motion and field geometry.

    Units are pixels and seconds unless stated otherwise.

    Notes:
      - `speed` is interpreted as pixels/second.
      - `direction_deg` uses the PsychoPy convention: 0=right, 90=up.
    """

    n_dots: int = 300
    dot_size_px: int = 3
    speed_px_per_s: float = 120.0
    dot_life_frames: int = 12
    direction_deg: float = 0.0
    coherence: float = 0.5

    field_diam_px: int = 300
    field_center_xy_px: tuple[float, float] = (0.0, 0.0)

    gauss_sigma_px: float | None = None
    reassign_life: bool = True

    # Optional non-overlap mode (for movie rendering):
    # When enabled, dot centers are kept at least `min_sep_px` apart.
    # If None, dots may overlap.
    #
    # Recommended default if you enable it: ~1.1 * dot_size_px
    min_sep_px: float | None = None


@dataclass(frozen=True, slots=True)
class RenderParams:
    """Parameters that define frame rendering and luminance balancing."""

    width_px: int = 512
    height_px: int = 512
    fps: int = 60
    duration_s: float = 1.0

    # Background luminance in [0, 1] (0=black, 1=white)
    background_lum: float = 0.5

    # Dot luminance settings
    # If `dot_contrast` is set, dots are rendered with two luminance levels
    # around the background and mixed to keep the *mean* at background.
    #
    # dot_high = clip(background + dot_contrast/2)
    # dot_low  = clip(background - dot_contrast/2)
    #
    # If you want explicit luminances, set both low/high and set dot_contrast=None.
    dot_contrast: float = 1.0
    dot_low_lum: float | None = None
    dot_high_lum: float | None = None

    seed: int | None = None
