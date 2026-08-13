from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RDKParams:
    """Parameters that define the dot motion and field geometry.

    Units are pixels and seconds unless stated otherwise.

    Notes:
      - `method` selects the algorithm; see `rdk_generator.methods` and docs/methods.md.
        Parameters that a method does not use are ignored.
      - `speed_px_per_s` is interpreted as pixels/second.
      - `direction_deg` uses the PsychoPy convention: 0=right, 90=up.
      - `dot_life_frames <= 0` disables ageing, which is the canonical setting for the
        Movshon-Newsome method.
    """

    method: str = "gaussian_nonoverlap"

    n_dots: int = 300
    dot_size_px: int = 3
    speed_px_per_s: float = 120.0
    # None defers to the method's canonical default: 12 frames for most methods,
    # but 0 (no ageing) for `movshon_newsome`, which does not use dot lifetime.
    dot_life_frames: int | None = None
    direction_deg: float = 0.0
    coherence: float = 0.5

    field_diam_px: int = 300
    field_center_xy_px: tuple[float, float] = (0.0, 0.0)

    # Which dots carry the signal: "different" redraws the subset every frame,
    # "same" fixes it for the whole run. Ignored by `movshon_newsome`, whose
    # probabilistic re-selection is inherently a "different" rule.
    signal_rule: str = "different"

    # `movshon_newsome` only: number of interleaved dot sequences (canonically 3).
    n_sequences: int = 3

    # `gaussian_nonoverlap` only.
    # Gaussian opacity envelope width; None uses a quarter of the field diameter.
    gauss_sigma_px: float | None = None
    # Minimum centre-to-centre dot distance; None uses 1.1 x dot size.
    min_sep_px: float | None = None


@dataclass(frozen=True, slots=True)
class RenderParams:
    """Parameters that define frame rendering and luminance."""

    width_px: int = 512
    height_px: int = 512
    fps: int = 60
    duration_s: float = 1.0

    # Background luminance in [0, 1] (0=black, 1=white)
    background_lum: float = 0.5

    # "uniform": every dot shares one luminance, the classic arrangement.
    # "balanced": dark/bright dots mixed so the field's mean luminance equals the
    #   background, leaving motion as the only cue. Available to every method.
    luminance_mode: str = "uniform"

    # `dot_contrast` is an absolute luminance span, not Michelson contrast:
    #   dot_high = clip(background + dot_contrast/2)
    #   dot_low  = clip(background - dot_contrast/2)
    # For white dots on black, pair background_lum=0.0 with dot_contrast=2.0.
    #
    # Explicit luminances override it: in "uniform" mode set `dot_high_lum`; in
    # "balanced" mode set both low and high.
    dot_contrast: float = 1.0
    dot_low_lum: float | None = None
    dot_high_lum: float | None = None

    seed: int | None = None
