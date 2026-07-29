---
name: rdk-domain
description: Psychophysics background for the RDK (random dot kinematogram) stimulus this repo generates — coherence, dot life, Gaussian envelope, luminance balancing, and coordinate conventions. Use before modifying stimulus math in core.py, luminance.py, sampling.py, or psychopy_adapter.py, or when reasoning about whether generated output is perceptually correct.
---

## What an RDK is here

A circular field of small dots on a grey background. Each frame, a fraction of dots (**coherence**, 0–1) moves in a common **signal** direction; the rest (**noise** dots) each move in an independently random direction. Every dot has a **dot life** in frames; when it expires (or the dot exits the field) the dot respawns at a uniform-over-area random position inside the circle (`rand_points_in_circle`: `r = R·sqrt(u)`, angle uniform — plain `r = R·u` would oversample the center).

Two features distinguish this from a stock `psychopy.visual.DotStim`:

1. **Gaussian envelope** — per-dot opacity `alpha = exp(-0.5 · r² / sigma²)` where `r` is distance from field center (`RDKEngine.compute_opacity`). Default `sigma = radius / 2`. `sigma² <= 0` means no envelope (alpha = 1).
2. **Luminance balancing** — each dot gets one of two luminances `low`/`high` straddling the background. The fraction of high dots is `p_high = (bg − low) / (high − low)` and the *exact count* `round(n · p_high)` is used (not sampled) so the mean luminance inside the field equals the background. Result: the field is invisible on average; only the motion is detectable.

## dot_contrast semantics (trap)

`resolve_dot_luminances` treats `dot_contrast` as an **absolute luminance span**: `low, high = bg − c/2, bg + c/2`, clipped to [0, 1]. It is **not** Michelson contrast. If clipping collapses `low == high`, it silently falls back to full range `(0.0, 1.0)`. Explicit `dot_low_lum`/`dot_high_lum` override the contrast-derived values.

## Conventions that cause bugs if forgotten

- **Direction**: PsychoPy convention — `direction_deg = 0` is rightward, `90` is **up** (counter-clockwise). Velocity: `(cos θ, sin θ) · speed_px_per_s / fps`.
- **Coordinates**: dot positions are stored **relative to field center**, y-up. `FrameRenderer.render` converts to pixel space with a Y flip: `ox = width/2 + cx`, `oy = height/2 − cy`, then `px = ox + x`, `py = oy − y`.
- **Engine selection**: `export.render_frames` uses `NonOverlappingRDKEngine` iff `rdk.min_sep_px is not None`, else `RDKEngine`. Recommended `min_sep_px ≈ 1.1 × dot_size_px`. Non-overlap is enforced by rejection sampling at spawn/respawn only, not during motion.
- **Frames**: 8-bit single-channel grayscale, shape `(height, width)`; luminance values are floats in [0, 1] until the final `uint8` quantization.
- **Reproducibility**: everything is seeded through `numpy.random.default_rng(seed)`; identical `RDKParams` + `RenderParams` (recorded in the `.json` sidecar) must reproduce identical frames.

## Layer boundary

Stimulus math changes normally belong in the headless layer (`core.py`, `luminance.py`, `sampling.py`). `psychopy_adapter.py` (`BalancedGaussianRDK`) duplicates the motion/life/membership logic for realtime PsychoPy presentation — when changing motion or luminance semantics, check whether the adapter needs the same change. The adapter is untested (`# pragma: no cover`) and must stay importable without PsychoPy installed.
