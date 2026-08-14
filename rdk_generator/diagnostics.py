"""Measure what a stimulus actually did, rather than what its parameters asked for.

The two are not the same. Dots occlude each other, so the visible dot count is below the
nominal one. Coherence is a per-frame Bernoulli draw in the Movshon-Newsome family, so the
realised signal fraction wanders. Respawns and non-overlap replanting destroy coherent
displacements that the engine intended to make. Sampling bugs show up as a radial density
gradient. None of that is visible by staring at the movie.

`compute_diagnostics` runs the simulation once and reports all of it. It is pure NumPy;
`figure_png` draws the result and is the only part that needs matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import FrameRenderer, MovshonNewsomeRDKEngine
from .export import build_engine
from .params import RDKParams, RenderParams

# Diagnostics are O(n_dots^2) per frame and render every frame, so they run on a bounded
# window rather than the whole clip.
DEFAULT_MAX_FRAMES = 90

# A displacement this many times the expected coherent step is a respawn, not motion.
RELOCATION_FACTOR = 4.0


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Per-frame and pooled statistics for one stimulus.

    Attributes:
      method: Method id the stimulus was generated with.
      n_frames: Frames measured (may be fewer than the clip's full length).
      n_dots: Dot count.
      dot_size_px: Drawn dot diameter, the threshold for "these two overlap".
      expected_step_px: Distance a coherent dot travels per update.
      nn_distances: Pooled nearest-neighbour distances across frames.
      overlap_pairs: Per frame, pairs of dots closer together than one dot diameter.
      occluded_dots: Per frame, dots whose centre is within half a diameter of another,
        i.e. substantially hidden.
      step_magnitudes: Pooled per-dot displacement magnitudes, relocations excluded.
      step_angles_deg: Pooled per-dot displacement directions, relocations excluded.
      relocations: Per frame, dots that were teleported rather than moved.
      radial_edges / radial_density / radial_density_weighted: Dots per unit area against
        distance from field centre, unweighted and weighted by the opacity envelope.
      signal_fraction: Per frame, fraction of dots the engine flagged as carrying signal.
      signal_delivered: Per frame, fraction of those dots that actually made a clean
        coherent displacement. A dot that teleported the same frame delivered no coherent
        motion, whatever it was flagged as.
      lost_to_lifecycle: Per frame, fraction of intended-signal dots that instead aged out
        or left the aperture. This is by design and happens in every method.
      lost_to_replanting: Per frame, fraction of intended-signal dots displaced by the
        minimum-separation constraint. Nonzero only for our variants; this is what the
        modification costs.
      mean_luminance: Per frame, mean of the rendered frame in [0, 1].
      background_lum: Background the frames were drawn against, for comparison.
      nominal_coherence: The coherence that was requested.
    """

    method: str
    n_frames: int
    n_dots: int
    dot_size_px: float
    expected_step_px: float
    nn_distances: np.ndarray
    overlap_pairs: np.ndarray
    occluded_dots: np.ndarray
    step_magnitudes: np.ndarray
    step_angles_deg: np.ndarray
    relocations: np.ndarray
    radial_edges: np.ndarray
    radial_density: np.ndarray
    radial_density_weighted: np.ndarray
    signal_fraction: np.ndarray
    signal_delivered: np.ndarray
    lost_to_lifecycle: np.ndarray
    lost_to_replanting: np.ndarray
    mean_luminance: np.ndarray
    background_lum: float
    nominal_coherence: float
    min_sep_px: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, float | str | None]:
        """The handful of numbers worth reading before looking at any plot."""
        visible = 1.0 - float(np.mean(self.occluded_dots)) / max(1, self.n_dots)
        return {
            "method": self.method,
            "n_frames": self.n_frames,
            "n_dots": self.n_dots,
            "overlapping_pairs_per_frame": float(np.mean(self.overlap_pairs)),
            "occluded_dots_per_frame": float(np.mean(self.occluded_dots)),
            "visible_dot_fraction": visible,
            "min_nn_distance_px": (
                float(np.min(self.nn_distances)) if self.nn_distances.size else None
            ),
            "median_nn_distance_px": (
                float(np.median(self.nn_distances)) if self.nn_distances.size else None
            ),
            "nominal_coherence": self.nominal_coherence,
            "mean_signal_fraction": float(np.mean(self.signal_fraction)),
            "signal_delivered": _mean_ignoring_nan(self.signal_delivered),
            "lost_to_lifecycle": _mean_ignoring_nan(self.lost_to_lifecycle),
            "lost_to_replanting": _mean_ignoring_nan(self.lost_to_replanting),
            "relocations_per_frame": float(np.mean(self.relocations)),
            "mean_luminance": float(np.mean(self.mean_luminance)),
            "luminance_sd": float(np.std(self.mean_luminance)),
            "background_lum": self.background_lum,
            "min_sep_px": self.min_sep_px,
        }


def _mean_ignoring_nan(values: np.ndarray, default: float = 0.0) -> float:
    """Mean of the non-NaN entries, or `default` when there are none.

    Frames with no signal dots at all leave NaN, which `np.nanmean` warns about and
    would put straight into the JSON summary.
    """
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else default


def _nearest_neighbour(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour distance per dot, and the full pairwise distance matrix.

    With a single dot there is no neighbour, so the distances come back empty rather
    than infinite, which would poison both the histogram and the JSON summary.
    """
    if xy.shape[0] < 2:
        return np.zeros(0), np.zeros((xy.shape[0], xy.shape[0]))
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1), dist


def compute_diagnostics(
    rdk: RDKParams,
    render: RenderParams,
    *,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
) -> Diagnostics:
    """Run the stimulus and measure it. Pure NumPy; no plotting."""
    engine = build_engine(rdk, render)
    renderer = FrameRenderer(
        width_px=render.width_px,
        height_px=render.height_px,
        background_lum=render.background_lum,
        field_center_xy_px=rdk.field_center_xy_px,
        dot_size_px=rdk.dot_size_px,
    )

    total = max(1, int(round(render.duration_s * render.fps)))
    n_frames = total if max_frames is None else min(total, max(1, int(max_frames)))

    per_frame_step = rdk.speed_px_per_s / float(render.fps)
    sequences = (
        getattr(engine, "n_sequences", 1) if isinstance(engine, MovshonNewsomeRDKEngine) else 1
    )
    expected_step = per_frame_step * sequences
    relocation_threshold = RELOCATION_FACTOR * max(expected_step, 1e-6)
    dot_size = float(rdk.dot_size_px)

    nn_pool: list[np.ndarray] = []
    mag_pool: list[np.ndarray] = []
    ang_pool: list[np.ndarray] = []
    radii_pool: list[np.ndarray] = []
    alpha_pool: list[np.ndarray] = []

    overlap_pairs = np.zeros(n_frames)
    occluded = np.zeros(n_frames)
    relocations = np.zeros(n_frames)
    signal_fraction = np.zeros(n_frames)
    signal_delivered = np.full(n_frames, np.nan)
    lost_to_lifecycle = np.full(n_frames, np.nan)
    lost_to_replanting = np.full(n_frames, np.nan)
    mean_lum = np.zeros(n_frames)

    # Attributing lost signal to a cause needs to know why each dot was teleported.
    engine.record_respawns = True

    for i in range(n_frames):
        xy_before = engine.xys.copy()
        alpha = engine.compute_opacity()

        # Render the state that is actually shown, matching render_frames' ordering.
        frame = renderer.render(xy_before, engine.dot_lum, alpha)
        mean_lum[i] = frame.mean() / 255.0

        nn, dist = _nearest_neighbour(xy_before)
        nn_pool.append(nn)
        overlap_pairs[i] = np.count_nonzero(dist < dot_size) / 2.0
        occluded[i] = np.count_nonzero(nn < dot_size / 2.0)
        radii_pool.append(np.linalg.norm(xy_before, axis=1))
        alpha_pool.append(alpha)

        engine.step()

        disp = engine.xys - xy_before
        mag = np.linalg.norm(disp, axis=1)
        relocated = mag > relocation_threshold
        relocations[i] = np.count_nonzero(relocated)

        real_motion = ~relocated & (mag > 1e-9)
        mag_pool.append(mag[real_motion])
        ang_pool.append(np.degrees(np.arctan2(disp[real_motion, 1], disp[real_motion, 0])))

        # `is_signal` is assigned during step(), so it describes the motion just applied.
        intended = engine.is_signal
        signal_fraction[i] = float(intended.mean())
        if np.any(intended):
            target = engine.signal_vec * expected_step
            tolerance = 0.05 * max(expected_step, 1.0)
            delivered = np.all(np.abs(disp - target[None, :]) < tolerance, axis=1)
            signal_delivered[i] = float(delivered[intended].mean())

            respawned = engine.respawned
            replanted = respawned.get("replanted")
            lifecycle = np.zeros(engine.n, dtype=bool)
            for reason in ("aged", "exited"):
                mask = respawned.get(reason)
                if mask is not None:
                    lifecycle |= mask
            lost_to_lifecycle[i] = float(lifecycle[intended].mean())
            lost_to_replanting[i] = (
                float(replanted[intended].mean()) if replanted is not None else 0.0
            )

    radii = np.concatenate(radii_pool)
    alphas = np.concatenate(alpha_pool)
    radius = engine.radius
    # Equal-area rings, so every bin holds the same expected count and the profile
    # is flat for correct uniform-over-area sampling.
    edges = radius * np.sqrt(np.linspace(0.0, 1.0, 21))
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    counts, _ = np.histogram(radii, bins=edges)
    weighted, _ = np.histogram(radii, bins=edges, weights=alphas)
    frames_measured = float(n_frames)

    notes: list[str] = []
    if n_frames < total:
        notes.append(f"Measured the first {n_frames} of {total} frames.")
    if sequences > 1:
        notes.append(
            f"Only 1 in {sequences} dots updates per frame, so the signal fraction is "
            f"about coherence / {sequences}."
        )

    return Diagnostics(
        method=rdk.method,
        n_frames=n_frames,
        n_dots=engine.n,
        dot_size_px=dot_size,
        expected_step_px=expected_step,
        nn_distances=np.concatenate(nn_pool),
        overlap_pairs=overlap_pairs,
        occluded_dots=occluded,
        step_magnitudes=np.concatenate(mag_pool) if mag_pool else np.zeros(0),
        step_angles_deg=np.concatenate(ang_pool) if ang_pool else np.zeros(0),
        relocations=relocations,
        radial_edges=edges,
        radial_density=counts / ring_area / frames_measured,
        radial_density_weighted=weighted / ring_area / frames_measured,
        signal_fraction=signal_fraction,
        signal_delivered=signal_delivered,
        lost_to_lifecycle=lost_to_lifecycle,
        lost_to_replanting=lost_to_replanting,
        mean_luminance=mean_lum,
        background_lum=float(render.background_lum),
        nominal_coherence=float(rdk.coherence),
        min_sep_px=getattr(engine, "min_sep_px", None),
        notes=tuple(notes),
    )


# --- plotting ---------------------------------------------------------------

_INK = "#ece7df"
_MUTED = "#a8a096"
_FAINT = "#776f66"
_BG = "#141416"
_ACCENT = "#d9b978"
_SIGNAL = "#8fbf9f"
_ALERT = "#cf8b7d"


def _hist(ax, data: np.ndarray, *, bins: int = 60, **kwargs) -> None:
    """Histogram that tolerates empty and single-valued data.

    Speed-matched noise makes every step exactly the same length, so the step-size
    series legitimately has zero range; NumPy refuses to bin that without an explicit
    range.
    """
    if data.size == 0:
        return
    low, high = float(np.min(data)), float(np.max(data))
    # Float32 jitter can leave a range that is nonzero but far too narrow to split into
    # `bins` representable edges, so widen anything below a magnitude-relative floor.
    floor = max(abs(low), abs(high), 1.0) * 1e-3
    if high - low < floor:
        pad = max(floor, 0.5)
        low, high = low - pad, high + pad
    ax.hist(data, bins=bins, range=(low, high), **kwargs)


def figure_png(diag: Diagnostics, *, dpi: int = 110) -> bytes:
    """Render the diagnostic panel as PNG bytes.

    Imported lazily so the headless generator keeps working without matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), facecolor=_BG)
    fig.subplots_adjust(hspace=0.68, wspace=0.30, top=0.895, bottom=0.05, left=0.09, right=0.96)

    for ax in axes.flat:
        ax.set_facecolor(_BG)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_FAINT)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.grid(alpha=0.12, color=_MUTED, linewidth=0.6)
        ax.set_axisbelow(True)

    def title(ax, text, sub=""):
        ax.set_title(text, color=_INK, fontsize=10.5, loc="left", pad=25 if sub else 10)
        if sub:
            ax.text(0, 1.015, sub, transform=ax.transAxes, color=_FAINT, fontsize=8, va="bottom")

    frames = np.arange(diag.n_frames)
    s = diag.summary()

    # 1. Nearest-neighbour distances: where overlap comes from.
    ax = axes[0, 0]
    _hist(ax, diag.nn_distances, color=_ACCENT, alpha=0.85)
    ax.axvline(
        diag.dot_size_px, color=_ALERT, lw=1.3, ls="--", label=f"dot size {diag.dot_size_px:g}px"
    )
    if diag.min_sep_px:
        ax.axvline(
            diag.min_sep_px, color=_SIGNAL, lw=1.3, ls=":", label=f"min sep {diag.min_sep_px:.2f}px"
        )
    ax.set_xlabel("nearest-neighbour distance (px)", color=_MUTED, fontsize=9)
    ax.set_ylabel("dot-frames", color=_MUTED, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(ax, "Dot spacing", "anything left of the dashed line is an overlap")

    # 2. Overlap over time.
    ax = axes[0, 1]
    ax.plot(frames, diag.overlap_pairs, color=_ALERT, lw=1.3, label="overlapping pairs")
    ax.plot(frames, diag.occluded_dots, color=_ACCENT, lw=1.3, label="mostly-hidden dots")
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("count", color=_MUTED, fontsize=9)
    peak = max(diag.overlap_pairs.max(), diag.occluded_dots.max())
    if peak <= 0:
        ax.set_ylim(-0.5, 5)
        ax.text(
            0.5,
            0.55,
            "no dot ever overlaps another",
            transform=ax.transAxes,
            ha="center",
            color=_SIGNAL,
            fontsize=10,
        )
    else:
        ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(
        ax,
        "Occlusion over time",
        f"{s['visible_dot_fraction'] * 100:.1f}% of dots distinctly visible",
    )

    # 3. Step magnitude: the noise rule's fingerprint.
    ax = axes[1, 0]
    _hist(ax, diag.step_magnitudes, color=_ACCENT, alpha=0.85)
    ax.axvline(
        diag.expected_step_px,
        color=_SIGNAL,
        lw=1.3,
        ls="--",
        label=f"coherent step {diag.expected_step_px:.2f}px",
    )
    ax.set_xlabel("displacement per frame (px)", color=_MUTED, fontsize=9)
    ax.set_ylabel("dot-frames", color=_MUTED, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(ax, "Step size", "relocations excluded; speed-matched noise sits on the line")

    # 4. Step direction.
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(4, 2, 4, projection="polar", facecolor=_BG)
    if diag.step_angles_deg.size:
        counts, edges = np.histogram(
            np.radians(diag.step_angles_deg % 360), bins=36, range=(0, 2 * np.pi)
        )
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=_ACCENT,
            alpha=0.85,
            edgecolor=_BG,
            linewidth=0.5,
        )
    ax.tick_params(colors=_MUTED, labelsize=7.5)
    ax.grid(alpha=0.15, color=_MUTED)
    ax.set_title(
        "Step direction\na spike on the signal, a pedestal of noise",
        color=_INK,
        fontsize=10.5,
        pad=14,
    )

    # 5. Radial density: uniform-over-area sampling, and the envelope.
    ax = axes[2, 0]
    centres = 0.5 * (diag.radial_edges[:-1] + diag.radial_edges[1:])
    ax.plot(centres, diag.radial_density, color=_ACCENT, lw=1.5, label="dot density")
    ax.plot(
        centres,
        diag.radial_density_weighted,
        color=_SIGNAL,
        lw=1.5,
        ls="--",
        label="weighted by opacity",
    )
    ax.set_xlabel("distance from centre (px)", color=_MUTED, fontsize=9)
    ax.set_ylabel("dots / px²", color=_MUTED, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(ax, "Radial density", "flat = uniform over area; the dashed line shows the envelope")

    # 6. Relocations: how much of the field is teleporting.
    ax = axes[2, 1]
    ax.plot(frames, diag.relocations, color=_ACCENT, lw=1.3)
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("dots relocated", color=_MUTED, fontsize=9)
    title(
        ax,
        "Relocations per frame",
        f"{s['relocations_per_frame']:.1f} of {diag.n_dots} dots on average",
    )

    # 7. Coherence: asked for, delivered, and survived.
    ax = axes[3, 0]
    ax.plot(frames, diag.signal_fraction, color=_ACCENT, lw=1.3, label="flagged as signal")
    ax.plot(frames, diag.signal_delivered, color=_SIGNAL, lw=1.3, label="delivered a coherent step")
    ax.plot(frames, diag.lost_to_replanting, color=_ALERT, lw=1.3, label="lost to replanting")
    ax.axhline(
        diag.nominal_coherence,
        color=_MUTED,
        lw=1.0,
        ls="--",
        label=f"nominal {diag.nominal_coherence:g}",
    )
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("fraction of signal dots", color=_MUTED, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(
        ax,
        "Coherence delivery",
        f"{s['lost_to_lifecycle'] * 100:.0f}% lost to dot life / aperture exits (by design)",
    )

    # 8. Luminance: is there a flicker confound?
    ax = axes[3, 1]
    ax.plot(frames, diag.mean_luminance, color=_ACCENT, lw=1.3, label="frame mean")
    ax.axhline(diag.background_lum, color=_SIGNAL, lw=1.0, ls="--", label="background")
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("mean luminance", color=_MUTED, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)
    title(ax, "Frame luminance", f"SD {s['luminance_sd']:.4f} — flat means no flicker cue")

    header = (
        f"{diag.method}  ·  {diag.n_dots} dots  ·  {diag.n_frames} frames  ·  "
        f"coherence {diag.nominal_coherence:g}"
    )
    fig.suptitle(header, color=_INK, fontsize=13, y=0.968, x=0.09, ha="left")
    if diag.notes:
        fig.text(0.09, 0.945, "  ".join(diag.notes), color=_FAINT, fontsize=8.5, ha="left")

    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def write_diagnostics_png(
    out_path,
    *,
    rdk: RDKParams,
    render: RenderParams,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
):
    """Measure a stimulus and write its diagnostic panel next to the exported clip."""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag = compute_diagnostics(rdk, render, max_frames=max_frames)
    out_path.write_bytes(figure_png(diag))
    return out_path
