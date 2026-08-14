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


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Per-frame and pooled statistics for one stimulus.

    Attributes:
      method: Method id the stimulus was generated with.
      n_frames: Frames measured (may be fewer than the clip's full length).
      total_frames: Frames the clip actually has, so callers can tell whether the
        measurement covered all of it.
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
    total_frames: int
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

    @property
    def truncated(self) -> bool:
        """Whether the measurement covered only part of the clip."""
        return self.n_frames < self.total_frames

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

        # Which dots teleported is recorded by the engine, so it is exact. Inferring it
        # from displacement size instead would misfile every respawn that happened to
        # land near its old position -- 1.7% of them in a small aperture.
        respawned = engine.respawned
        teleported = np.zeros(engine.n, dtype=bool)
        for mask in respawned.values():
            teleported |= mask
        relocations[i] = np.count_nonzero(teleported)

        real_motion = ~teleported & (mag > 1e-9)
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
        total_frames=total,
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


def _style_axes(ax, *, polar: bool = False) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(alpha=0.12 if not polar else 0.15, color=_MUTED, linewidth=0.6)
    ax.set_axisbelow(True)
    if polar:
        return
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_FAINT)


def _set_title(ax, text: str, sub: str = "", *, polar: bool = False) -> None:
    if polar:
        ax.set_title(f"{text}\n{sub}" if sub else text, color=_INK, fontsize=10.5, pad=16)
        return
    ax.set_title(text, color=_INK, fontsize=10.5, loc="left", pad=25 if sub else 10)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=_FAINT, fontsize=8, va="bottom")


def _legend(ax) -> None:
    ax.legend(fontsize=7.5, frameon=False, labelcolor=_MUTED)


# Each panel draws itself onto a single axes, so the combined report and the
# individually exported figures come from exactly the same code.


def _panel_spacing(ax, diag: Diagnostics, s: dict) -> None:
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
    _legend(ax)
    _set_title(ax, "Dot spacing", "anything left of the dashed line is an overlap")


def _panel_occlusion(ax, diag: Diagnostics, s: dict) -> None:
    frames = np.arange(diag.n_frames)
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
    _legend(ax)
    _set_title(
        ax,
        "Occlusion over time",
        f"{s['visible_dot_fraction'] * 100:.1f}% of dots distinctly visible",
    )


def _panel_step_size(ax, diag: Diagnostics, s: dict) -> None:
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
    _legend(ax)
    _set_title(ax, "Step size", "relocations excluded; speed-matched noise sits on the line")


def _panel_direction(ax, diag: Diagnostics, s: dict) -> None:
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
    _set_title(ax, "Step direction", "a spike on the signal, a pedestal of noise", polar=True)


def _panel_density(ax, diag: Diagnostics, s: dict) -> None:
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
    ax.set_ylabel("dots / px\u00b2", color=_MUTED, fontsize=9)
    _legend(ax)
    _set_title(ax, "Radial density", "flat = uniform over area; the dashed line shows the envelope")


def _panel_relocations(ax, diag: Diagnostics, s: dict) -> None:
    ax.plot(np.arange(diag.n_frames), diag.relocations, color=_ACCENT, lw=1.3)
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("dots relocated", color=_MUTED, fontsize=9)
    _set_title(
        ax,
        "Relocations per frame",
        f"{s['relocations_per_frame']:.1f} of {diag.n_dots} dots on average",
    )


def _panel_coherence(ax, diag: Diagnostics, s: dict) -> None:
    frames = np.arange(diag.n_frames)
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
    _legend(ax)
    _set_title(
        ax,
        "Coherence delivery",
        f"{s['lost_to_lifecycle'] * 100:.0f}% lost to dot life / aperture exits (by design)",
    )


def _panel_luminance(ax, diag: Diagnostics, s: dict) -> None:
    ax.plot(
        np.arange(diag.n_frames), diag.mean_luminance, color=_ACCENT, lw=1.3, label="frame mean"
    )
    ax.axhline(diag.background_lum, color=_SIGNAL, lw=1.0, ls="--", label="background")
    ax.set_xlabel("frame", color=_MUTED, fontsize=9)
    ax.set_ylabel("mean luminance", color=_MUTED, fontsize=9)
    _legend(ax)
    _set_title(
        ax, "Frame luminance", f"SD {s['luminance_sd']:.4f} \u2014 flat means no flicker cue"
    )


# Keys match `DIAGNOSTIC_GUIDE`, so the report and its explanations cannot drift apart.
PANELS: tuple[tuple[str, str, object, bool], ...] = (
    ("spacing", "Dot spacing", _panel_spacing, False),
    ("occlusion", "Occlusion over time", _panel_occlusion, False),
    ("step_size", "Step size", _panel_step_size, False),
    ("direction", "Step direction", _panel_direction, True),
    ("density", "Radial density", _panel_density, False),
    ("relocations", "Relocations per frame", _panel_relocations, False),
    ("coherence", "Coherence delivery", _panel_coherence, False),
    ("luminance", "Frame luminance", _panel_luminance, False),
)

PANEL_KEYS: tuple[str, ...] = tuple(key for key, _, _, _ in PANELS)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _render(fig, plt, fmt: str, dpi: int) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def _header(diag: Diagnostics) -> str:
    return (
        f"{diag.method}  \u00b7  {diag.n_dots} dots  \u00b7  {diag.n_frames} frames  \u00b7  "
        f"coherence {diag.nominal_coherence:g}"
    )


def figure_bytes(diag: Diagnostics, *, fmt: str = "png", dpi: int = 110) -> bytes:
    """Render the full eight-panel report. `fmt` may be png, pdf or svg.

    matplotlib is imported lazily so the headless generator works without it.
    """
    plt = _pyplot()
    summary = diag.summary()

    fig = plt.figure(figsize=(11, 13), facecolor=_BG)
    fig.subplots_adjust(hspace=0.68, wspace=0.30, top=0.895, bottom=0.05, left=0.09, right=0.96)

    for i, (_key, _title, draw, polar) in enumerate(PANELS):
        ax = fig.add_subplot(4, 2, i + 1, projection="polar" if polar else None)
        _style_axes(ax, polar=polar)
        draw(ax, diag, summary)

    fig.suptitle(_header(diag), color=_INK, fontsize=13, y=0.968, x=0.09, ha="left")
    if diag.notes:
        fig.text(0.09, 0.945, "  ".join(diag.notes), color=_FAINT, fontsize=8.5, ha="left")
    return _render(fig, plt, fmt, dpi)


def figure_png(diag: Diagnostics, *, dpi: int = 110) -> bytes:
    """Render the full report as PNG bytes."""
    return figure_bytes(diag, fmt="png", dpi=dpi)


def panel_bytes(diag: Diagnostics, key: str, *, fmt: str = "png", dpi: int = 200) -> bytes:
    """Render one panel on its own, sized for dropping into a paper or slide."""
    match = [p for p in PANELS if p[0] == key]
    if not match:
        raise KeyError(f"unknown panel {key!r}; expected one of {PANEL_KEYS}")
    _key, _title, draw, polar = match[0]

    plt = _pyplot()
    fig = plt.figure(figsize=(6.4, 5.0) if polar else (6.8, 4.6), facecolor=_BG)
    ax = fig.add_subplot(111, projection="polar" if polar else None)
    _style_axes(ax, polar=polar)
    draw(ax, diag, diag.summary())
    fig.text(0.02, 0.015, _header(diag), color=_FAINT, fontsize=7, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _render(fig, plt, fmt, dpi)


def per_frame_csv(diag: Diagnostics) -> str:
    columns = {
        "frame": np.arange(diag.n_frames),
        "overlapping_pairs": diag.overlap_pairs,
        "occluded_dots": diag.occluded_dots,
        "relocations": diag.relocations,
        "signal_fraction": diag.signal_fraction,
        "signal_delivered": diag.signal_delivered,
        "lost_to_lifecycle": diag.lost_to_lifecycle,
        "lost_to_replanting": diag.lost_to_replanting,
        "mean_luminance": diag.mean_luminance,
    }
    lines = [",".join(columns)]
    for i in range(diag.n_frames):
        lines.append(",".join(f"{columns[c][i]:.6g}" for c in columns))
    return "\n".join(lines) + "\n"


def radial_csv(diag: Diagnostics) -> str:
    edges = diag.radial_edges
    lines = ["ring_inner_px,ring_outer_px,density_dots_per_px2,density_weighted_by_opacity"]
    for i in range(len(diag.radial_density)):
        lines.append(
            f"{edges[i]:.6g},{edges[i + 1]:.6g},"
            f"{diag.radial_density[i]:.6g},{diag.radial_density_weighted[i]:.6g}"
        )
    return "\n".join(lines) + "\n"


def figures_zip(
    diag: Diagnostics,
    *,
    dpi: int = 300,
    rdk: RDKParams | None = None,
    render: RenderParams | None = None,
) -> bytes:
    """Bundle publication-quality figures and the numbers behind them.

    Contains the combined report and every panel separately, each as a high-resolution
    PNG and as vector PDF, plus the measured series as CSV so the figures can be redrawn
    in another style, and the parameters that produced them.
    """
    import json
    import zipfile
    from dataclasses import asdict
    from io import BytesIO

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("diagnostics.png", figure_bytes(diag, fmt="png", dpi=dpi))
        archive.writestr("diagnostics.pdf", figure_bytes(diag, fmt="pdf"))

        for index, (key, _title, _draw, _polar) in enumerate(PANELS, start=1):
            archive.writestr(
                f"panels/{index:02d}_{key}.png", panel_bytes(diag, key, fmt="png", dpi=dpi)
            )
            archive.writestr(f"panels/{index:02d}_{key}.pdf", panel_bytes(diag, key, fmt="pdf"))

        archive.writestr("data/per_frame.csv", per_frame_csv(diag))
        archive.writestr("data/radial_density.csv", radial_csv(diag))
        archive.writestr("data/summary.json", json.dumps(diag.summary(), indent=2))
        if rdk is not None and render is not None:
            archive.writestr(
                "data/params.json",
                json.dumps({"rdk": asdict(rdk), "render": asdict(render)}, indent=2),
            )

        notes = " ".join(diag.notes) or "The whole clip was measured."
        archive.writestr(
            "README.txt",
            "RDK stimulus diagnostics\n"
            "========================\n\n"
            f"{_header(diag)}\n\n"
            f"{notes}\n\n"
            f"diagnostics.png   combined report, {dpi} dpi\n"
            "diagnostics.pdf   the same, as vector\n"
            "panels/           each panel separately, PNG and vector PDF\n"
            "data/per_frame.csv       the per-frame series behind the time plots\n"
            "data/radial_density.csv  the radial density profile\n"
            "data/summary.json        the headline numbers\n"
            "data/params.json         the parameters that produced this stimulus\n\n"
            "See docs/diagnostics.md for what each panel measures and how it was validated.\n",
        )
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


def write_figures_zip(
    out_path,
    *,
    rdk: RDKParams,
    render: RenderParams,
    dpi: int = 300,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
):
    """Measure a stimulus and write the full figure bundle."""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag = compute_diagnostics(rdk, render, max_frames=max_frames)
    out_path.write_bytes(figures_zip(diag, dpi=dpi, rdk=rdk, render=render))
    return out_path


# --- the guide --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiagnosticDoc:
    """User-facing explanation of one diagnostic: what it computes and what it catches."""

    key: str
    title: str
    measures: str
    detects: str
    reading: str
    validation: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "title": self.title,
            "measures": self.measures,
            "detects": self.detects,
            "reading": self.reading,
            "validation": self.validation,
        }


DIAGNOSTIC_GUIDE: tuple[DiagnosticDoc, ...] = (
    DiagnosticDoc(
        key="spacing",
        title="Dot spacing",
        measures=(
            "Every frame, the full pairwise distance matrix between dot centres, from which "
            "each dot's distance to its nearest neighbour is taken. Distances from all "
            "measured frames are pooled into one histogram. A pair counts as overlapping "
            "when it is closer than one dot diameter, since that is when the drawn discs "
            "intersect."
        ),
        detects=(
            "How much of the field is drawn on top of itself. Uniform random placement puts "
            "dots on each other constantly, so the number you can actually see is below the "
            "number you asked for."
        ),
        reading=(
            "Everything left of the dot-size line is an overlap. The canonical methods have a "
            "long left tail; our variants have a hard wall at the minimum separation and "
            "nothing to its left."
        ),
        validation=(
            "For n uniform points in a disc of radius R, the expected number of pairs closer "
            "than s is C(n,2)(s/R)². Measured counts match that prediction to within 2% "
            "across dot counts, dot sizes and field sizes."
        ),
    ),
    DiagnosticDoc(
        key="occlusion",
        title="Occlusion over time",
        measures=(
            "Per frame: the count of overlapping pairs, and the count of dots whose nearest "
            "neighbour is within half a dot diameter — close enough to be substantially "
            "hidden rather than merely touching."
        ),
        detects=(
            "Whether the effective dot count is stable or fluctuates with chance clustering, "
            "and whether occlusion arrives in bursts."
        ),
        reading=(
            "A flat line at zero means no dot is ever hidden. A noisy trace means the number "
            "of visible dots — and therefore the field's apparent contrast — is wandering "
            "frame to frame."
        ),
    ),
    DiagnosticDoc(
        key="step_size",
        title="Step size",
        measures=(
            "Each dot's displacement between consecutive frames, excluding dots the engine "
            "recorded as teleported. The expected coherent step is speed / fps, multiplied "
            "by the sequence count for the interleaved methods, since a dot there only "
            "updates every nth frame."
        ),
        detects=(
            "Which noise rule is actually running. This is the fastest way to confirm the "
            "stimulus is the algorithm you think it is."
        ),
        reading=(
            "Speed-matched noise (Brownian, random direction) puts every dot on the line. "
            "White noise puts only the signal dots there, because its noise dots teleport "
            "rather than move. Movshon–Newsome puts its updating third at three frames' "
            "worth of travel."
        ),
        validation=(
            "Teleports are read from the engine's own record of which dots it respawned and "
            "why, not inferred from how far a dot moved. Inferring it would misfile every "
            "respawn that happened to land near its old position — 1.7% of them in a small "
            "aperture. With the exact record, 100% of measured steps land on the expected "
            "value for all eight methods."
        ),
    ),
    DiagnosticDoc(
        key="direction",
        title="Step direction",
        measures=(
            "The polar angle of every non-teleport displacement, pooled across frames into a "
            "36-bin circular histogram. Angles follow the stimulus convention: 0° is "
            "rightward, 90° is up."
        ),
        detects=(
            "The shape of the noise, and whether the signal is going where you asked. Each "
            "noise rule leaves a different pedestal under the signal spike."
        ),
        reading=(
            "A spike at the signal direction sitting on a pedestal of noise. The pedestal is "
            "flat for Brownian, lumpy for random direction (each dot keeps one heading for "
            "life), and nearly absent for white noise."
        ),
        validation=(
            "At coherence 1 every measured angle equals the requested direction exactly; at "
            "coherence 0 the mean resultant vector length is under 0.01, i.e. uniform."
        ),
    ),
    DiagnosticDoc(
        key="density",
        title="Radial density",
        measures=(
            "Dots per unit area against distance from the field centre, binned into 20 "
            "equal-area rings so every bin holds the same expected count. The dashed line "
            "weights each dot by its opacity, which is what the Gaussian envelope changes."
        ),
        detects=(
            "Placement-sampler bugs. The classic one is sampling radius as r = R·u instead of "
            "r = R·√u, which crowds dots into the centre and is nearly invisible by eye."
        ),
        reading=(
            "The solid line should be flat. A centre-heavy slope means the sampler is wrong. "
            "The dashed line falls off for our variants and tracks the solid line for the "
            "canonical four, which is the envelope made visible."
        ),
        validation=(
            "The sampler alone is flat to the Poisson noise floor. The engine profile is flat "
            "across the inner rings and dips about 6–15% in the outermost one — that is real, "
            "not an artefact: dots crossing the rim respawn uniformly across the whole "
            "aperture, so the edge annulus is continuously drained. Wraparound would not do "
            "this; uniform respawn does."
        ),
    ),
    DiagnosticDoc(
        key="relocations",
        title="Relocations per frame",
        measures=(
            "The number of dots the engine teleported this frame rather than moved, taken "
            "from its own record. Covers all four causes: dot life expiry, aperture exit, "
            "white noise's per-frame relocation, and replanting for crowding."
        ),
        detects=(
            "How much of the field is jumping rather than moving, and whether dot deaths are "
            "spread evenly or arriving in synchronised cohorts."
        ),
        reading=(
            "White noise relocates roughly its noise-dot count every frame; Brownian "
            "relocates only what ages out or exits, about n / dot_life. A spiky trace means "
            "the staggered initial lifetimes are not doing their job."
        ),
    ),
    DiagnosticDoc(
        key="coherence",
        title="Coherence delivery",
        measures=(
            "Per frame: the fraction of dots flagged as carrying signal, the fraction of "
            "those that actually made a clean coherent displacement, and the fraction "
            "displaced by the minimum-separation constraint. A dot that teleported delivered "
            "no coherent motion that frame, whatever it was flagged as."
        ),
        detects=(
            "The gap between the coherence you asked for and the coherent motion the display "
            "actually delivered — and, crucially, which mechanism ate the difference."
        ),
        reading=(
            "The gap between the first two traces is normal: dots that aged out or left the "
            "aperture contribute no coherent motion, and the panel subtitle reports how much "
            "of the shortfall is that. The replanting trace should sit at zero; if it does "
            "not, the field is too crowded for the requested separation."
        ),
        validation=(
            "Separating the causes needs the engine to record why each dot moved, because a "
            "naive version of this metric reads about 0.91 for plain Brownian purely from "
            "dot ageing, which would read as a fault where there is none. Against a resolver "
            "that replants every dot in conflict the replanting trace reads 0.71, so the "
            "zero our variants report is a real zero and not a blind metric."
        ),
    ),
    DiagnosticDoc(
        key="luminance",
        title="Frame luminance",
        measures=(
            "The mean pixel value of each frame, rendered with the same renderer the exporter "
            "uses, compared against the background luminance."
        ),
        detects=(
            "A luminance flicker riding along with the motion, which would give an observer a "
            "cue that has nothing to do with direction."
        ),
        reading=(
            "With balanced dot luminance this sits flat on the background line: mean "
            "luminance carries no information, so only motion does. With uniform dots it sits "
            "above the background, which is expected — just be aware the dots add a luminance "
            "signal that co-varies with dot count."
        ),
    ),
)


def guide_payload() -> list[dict[str, str]]:
    """JSON-serialisable diagnostic guide for the webapp."""
    return [d.to_dict() for d in DIAGNOSTIC_GUIDE]
