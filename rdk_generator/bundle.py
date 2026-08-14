"""One archive holding everything about a single stimulus.

Kept apart from `export` and `diagnostics` because it draws on both, and importing it
here rather than there avoids a cycle.

The point of this module is that every artifact describes the *same* stimulus. That is not
automatic: a `None` seed means "different every time", so building the video, the frames
and the diagnostics from the same parameters would otherwise produce three unrelated
stimuli and a set of diagnostic plots that measured none of them.
"""

from __future__ import annotations

import io
import json
import zipfile

from .diagnostics import (
    PANELS,
    compute_diagnostics,
    figure_bytes,
    panel_bytes,
    per_frame_csv,
    radial_csv,
)
from .export import frames_zip_bytes, params_json, resolve_seed, video_bytes
from .methods import get_method
from .params import RDKParams, RenderParams

# Full-resolution figures plus a video plus every frame adds up. 200 dpi keeps the plots
# comfortably print-quality while leaving room for the rest.
BUNDLE_DPI = 200


def _readme(rdk: RDKParams, render: RenderParams, diag) -> str:
    spec = get_method(rdk.method)
    window = (
        f"Diagnostics measured all {diag.n_frames} frames."
        if not diag.truncated
        else (
            f"Diagnostics measured the first {diag.n_frames} of {diag.total_frames} frames; "
            "the video and frames cover the whole clip."
        )
    )
    return (
        "RDK stimulus bundle\n"
        "===================\n\n"
        f"{spec.label} ({spec.id})\n"
        f"{spec.tagline}\n\n"
        f"{rdk.n_dots} dots, coherence {rdk.coherence:g}, direction {rdk.direction_deg:g} deg, "
        f"{render.duration_s:g}s at {render.fps} fps, seed {render.seed}.\n\n"
        "Every file here describes the same stimulus: the seed above was fixed before\n"
        "anything was generated, so the diagnostics measure exactly the clip in\n"
        "stimulus.mp4. Re-running with params.json reproduces all of it.\n\n"
        f"{window}\n\n"
        "params.json        every parameter, including the method\n"
        "stimulus.mp4       the clip\n"
        "frames/            the same clip as per-frame PNGs\n"
        "diagnostics/       what the stimulus actually did -- see docs/diagnostics.md\n"
        "  diagnostics.png    combined report\n"
        "  diagnostics.pdf    the same, as vector\n"
        "  panels/            each panel separately, PNG and vector PDF\n"
        "  data/              the measured series as CSV, and the summary numbers\n"
    )


def everything_zip(
    rdk: RDKParams,
    render: RenderParams,
    *,
    dpi: int = BUNDLE_DPI,
    diagnostics_max_frames: int | None = None,
) -> bytes:
    """Bundle the video, the frames, the parameters and the diagnostics into one zip.

    The seed is pinned first, so all four describe one stimulus rather than four. By
    default the diagnostics measure the entire clip rather than the usual bounded window,
    since the whole point of the bundle is that the plots describe the video beside them.

    Args:
      rdk: Motion and geometry parameters.
      render: Timing and luminance parameters. A `None` seed is resolved and recorded.
      dpi: Raster resolution for the figures.
      diagnostics_max_frames: Cap on frames measured; `None` measures the whole clip.

    Returns:
      The zip archive as bytes.
    """
    render = resolve_seed(render)
    diag = compute_diagnostics(rdk, render, max_frames=diagnostics_max_frames)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("params.json", params_json(rdk, render))
        z.writestr("stimulus.mp4", video_bytes(rdk, render))

        # Unpack the frames archive so the bundle has one flat layout rather than a
        # zip inside a zip.
        with zipfile.ZipFile(io.BytesIO(frames_zip_bytes(rdk, render))) as frames:
            for name in frames.namelist():
                if name.endswith(".png"):
                    z.writestr(f"frames/{name}", frames.read(name))

        z.writestr("diagnostics/diagnostics.png", figure_bytes(diag, fmt="png", dpi=dpi))
        z.writestr("diagnostics/diagnostics.pdf", figure_bytes(diag, fmt="pdf"))
        for index, (key, _title, _draw, _polar) in enumerate(PANELS, start=1):
            z.writestr(
                f"diagnostics/panels/{index:02d}_{key}.png",
                panel_bytes(diag, key, fmt="png", dpi=dpi),
            )
            z.writestr(
                f"diagnostics/panels/{index:02d}_{key}.pdf", panel_bytes(diag, key, fmt="pdf")
            )

        z.writestr("diagnostics/data/per_frame.csv", per_frame_csv(diag))
        z.writestr("diagnostics/data/radial_density.csv", radial_csv(diag))
        z.writestr("diagnostics/data/summary.json", json.dumps(diag.summary(), indent=2))

        z.writestr("README.txt", _readme(rdk, render, diag))
    return archive.getvalue()
