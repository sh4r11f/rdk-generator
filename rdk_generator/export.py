from __future__ import annotations

import io
import os
import zipfile
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .core import BaseRDKEngine, FrameRenderer
from .methods import get_method
from .params import RDKParams, RenderParams


def build_engine(rdk: RDKParams, render: RenderParams) -> BaseRDKEngine:
    """Construct the engine for `rdk.method`, passing only the params it accepts."""
    spec = get_method(rdk.method)

    kwargs: dict = {
        "n_dots": rdk.n_dots,
        "coherence": rdk.coherence,
        "direction_deg": rdk.direction_deg,
        "speed_px_per_s": rdk.speed_px_per_s,
        "field_diam_px": rdk.field_diam_px,
        "fps": render.fps,
        "seed": render.seed,
        "background_lum": render.background_lum,
        "dot_contrast": render.dot_contrast,
        "dot_low_lum": render.dot_low_lum,
        "dot_high_lum": render.dot_high_lum,
        "luminance_mode": render.luminance_mode,
    }

    accepted = set(spec.method_params)
    if "dot_life_frames" in accepted:
        life = rdk.dot_life_frames
        kwargs["dot_life_frames"] = int(
            spec.default_for("dot_life_frames") if life is None else life
        )
    if "signal_rule" in accepted:
        kwargs["signal_rule"] = rdk.signal_rule
    if "n_sequences" in accepted:
        kwargs["n_sequences"] = rdk.n_sequences
    if "gauss_sigma_px" in accepted:
        # 0 and None both mean "use the method's default width".
        kwargs["gauss_sigma_px"] = rdk.gauss_sigma_px or None
    if "min_sep_px" in accepted:
        kwargs["min_sep_px"] = rdk.min_sep_px or (1.1 * float(rdk.dot_size_px))

    return spec.engine(**kwargs)


def simulate(
    rdk: RDKParams, render: RenderParams, *, max_frames: int | None = None
) -> dict[str, object]:
    """Run the simulation and return per-frame dot state, without drawing any images.

    This is what the webapp's live preview uses: sending a few hundred kilobytes of dot
    positions and letting the browser draw them is far quicker than encoding a video, and
    it keeps every bit of stimulus math in Python rather than duplicating it in JavaScript.

    Args:
      rdk: Motion and geometry parameters.
      render: Timing and luminance parameters.
      max_frames: Optional cap, to bound the payload for interactive previews.

    Returns:
      A dict with float32 `xy` of shape (frames, dots, 2) in field-centred coordinates,
      `alpha` of shape (frames, dots), constant per-dot `dot_lum`, and the frame counts.
    """
    engine = build_engine(rdk, render)

    total_frames = max(1, int(round(render.duration_s * render.fps)))
    n_frames = total_frames
    if max_frames is not None:
        n_frames = min(n_frames, max(1, int(max_frames)))

    xy = np.empty((n_frames, engine.n, 2), dtype=np.float32)
    alpha = np.empty((n_frames, engine.n), dtype=np.float32)
    for i in range(n_frames):
        xy[i] = engine.xys
        alpha[i] = engine.compute_opacity()
        engine.step()

    return {
        "xy": xy,
        "alpha": alpha,
        "dot_lum": np.asarray(engine.dot_lum, dtype=np.float32),
        "n_frames": n_frames,
        "total_frames": total_frames,
        "n_dots": engine.n,
        "truncated": n_frames < total_frames,
    }


def render_frames(rdk: RDKParams, render: RenderParams) -> list[np.ndarray]:
    """Render frames as a list of uint8 grayscale images."""
    n_frames = int(round(render.duration_s * render.fps))
    n_frames = max(1, n_frames)

    engine = build_engine(rdk, render)

    renderer = FrameRenderer(
        width_px=render.width_px,
        height_px=render.height_px,
        background_lum=render.background_lum,
        field_center_xy_px=rdk.field_center_xy_px,
        dot_size_px=rdk.dot_size_px,
    )

    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        opacity = engine.compute_opacity()
        frames.append(renderer.render(engine.xys, engine.dot_lum, opacity))
        engine.step()
    return frames


def write_mp4(
    out_path: str | os.PathLike,
    *,
    rdk: RDKParams,
    render: RenderParams,
) -> Path:
    """Render and write an `.mp4` to `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = render_frames(rdk, render)
    # imageio expects frames as HxW or HxWx3; we use HxW and specify fps.
    with imageio.get_writer(out_path, fps=render.fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)

    # Store a small sidecar for reproducibility
    sidecar = out_path.with_suffix(".json")
    try:
        import json

        payload = {"rdk": asdict(rdk), "render": asdict(render)}
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out_path


def write_frames_zip(
    out_path: str | os.PathLike,
    *,
    rdk: RDKParams,
    render: RenderParams,
    image_format: str = "png",
) -> Path:
    """Render and write frames as a zip of images."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = render_frames(rdk, render)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame, mode="L")
            buf = io.BytesIO()
            img.save(buf, format=image_format.upper())
            z.writestr(f"frame_{i:05d}.{image_format}", buf.getvalue())

    return out_path
