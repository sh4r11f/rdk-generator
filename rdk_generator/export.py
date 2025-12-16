from __future__ import annotations

import io
import os
import zipfile
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .core import FrameRenderer, RDKEngine
from .params import RDKParams, RenderParams


def render_frames(rdk: RDKParams, render: RenderParams) -> list[np.ndarray]:
    """Render frames as a list of uint8 grayscale images."""
    n_frames = int(round(render.duration_s * render.fps))
    n_frames = max(1, n_frames)

    engine = RDKEngine(
        n_dots=rdk.n_dots,
        dot_life_frames=rdk.dot_life_frames,
        coherence=rdk.coherence,
        direction_deg=rdk.direction_deg,
        speed_px_per_s=rdk.speed_px_per_s,
        field_diam_px=rdk.field_diam_px,
        gauss_sigma_px=(rdk.gauss_sigma_px if rdk.gauss_sigma_px is not None else None),
        reassign_life=rdk.reassign_life,
        fps=render.fps,
        seed=render.seed,
        background_lum=render.background_lum,
        dot_contrast=render.dot_contrast,
        dot_low_lum=render.dot_low_lum,
        dot_high_lum=render.dot_high_lum,
    )

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
