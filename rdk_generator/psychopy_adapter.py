"""Optional PsychoPy integration.

This module is designed to be importable even when PsychoPy is not installed.
Callers should handle `ImportError` if they need PsychoPy-only functionality.

The stimulus math is *not* duplicated here. `RDKStim` drives one of the headless engines
from `rdk_generator.core` (selected by `RDKParams.method`) and only handles presentation,
so realtime display and offline export cannot drift apart.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .export import build_engine
from .params import RDKParams, RenderParams


def _require_psychopy():
    try:
        from psychopy import visual  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PsychoPy is required for rdk_generator.psychopy_adapter; "
            "install extras: pip install 'rdk-generator[psychopy]'"
        ) from e


class RDKStim:  # pragma: no cover (requires PsychoPy runtime)
    """Realtime PsychoPy presentation of any registered RDK method.

    Geometry is in whatever units the window uses, but the engines work in pixels, so
    create the window with `units="pix"` unless you have converted the parameters
    yourself.

    Color model:
      - Engine luminances are in [0, 1]; PsychoPy's `rgb` space is [-1, 1], so they are
        mapped as `rgb = 2 * lum - 1`.
      - With `RenderParams.luminance_mode="balanced"` the dots are a dark/bright mix whose
        mean equals the background. That balancing is done in linear RGB space; exact
        photometric luminance still depends on your monitor calibration and gamma.

    Args:
      win: An open `psychopy.visual.Window`.
      rdk: Motion/geometry parameters, including which method to use.
      render: Luminance and timing parameters. `fps` should match the display; pass
        `use_window_fps=True` to take it from the window instead.
      field_pos: Field centre offset, in window units.
      use_window_fps: Read the refresh rate from the window and override `render.fps`.
    """

    def __init__(
        self,
        *,
        win,
        rdk: RDKParams | None = None,
        render: RenderParams | None = None,
        field_pos: tuple[float, float] = (0.0, 0.0),
        use_window_fps: bool = True,
    ) -> None:
        _require_psychopy()
        from psychopy import visual  # type: ignore[import-not-found]

        self.win = win
        self.rdk = rdk if rdk is not None else RDKParams()
        render = render if render is not None else RenderParams()

        if use_window_fps:
            period = getattr(win, "monitorFramePeriod", None)
            measured = round(1.0 / period) if period else None
            if measured:
                render = replace(render, fps=int(measured))
        self.render = render

        self.field_pos = np.asarray(field_pos, dtype=float)
        self.engine = build_engine(self.rdk, self.render)

        rgb = np.asarray(self.engine.dot_lum, dtype=float) * 2.0 - 1.0
        self.colors = np.column_stack([rgb, rgb, rgb])

        self.stim = visual.ElementArrayStim(
            self.win,
            nElements=self.engine.n,
            sizes=float(self.rdk.dot_size_px),
            xys=self._offset(self.engine.xys),
            colors=self.colors,
            colorSpace="rgb",
            elementTex=None,
            elementMask="circle",
            opacities=self.engine.compute_opacity(),
            interpolate=False,
            autoLog=False,
        )

    def _offset(self, xys: np.ndarray) -> np.ndarray:
        return np.asarray(xys, dtype=float) + self.field_pos[None, :]

    def draw(self) -> None:
        """Draw the current frame, then advance the simulation by one frame."""
        self.stim.xys = self._offset(self.engine.xys)
        self.stim.opacities = self.engine.compute_opacity()
        self.stim.draw()
        self.engine.step()

    def reset(self) -> None:
        """Rebuild the engine, restarting the stimulus from its initial state."""
        self.engine = build_engine(self.rdk, self.render)
