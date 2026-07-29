"""Top-level package for rdk-generator.

This package provides:
- A headless (NumPy-based) RDK generator that can render frames/videos.
- Optional PsychoPy integration for real-time experimental display.
- A small Flask webapp for interactive parameter tuning and export.

PsychoPy integration is optional and lives in `rdk_generator.psychopy_adapter`.
"""

from .export import render_frames, write_frames_zip, write_mp4
from .params import RDKParams, RenderParams

__all__ = [
    "RDKParams",
    "RenderParams",
    "render_frames",
    "write_frames_zip",
    "write_mp4",
]
