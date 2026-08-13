"""Top-level package for rdk-generator.

This package provides:
- A headless (NumPy-based) RDK generator that can render frames/videos.
- Four canonical RDK algorithms plus a Gaussian non-overlapping variant
  (see `rdk_generator.methods` and docs/methods.md).
- Optional PsychoPy integration for real-time experimental display.
- A small Flask webapp for interactive parameter tuning and export.

PsychoPy integration is optional and lives in `rdk_generator.psychopy_adapter`.
"""

from .export import build_engine, render_frames, simulate, write_frames_zip, write_mp4
from .methods import DEFAULT_METHOD, METHODS, MethodSpec, get_method, list_methods
from .params import RDKParams, RenderParams

__all__ = [
    "DEFAULT_METHOD",
    "METHODS",
    "MethodSpec",
    "RDKParams",
    "RenderParams",
    "build_engine",
    "get_method",
    "list_methods",
    "render_frames",
    "simulate",
    "write_frames_zip",
    "write_mp4",
]
