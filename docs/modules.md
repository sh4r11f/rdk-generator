# Module Guide

This repo is split into two layers:

- **Headless generator** (`rdk_generator/*`): NumPy-based simulation + rendering/export.
  This layer can be tested on CI and used on servers.
- **PsychoPy integration** (`rdk_generator/psychopy_adapter.py`): realtime presentation.

## Modules

- `rdk_generator/params.py`
  - Dataclasses for RDK simulation parameters and rendering/export parameters.

- `rdk_generator/luminance.py`
  - Functions to compute black/white (low/high) dot luminances and assign per-dot colors such that the **mean** equals the background.

- `rdk_generator/core.py`
  - `RDKEngine`: advances dot positions each frame (signal/noise, dot life, wrap/respawn).
  - `FrameRenderer`: draws dots into grayscale images (uint8) with a Gaussian envelope.

- `rdk_generator/export.py`
  - `render_frames`: produce a list of frames.
  - `write_mp4`: export `.mp4` via imageio+ffmpeg.
  - `write_frames_zip`: export a `.zip` containing per-frame images.

- `rdk_generator/webapp/app.py`
  - Flask app with a single form that renders an `.mp4` preview and a `.zip` of frames.

- `rdk_generator/psychopy_adapter.py`
  - Optional PsychoPy class `BalancedGaussianRDK` with luminance-balanced dot colors.
