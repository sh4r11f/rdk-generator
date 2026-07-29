# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'` (plain `pip install -e .` skips pytest-cov and ruff)
- Test: `pytest` (pyproject sets `testpaths=tests`, `-q`)
- Lint/format: `ruff check --fix . && ruff format .` (line-length 100 — configured in pyproject.toml)
- Webapp: `flask --app rdk_generator.webapp.app run` → http://127.0.0.1:5000/

## Architecture

Two strictly separated layers (see docs/modules.md):

- Headless generator (`params`, `sampling`, `luminance`, `core`, `export`): pure NumPy + PIL + imageio, no display. All stimulus math lives here. Never import PsychoPy from this layer.
- PsychoPy integration (`psychopy_adapter.py`): optional realtime presentation. Must stay import-safe when PsychoPy is not installed; its PsychoPy-only paths are marked `# pragma: no cover`.

The Flask webapp (`rdk_generator/webapp/`) is a thin localhost-only UI over `export`. Synchronous blocking render in `POST /generate`, the dev fallback secret key, and unbounded `instance/outputs/` growth are accepted by design — don't fix them unprompted.

## Conventions

- Public constructors/functions take keyword-only arguments (bare `*`); dataclasses are `frozen=True, slots=True`; Google-style docstrings; `from __future__ import annotations` in every module.
- `direction_deg` uses the PsychoPy convention: 0° = right, 90° = up. Dot positions are stored relative to field center; `FrameRenderer.render` flips Y when converting to pixel coordinates.
- `render_frames` selects `NonOverlappingRDKEngine` iff `min_sep_px is not None` (recommended ≈ 1.1 × dot_size_px), otherwise `RDKEngine`.
- Frames are 8-bit single-channel grayscale arrays of shape `(height, width)`; `write_mp4` also writes a `<name>.json` params sidecar used by the webapp's `/meta/` route.

## Gotchas

- If webapp session key semantics change, bump `SESSION_SCHEMA_VERSION` in `webapp/app.py`.
- `tests/test_webapp.py` monkeypatches `write_mp4`/`write_frames_zip` on `rdk_generator.webapp.app` (the module that imported the names), not on `rdk_generator.export` — patching the export module has no effect.
- Importing `rdk_generator.webapp.app` creates `instance/outputs/` on disk (module-level `app = create_app()` runs `mkdir`).

## Git

- Feature branches + PRs to `main`; don't commit directly to `main`.
