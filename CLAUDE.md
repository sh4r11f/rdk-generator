# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'` (plain `pip install -e .` skips pytest-cov and ruff)
- Test: `pytest` (pyproject sets `testpaths=tests`, `-q`)
- Lint/format: `ruff check --fix . && ruff format .` (line-length 100 — configured in pyproject.toml)
- Webapp: `flask --app rdk_generator.webapp.app run --port 5050` → http://127.0.0.1:5050/ (port 5000 is taken by macOS AirPlay)
- Deploy: `vercel` (config in `vercel.json` + `api/index.py`; `requirements.txt` mirrors `[project.dependencies]` and must be kept in step)

## Architecture

Two strictly separated layers (see docs/modules.md):

- Headless generator (`params`, `sampling`, `luminance`, `core`, `export`): pure NumPy + PIL + imageio, no display. All stimulus math lives here. Never import PsychoPy from this layer.
- PsychoPy integration (`psychopy_adapter.py`): optional realtime presentation. Must stay import-safe when PsychoPy is not installed; its PsychoPy-only paths are marked `# pragma: no cover`.

The Flask webapp (`rdk_generator/webapp/`) is a thin UI over `export`. It is deployable to serverless hosts (see docs/deploy.md), which constrains it: **nothing may be written to disk and nothing may persist between requests**. Exports are rendered on demand and streamed back from `POST /export/*`; never reintroduce a write-then-download-later flow or a module-import side effect that touches the filesystem. `tests/test_webapp.py::test_nothing_is_written_to_disk_by_any_request` guards this. The dev fallback secret key is accepted by design.

## Conventions

- Public constructors/functions take keyword-only arguments (bare `*`); dataclasses are `frozen=True, slots=True`; Google-style docstrings; `from __future__ import annotations` in every module.
- `direction_deg` uses the PsychoPy convention: 0° = right, 90° = up. Dot positions are stored relative to field center; `FrameRenderer.render` flips Y when converting to pixel coordinates.
- `methods.py` is the single registry of algorithms (see docs/methods.md). A `MethodSpec` names its engine, the params it accepts, its prose description, and its citations; `export.build_engine` and the webapp both read from it, so adding a method means adding a `MethodSpec` and nothing else. Don't hardcode method lists elsewhere.
- `RDKParams.dot_life_frames=None` means "use the method's canonical default" (0 for `movshon_newsome`, 12 otherwise). `gauss_sigma_px`/`min_sep_px` of `None` or 0 likewise mean "derive it" (envelope σ = field_diam/4; separation = 1.1 × dot_size_px).
- Our two modifications (Gaussian envelope + minimum separation) live in `GaussianNonOverlapMixin`, which only overrides base hooks so it composes onto any engine: there is one variant per canonical base (`gaussian_nonoverlap` = MN and the default, plus `_brownian`, `_white_noise`, `_random_direction`). Canonical methods never expose those two params. Luminance balancing is *not* part of any method: it's the render-level `luminance_mode="balanced"`, available to all of them.
- Noise rules that teleport dots must go through `_respawn(mask, reason=...)`, never place positions directly — that is what records the teleport for diagnostics and what applies the non-overlap constraint. `WhiteNoiseRDKEngine._move_noise` is the example.
- Diagnostic panels are drawn by one function each, registered in `PANELS` with keys matching `DIAGNOSTIC_GUIDE`, so the combined report, the individually exported figures, and the UI's guide all stay in sync. A test asserts the key sets match.
- `diagnostics.py` measures a stimulus (overlap, step size/direction, radial density, coherence delivery, luminance) and is the only module that uses matplotlib — imported lazily inside `figure_png`. Keep `compute_diagnostics` pure NumPy. Engines record *why* a dot was teleported (`RESPAWN_REASONS`) only when `engine.record_respawns` is set, which is what lets diagnostics separate replanting loss from dot-life and aperture-exit loss; don't conflate them.
- The webapp previews via `POST /api/preview` (packed float32 dot positions animated on a canvas), not by rendering video. Keep stimulus math in Python — never reimplement motion in JS. Only `POST /generate` writes files.
- Frames are 8-bit single-channel grayscale arrays of shape `(height, width)`; `write_mp4` also writes a `<name>.json` params sidecar used by the webapp's `/meta/` route.

## Gotchas

- If webapp session key semantics change, bump `SESSION_SCHEMA_VERSION` in `webapp/app.py`.
- `tests/test_webapp.py` monkeypatches `write_mp4`/`write_frames_zip` on `rdk_generator.webapp.app` (the module that imported the names), not on `rdk_generator.export` — patching the export module has no effect.
- Importing `rdk_generator.webapp.app` creates `instance/outputs/` on disk (module-level `app = create_app()` runs `mkdir`).

## Git

- Feature branches + PRs to `main`; don't commit directly to `main`.
