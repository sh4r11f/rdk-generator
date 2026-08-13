# Module Guide

This repo is split into two layers:

- **Headless generator** (`rdk_generator/*`): NumPy-based simulation + rendering/export.
  This layer can be tested on CI and used on servers.
- **PsychoPy integration** (`rdk_generator/psychopy_adapter.py`): realtime presentation.

For what the algorithms actually are and where they come from, see [methods.md](methods.md).

## Modules

- `rdk_generator/params.py`
  - `RDKParams` (motion/geometry, including which `method` to use) and `RenderParams`
    (frame size, timing, luminance).

- `rdk_generator/methods.py`
  - The single registry of algorithms. Each `MethodSpec` names the engine that implements
    it, the parameters it accepts, prose describing how its frames are generated, and its
    citations. `export` and the webapp both read from here, so adding a method means adding
    a `MethodSpec` and nothing else.

- `rdk_generator/sampling.py`
  - `rand_points_in_circle`: uniform over the *area* of a disc.
  - `rand_unit_vectors`: uniformly distributed directions.
  - `poisson_points_in_circle`: rejection-sampled placement with a minimum separation,
    relaxing the constraint if the requested density is unachievable.

- `rdk_generator/luminance.py`
  - `uniform_dot_luminance`: the single shade used in the classic arrangement.
  - `resolve_dot_luminances` / `assign_binary_luminances` / `white_fraction_for_mean`: the
    two-level scheme whose **mean** equals the background, used by `luminance_mode="balanced"`.

- `rdk_generator/core.py`
  - `BaseRDKEngine`: shared scaffolding — placement, dot lifetime, signal-subset selection,
    aperture exits, the Gaussian envelope, and luminance assignment.
  - `BrownianRDKEngine`, `WhiteNoiseRDKEngine`, `RandomDirectionRDKEngine`,
    `MovshonNewsomeRDKEngine`: the four canonical algorithms. Each supplies its noise rule;
    `MovshonNewsomeRDKEngine` also overrides `step` for its interleaved sequences.
  - `GaussianNonOverlapRDKEngine`: our variant — `BrownianRDKEngine` plus a soft aperture
    and a minimum dot separation.
  - `FrameRenderer`: draws dots into grayscale `uint8` images.

- `rdk_generator/export.py`
  - `build_engine`: constructs the engine for `RDKParams.method`, passing only the
    parameters that method accepts and filling in its canonical defaults.
  - `render_frames` / `write_mp4` / `write_frames_zip`.

- `rdk_generator/webapp/app.py`
  - Flask app. `parse_form` coerces, clamps, and defaults submitted values against the
    registry; `GET /api/methods` serves the registry as JSON; the page lets you pick a
    method, edit only its parameters, and read its description and references.

- `rdk_generator/psychopy_adapter.py`
  - Optional `RDKStim`, which drives a headless engine and only handles presentation, so
    realtime display and offline export cannot drift apart.
