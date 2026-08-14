# rdk-generator

Random dot kinematogram (RDK) stimuli for vision science — four canonical algorithms from
the literature, a variant of our own on each, diagnostics that measure what a stimulus
actually did, and a web UI that previews it live.

**Live: [rdk-generator.vercel.app](https://rdk-generator.vercel.app)**

## What's here

**Eight methods.** The four canonical algorithms — Movshon–Newsome, Brownian (random walk),
white noise (random position) and random direction — plus a **Gaussian non-overlapping**
variant built on each of them. The variants add a soft aperture and evenly spaced dots
*without touching the motion*, so each one's coherence is directly comparable to its own
base. Coherence is **not** comparable across different bases: thresholds differ several-fold
between algorithms, so "50% coherence" only means something once you name the method.

**Diagnostics.** What a stimulus is and what its parameters asked for are different things.
Dots occlude each other, so fewer are visible than you specified. Coherence is a per-frame
draw in the Movshon–Newsome family. Dots that age out or leave the aperture deliver no
coherent motion that frame. An eight-panel report measures all of it, and each metric is
validated against a value derived independently of the code that computes it.

**A live web UI.** Pick a method from the menu bar and the stimulus re-simulates as you edit
its parameters — there is no button to press. Each method shows how it generates each frame
and what to cite.

**Exports.** The clip, its frames, its parameters, the diagnostic figures, or all of it in
one archive. Every download pins a seed and records it, so it is reproducible.

**Optional extras.** Luminance balancing (dark/bright dots mixed so mean field luminance
equals the background, leaving motion as the only cue) is available to every method. PsychoPy
integration presents any method in realtime from the same engine used for export.

## Documentation

| | |
|---|---|
| [methods.md](docs/methods.md) | The algorithms, what distinguishes them, what our variants add, and the references |
| [diagnostics.md](docs/diagnostics.md) | What each panel measures, the fault it catches, and how the measurement was validated |
| [exports.md](docs/exports.md) | Every download, and what reproducible does and does not cover |
| [deploy.md](docs/deploy.md) | Running it as a served app, and the real limits |
| [modules.md](docs/modules.md) | How the code is laid out |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'      # plain `pip install -e .` skips pytest and ruff
```

Optional PsychoPy support: `pip install -e '.[psychopy]'`

## Run the web UI

```bash
flask --app rdk_generator.webapp.app run --port 5050
```

Then open <http://127.0.0.1:5050/>. Port 5000 is worth avoiding on macOS, where AirPlay
Receiver usually holds it.

Put a value in the **Seed** box to make a stimulus reproducible; leave it blank for a fresh
one each time. **Measure stimulus** in the Diagnostics section reports what the current
parameters actually produced, and the **Diagnostics guide** view explains every panel.

## Use as a library

```python
from rdk_generator import RDKParams, RenderParams, list_methods, write_mp4

# "movshon_newsome", "brownian", "white_noise", "random_direction",
# and our variant on each: "gaussian_nonoverlap" (the default, MN-based),
# "gaussian_nonoverlap_brownian", "_white_noise", "_random_direction".
print([m.id for m in list_methods()])

rdk = RDKParams(method="brownian", n_dots=300, coherence=0.5, direction_deg=90)
render = RenderParams(width_px=512, height_px=512, fps=60, duration_s=1.0, seed=1)

write_mp4("out/stimulus.mp4", rdk=rdk, render=render)
```

Parameters a method does not use are ignored, and anything left as `None` falls back to that
method's canonical default — so `RDKParams(method="movshon_newsome")` gets no dot lifetime,
as the original algorithm specifies.

Measuring a stimulus, and getting the figures out:

```python
from rdk_generator.bundle import everything_zip
from rdk_generator.diagnostics import compute_diagnostics, write_figures_zip

print(compute_diagnostics(rdk, render).summary()["visible_dot_fraction"])
write_figures_zip("out/figures.zip", rdk=rdk, render=render)  # PNG + vector PDF + CSV
open("out/bundle.zip", "wb").write(everything_zip(rdk, render))  # everything at once
```

## PsychoPy

`rdk_generator.psychopy_adapter.RDKStim` presents any registered method in realtime. It
drives the same headless engine the exporters use, so what you display and what you export
cannot drift apart. The module stays importable without PsychoPy installed.

## Development

```bash
pytest                               # unit tests
ruff check --fix . && ruff format .  # lint and format
```

`.claude/skills/verify-rdk` runs all of that plus an end-to-end smoke render that confirms
the `.mp4`, the frames `.zip` and the params sidecar are actually produced and valid.

Feature branches and PRs to `main`. See [CLAUDE.md](CLAUDE.md) for the conventions the code
follows and the constraints that are deliberate.
