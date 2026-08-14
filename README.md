# rdk-generator
Random Dot Kinematogram (RDK) Display Generator

This repo provides:

- **Four canonical RDK algorithms** from the vision-science literature — Movshon–Newsome,
  Brownian (random walk), white noise (random position), and random direction — plus a
  **Gaussian non-overlapping** variant of our own on each of the four bases, giving eight
  methods in total.
- Optional **luminance balancing** (dark/bright dots mixed so the field's mean luminance
  equals the background), available to every method.
- **Diagnostics** that measure what a stimulus actually did rather than what it was asked
  for — dot overlap and effective dot count, the step-size and direction signatures of each
  noise rule, radial density, coherence delivery, and luminance flicker.
  See [docs/diagnostics.md](docs/diagnostics.md).
- A small **Flask webapp** with a **live preview**: pick a method from the menu bar, and
  the stimulus re-simulates as you edit its parameters — no button to press. Each method
  shows how it is generated and what to cite. Export writes `.mp4`, a per-frame `.zip`, and
  the diagnostics `.png`.

See [docs/methods.md](docs/methods.md) for what each algorithm does and how they differ.
Coherence is **not** interchangeable between them.

## Install

From the repo root:

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -e .`

Optional PsychoPy support:

- `pip install -e '.[psychopy]'`

## Run the webapp

- `flask --app rdk_generator.webapp.app run`

Then open `http://127.0.0.1:5000/`.

The UI generates:

- An `.mp4` preview you can play in-browser or download
- A `.zip` containing per-frame images

## Use as a library (headless rendering)

```python
from rdk_generator import RDKParams, RenderParams, list_methods
from rdk_generator.export import write_frames_zip, write_mp4

# Four canonical methods -- "movshon_newsome", "brownian", "white_noise",
# "random_direction" -- and our variant on each, e.g. "gaussian_nonoverlap"
# (the default, MN-based) or "gaussian_nonoverlap_white_noise".
print([m.id for m in list_methods()])

rdk = RDKParams(method="brownian", n_dots=300, coherence=0.5, direction_deg=90, field_diam_px=300)
render = RenderParams(
    width_px=512, height_px=512, fps=60, duration_s=1.0, background_lum=0.5, dot_contrast=1.0
)

write_mp4("out/stimulus.mp4", rdk=rdk, render=render)
write_frames_zip("out/frames.zip", rdk=rdk, render=render)
```

Parameters a method does not use are ignored, and anything left as `None` falls back to
that method's canonical default — so `RDKParams(method="movshon_newsome")` gets no dot
lifetime, as the original algorithm specifies.

## PsychoPy usage

PsychoPy integration is optional and provided by [rdk_generator/psychopy_adapter.py](rdk_generator/psychopy_adapter.py):

- `rdk_generator.psychopy_adapter.RDKStim` presents any registered method in realtime. It
  drives the same headless engine used for export, so display and export cannot diverge.

## Module guide

See [docs/modules.md](docs/modules.md).
