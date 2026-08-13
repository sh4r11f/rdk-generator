# rdk-generator
Random Dot Kinematogram (RDK) Display Generator

This repo provides:

- **Four canonical RDK algorithms** from the vision-science literature — Movshon–Newsome,
  Brownian (random walk), white noise (random position), and random direction — plus a
  **Gaussian non-overlapping** variant of our own.
- Optional **luminance balancing** (dark/bright dots mixed so the field's mean luminance
  equals the background), available to every method.
- A small **Flask webapp** to pick a method, tweak only its parameters, read how it is
  generated and what to cite, and export previews.

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

# Any registered method: "movshon_newsome", "brownian", "white_noise",
# "random_direction", or "gaussian_nonoverlap" (the default).
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
