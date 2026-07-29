# rdk-generator
Random Dot Kinematogram (RDK) Display Generator

This repo provides:

- A **Gaussian-masked** RDK implementation (based on PsychoPy-style parameters).
- **Luminance-balanced** black/white dot assignment so the **mean** luminance inside the field matches the grey background.
- A small **Flask webapp** to tweak parameters and export previews.

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
from rdk_generator import RDKParams, RenderParams
from rdk_generator.export import write_mp4, write_frames_zip

rdk = RDKParams(n_dots=300, coherence=0.5, direction_deg=90, field_diam_px=300)
render = RenderParams(
    width_px=512, height_px=512, fps=60, duration_s=1.0, background_lum=0.5, dot_contrast=1.0
)

write_mp4("out/stimulus.mp4", rdk=rdk, render=render)
write_frames_zip("out/frames.zip", rdk=rdk, render=render)
```

## PsychoPy usage

PsychoPy integration is optional and provided by [rdk_generator/psychopy_adapter.py](rdk_generator/psychopy_adapter.py):

- `rdk_generator.psychopy_adapter.BalancedGaussianRDK`

## Module guide

See [docs/modules.md](docs/modules.md).
