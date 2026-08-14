# Exports

Four downloads, all rendered on demand and streamed straight back. Nothing is stored on
the server between requests, which is what lets the app run on a serverless host — see
[deploy.md](deploy.md).

| Button | Endpoint | Contents |
|---|---|---|
| Download everything | `POST /export/bundle.zip` | the clip, every frame, the parameters and the diagnostic figures |
| Video | `POST /export/video.mp4` | the encoded clip |
| Frames | `POST /export/frames.zip` | per-frame PNGs plus `params.json` |
| Parameters | `POST /export/params.json` | every parameter, naming the method |
| Download figures (Diagnostics section) | `POST /api/diagnostics.zip` | the diagnostic figures alone — see [diagnostics.md](diagnostics.md) |

Each takes the same form fields as the live preview, so whatever is on screen is what you
get.

## Seeds, and why a download pins one

`RenderParams.seed = None` means "different every time". That is right for a preview and
wrong for a download, and the difference is not cosmetic.

The simulation draws from a fresh generator on every call, so building a video, a set of
frames and a diagnostics report from the same `None`-seeded parameters produces **three
unrelated stimuli**. A naive bundle would ship diagnostic plots that measured a clip which
appears nowhere in the archive.

Every export therefore calls `resolve_seed`, which draws one concrete seed if none was
given and records it in `params.json`. So:

- Everything inside one download describes one stimulus.
- Any download can be reproduced from its own `params.json`.
- Pinning a seed in the UI still works exactly as before and is untouched.

Downloads taken separately are *not* related to each other unless you pin a seed yourself.
Clicking Video and then Frames with the seed box blank gives you two different stimuli, by
design — each is internally consistent, and each records the seed it used.

## What "reproducible" covers

Feeding a `params.json` back in returns the same stimulus **exactly**: frames come back
pixel for pixel and every diagnostic number to the last digit, on any machine.

The `.mp4` bytes are a different matter. Encoding depends on the ffmpeg build doing it, so
a clip rendered by the deployment and the same clip rendered on your laptop are visually
identical but need not be byte-identical. If you are checksumming, checksum the frames.

```python
import json
from rdk_generator import RDKParams, RenderParams, render_frames

saved = json.load(open("rdk-params-brownian.json"))
rdk = RDKParams(**saved["rdk"])
render = RenderParams(**saved["render"])
frames = render_frames(rdk, render)  # identical to the ones in the download
```

`params.json` round-trips into the dataclasses without conversion; `RDKParams` coerces
`field_center_xy_px` back to a tuple, since JSON has no tuples.

## params.json

```json
{
  "method": {
    "id": "gaussian_nonoverlap",
    "label": "Gaussian non-overlapping (MN)",
    "tagline": "The Movshon-Newsome method with a soft aperture and evenly spaced dots."
  },
  "rdk":    { "method": "gaussian_nonoverlap", "n_dots": 300, ... },
  "render": { "width_px": 512, "fps": 60, "seed": 1469275254, ... }
}
```

The `method` block names the algorithm in full so the file makes sense on its own; `rdk`
and `render` are the two dataclasses verbatim so it can be fed straight back in.

## The combined bundle

```
params.json                       every parameter, including the method
stimulus.mp4                      the clip
frames/frame_00000.png            the same clip as per-frame PNGs
diagnostics/diagnostics.png       combined eight-panel report
diagnostics/diagnostics.pdf       the same, as vector
diagnostics/panels/01_spacing.*   each panel separately, PNG and vector PDF
diagnostics/data/per_frame.csv    the series behind the time plots
diagnostics/data/radial_density.csv
diagnostics/data/summary.json     the headline numbers
README.txt                        what is in the bundle, and the seed it used
```

Two things differ from the figures-only download. The seed is pinned before anything is
generated, so the plots measure exactly the clip beside them. And the diagnostics measure
the **whole clip** rather than the usual bounded window, since the point of the bundle is
that the report describes the video it ships with.

Sizes are modest because dot fields are mostly flat background and compress well: a default
one-second 512 px bundle is about 1.4 MB, and five seconds is about 2.9 MB in under six
seconds of work.

## From the library

```python
from rdk_generator import RDKParams, RenderParams, video_bytes, frames_zip_bytes
from rdk_generator.export import params_json, resolve_seed, write_frames_zip, write_mp4
from rdk_generator.bundle import everything_zip

rdk = RDKParams(method="gaussian_nonoverlap", coherence=0.35, direction_deg=90)
render = resolve_seed(RenderParams(duration_s=2.0))  # pin once, reuse everywhere

open("bundle.zip", "wb").write(everything_zip(rdk, render))

# ...or the pieces
open("clip.mp4", "wb").write(video_bytes(rdk, render))
open("frames.zip", "wb").write(frames_zip_bytes(rdk, render))
open("params.json", "w").write(params_json(rdk, render))

# The path-writing helpers are the same code, and write a .json sidecar next to the mp4.
write_mp4("out/stimulus.mp4", rdk=rdk, render=render)
write_frames_zip("out/frames.zip", rdk=rdk, render=render)
```

Call `resolve_seed` yourself if you want several artifacts to describe one stimulus;
`everything_zip` does it internally.
