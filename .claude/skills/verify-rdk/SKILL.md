---
name: verify-rdk
description: Full verification of rdk-generator changes — lint/format check, test suite, and a headless end-to-end smoke render that confirms the .mp4, frames .zip, and .json params sidecar are produced and valid. Use after modifying the generator, exporters, or webapp, or before opening a PR.
---

Run all three stages from the repo root using the project venv (`./.venv/bin/...`). Report each stage's result; stop and fix on the first failure.

## 1. Lint + format check

```bash
./.venv/bin/ruff check . && ./.venv/bin/ruff format --check .
```

## 2. Test suite

```bash
./.venv/bin/pytest
```

## 3. Headless smoke render

Renders a tiny stimulus end-to-end (real ffmpeg encode, unlike the webapp tests which monkeypatch the exporters) and validates all three artifacts:

```bash
./.venv/bin/python - <<'EOF'
import json, tempfile, zipfile
from pathlib import Path
import imageio.v3 as iio
from rdk_generator import RDKParams, RenderParams, write_mp4, write_frames_zip

tmp = Path(tempfile.mkdtemp(prefix="rdk-verify-"))
rdk = RDKParams(n_dots=50, coherence=0.5, direction_deg=0, field_diam_px=64)
render = RenderParams(width_px=96, height_px=96, fps=30, duration_s=0.5,
                      background_lum=0.5, dot_contrast=1.0)
expected = max(1, round(render.fps * render.duration_s))

mp4 = Path(write_mp4(tmp / "smoke.mp4", rdk=rdk, render=render))
assert mp4.stat().st_size > 0, "mp4 is empty"
n_mp4 = sum(1 for _ in iio.imiter(mp4))
assert n_mp4 == expected, f"mp4 has {n_mp4} frames, expected {expected}"

sidecar = mp4.with_suffix(".json")
meta = json.loads(sidecar.read_text())
assert set(meta) == {"rdk", "render"}, f"sidecar keys: {set(meta)}"

zpath = Path(write_frames_zip(tmp / "smoke.zip", rdk=rdk, render=render))
with zipfile.ZipFile(zpath) as zf:
    pngs = [n for n in zf.namelist() if n.endswith(".png")]
assert len(pngs) == expected, f"zip has {len(pngs)} PNGs, expected {expected}"

print(f"smoke render OK: {n_mp4} mp4 frames, {len(pngs)} PNGs, sidecar valid ({tmp})")
EOF
```

If the sidecar assertion fails, remember `write_mp4` swallows sidecar write errors silently (`except Exception: pass` in `export.py`) — the root cause will not have surfaced as an exception.
