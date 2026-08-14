# Deploying

The app is a plain WSGI Flask app with no database and no writable state, so most Python
hosts will run it as-is. `api/index.py` exposes the WSGI callable, and `vercel.json`
routes every path to it.

## What had to change to make this deployable

The original export flow rendered an `.mp4` and a `.zip` into `instance/outputs/`, then
served them on a *later* request via a job id in the session. That works on one long-lived
process and nowhere else:

- **Nothing may be written at import.** `create_app` used to `mkdir` its output directory
  as a side effect of module import, which crashes on a read-only serverless filesystem.
- **Two requests may not land on the same machine.** A file written while handling
  `POST /generate` need not exist when `GET /download/<id>` arrives.

Exports are now rendered on demand and streamed straight back:

| Endpoint | Returns |
|---|---|
| `POST /export/video.mp4` | the encoded clip |
| `POST /export/frames.zip` | per-frame PNGs plus `params.json` |
| `POST /api/diagnostics.zip` | the figure bundle |
| `POST /api/preview` | packed dot positions for the canvas |
| `POST /api/diagnostics` | the report figure, base64 |

Nothing touches disk, and a test asserts that. The session is still used, but only to
remember form values — that is a signed cookie, so it survives moving between instances.

## Vercel

```bash
vercel login      # interactive; only you can do this
vercel            # preview deployment
vercel --prod     # production
```

`requirements.txt` mirrors `[project.dependencies]` in `pyproject.toml` and must be kept in
step with it. `.vercelignore` keeps the virtualenv, tests and docs out of the bundle.

### Known limits on serverless

- **Cold starts.** Importing NumPy plus matplotlib costs a couple of seconds on a cold
  instance. The first request after idling will feel slow; subsequent ones will not.
- **Request duration.** `vercel.json` asks for 60s. The live preview is comfortably inside
  that, but a long clip at high resolution can exceed it — a 10s clip at 60fps and 1024px
  is 600 frames to simulate, render and encode. Export shorter clips, or render locally
  with the library for anything large.
- **Bundle size.** The dependencies come to roughly 63 MB compressed, against Vercel's
  250 MB unzipped limit. `imageio-ffmpeg` is 30 MB of that, and it is what makes `.mp4`
  export possible; dropping it would halve the bundle at the cost of video export.
- **No persistence.** Every export is regenerated per request. That is fine here because
  renders are deterministic given the seed, so a link is reproducible from `params.json`.

A long-lived container host (Fly.io, Render, Railway) avoids the cold start and the
duration cap, and would be the better home if you want to render long clips through the
web UI rather than the library.
