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

Checked against Vercel's [function limits](https://vercel.com/docs/functions/limitations),
which are more generous for Python than for Node.

- **Response body: 4.5 MB.** This is the binding limit, since exports are streamed in the
  response. In practice there is headroom — dot fields are mostly flat background and
  compress well, so a 10s clip at 60fps and 1024px measures about 2.4 MB of frames or
  3.0 MB of video. Past that the app returns a clear message rather than an opaque
  platform error; `VERCEL_RESPONSE_LIMIT` in `webapp/app.py` sets the threshold, and the
  check only runs when the `VERCEL` environment variable is present.
- **Cold starts.** Importing NumPy plus matplotlib costs a couple of seconds on a cold
  instance. The first request after idling will feel slow; subsequent ones will not. This
  is the limitation you will actually notice.
- **Request duration: 300s on Hobby**, which `vercel.json` requests. Comfortably more than
  any export that fits under the response cap.
- **Bundle size: 500 MB uncompressed for Python runtimes.** The dependencies are roughly
  63 MB compressed, so there is a wide margin. `imageio-ffmpeg` is about half of it and is
  what makes `.mp4` export possible.
- **Memory: 2 GB / 1 vCPU on Hobby**, ample for these renders.
- **No persistence.** Every export is regenerated per request. That is fine here because
  the simulation is deterministic given a seed, and exports pin one, so any download is
  reproducible from its own `params.json`. Note that reproducible means the *stimulus*:
  frames come back pixel for pixel and measurements to the last digit on any machine, but
  the `.mp4` bytes depend on the ffmpeg build doing the encoding and so differ between
  the deployment and your laptop.

**Plan restriction.** Vercel's Hobby plan is free but is for personal, non-commercial
projects. A lab tool published under an institution may need a paid plan; check the
[Hobby plan terms](https://vercel.com/docs/plans/hobby) if that applies.
