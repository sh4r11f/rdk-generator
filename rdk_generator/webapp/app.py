from __future__ import annotations

import base64
import json
import os
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
)

from ..bundle import everything_zip
from ..diagnostics import (
    compute_diagnostics,
    figure_png,
    figures_zip,
    guide_payload,
)
from ..export import (
    frames_zip_bytes,
    params_json,
    resolve_seed,
    simulate,
    video_bytes,
)
from ..methods import DEFAULT_METHOD, FIELDS, GROUP_LABELS, GROUP_ORDER, get_method, methods_payload
from ..params import RDKParams, RenderParams

_RDK_FIELDS = {f.name for f in dataclass_fields(RDKParams)}
_RENDER_FIELDS = {f.name for f in dataclass_fields(RenderParams)}

# Parameters where a non-positive value means "derive a sensible default".
_AUTO_WHEN_ZERO = ("gauss_sigma_px", "min_sep_px")

# The live preview streams raw dot positions, so its cost scales with frames x dots.
# Long clips are truncated for the preview only; exports always use the full duration.
PREVIEW_MAX_FRAMES = 150

# Vercel caps a function response body at 4.5 MB. Exports are streamed in the response, so
# a long or high-resolution clip can exceed it. Enforced only when actually running there,
# where it turns an opaque platform 413 into an actionable message; a normal server has no
# such limit and is left alone.
VERCEL_RESPONSE_LIMIT = 4_400_000


def _pack(array) -> str:
    """Encode an array as little-endian float32 base64, for the browser to decode."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def _coerce(name: str, raw: str | None) -> Any:
    """Coerce one raw form value to its declared type, or None if blank/unparseable."""
    spec_field = FIELDS[name]
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None

    if spec_field.kind == "int":
        try:
            return int(round(float(raw)))
        except ValueError:
            return None
    if spec_field.kind == "float":
        try:
            return float(raw)
        except ValueError:
            return None
    if spec_field.kind == "choice":
        return raw if raw in {value for value, _ in spec_field.choices} else None
    return raw


def _clamp(name: str, value: Any) -> Any:
    spec_field = FIELDS[name]
    if spec_field.kind not in ("int", "float"):
        return value
    if spec_field.min is not None and value < spec_field.min:
        value = spec_field.min
    if spec_field.max is not None and value > spec_field.max:
        value = spec_field.max
    return int(value) if spec_field.kind == "int" else float(value)


def parse_form(form) -> tuple[RDKParams, RenderParams, dict[str, Any]]:
    """Build params from a submitted form, using the method's defaults for anything missing.

    Returns the two param objects plus a plain dict suitable for re-populating the UI.
    """
    spec = get_method(form.get("method"))

    values: dict[str, Any] = {}
    for name in spec.param_names:
        value = _coerce(name, form.get(name))
        if value is None:
            if name == "seed":
                values[name] = ""
                continue
            value = spec.default_for(name)
        values[name] = _clamp(name, value)

    sticky = dict(values)
    sticky["method"] = spec.id

    rdk_kwargs: dict[str, Any] = {"method": spec.id}
    render_kwargs: dict[str, Any] = {}
    for name, value in values.items():
        if name == "seed":
            render_kwargs["seed"] = None if value == "" else int(value)
            continue
        if name in _AUTO_WHEN_ZERO and (value is None or float(value) <= 0):
            value = None
        if name in _RDK_FIELDS:
            rdk_kwargs[name] = value
        elif name in _RENDER_FIELDS:
            render_kwargs[name] = value

    return RDKParams(**rdk_kwargs), RenderParams(**render_kwargs), sticky


def create_app(*, instance_path: str | None = None) -> Flask:
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(Path(__file__).with_name("templates")),
        instance_path=instance_path,
    )

    # Needed for session-based "remember last params" behavior.
    # For real deployments set RDK_SECRET_KEY to a strong random value.
    app.config["SECRET_KEY"] = os.environ.get("RDK_SECRET_KEY", "dev-secret-key-change-me")
    # Bump this if session semantics change.
    app.config["SESSION_SCHEMA_VERSION"] = "v3"

    # Deliberately no output directory: exports are streamed straight back in the
    # response. Writing them to disk and serving them on a later request needs a
    # writable filesystem shared across requests, which a serverless host has not got.

    @app.before_request
    def _reset_if_stale() -> None:
        # Runs for every route, so a handler that writes to the session cannot forget to
        # stamp the schema and have its values wiped by the next page load.
        schema = app.config.get("SESSION_SCHEMA_VERSION")
        if session.get("_schema") != schema:
            session["_schema"] = schema
            session.pop("saved_params", None)

    def _download(blob: bytes, mimetype: str, filename: str) -> Response:
        if os.environ.get("VERCEL") and len(blob) > VERCEL_RESPONSE_LIMIT:
            megabytes = len(blob) / 1e6
            return (
                jsonify(
                    {
                        "error": (
                            f"This export is {megabytes:.1f} MB, over the 4.5 MB response "
                            "limit this deployment can return. Shorten the clip, lower the "
                            "frame rate, or reduce the width and height — or render it "
                            "locally with the library, which has no such limit."
                        )
                    }
                ),
                413,
            )
        return Response(
            blob,
            mimetype=mimetype,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _remember(rdk: RDKParams, sticky: dict) -> None:
        saved = dict(session.get("saved_params") or {})
        saved[rdk.method] = sticky
        session["saved_params"] = saved
        session["last_method"] = rdk.method

    @app.get("/")
    def index() -> str:
        saved = session.get("saved_params") or {}
        selected = session.get("last_method") or DEFAULT_METHOD
        if selected not in {m["id"] for m in methods_payload()}:
            selected = DEFAULT_METHOD

        return render_template(
            "index.html",
            methods_json=json.dumps(methods_payload()),
            guide_json=json.dumps(guide_payload()),
            saved_json=json.dumps(saved),
            selected=selected,
            group_order=list(GROUP_ORDER),
            group_labels=GROUP_LABELS,
        )

    @app.get("/api/methods")
    def api_methods() -> Response:
        return jsonify(methods_payload())

    @app.get("/api/diagnostic-guide")
    def api_diagnostic_guide() -> Response:
        return jsonify(guide_payload())

    @app.post("/api/preview")
    def api_preview() -> Response:
        """Simulate the current parameters and hand the browser raw dot positions.

        Deliberately does not touch disk or the session: this fires on every parameter
        change, and only the explicit export writes files.
        """
        rdk, render, _ = parse_form(request.form)
        sim = simulate(rdk, render, max_frames=PREVIEW_MAX_FRAMES)
        spec = get_method(rdk.method)

        return jsonify(
            {
                "method": spec.id,
                "label": spec.label,
                "n_frames": sim["n_frames"],
                "total_frames": sim["total_frames"],
                "truncated": sim["truncated"],
                "n_dots": sim["n_dots"],
                "fps": render.fps,
                "width": render.width_px,
                "height": render.height_px,
                "background": render.background_lum,
                "dot_size": rdk.dot_size_px,
                "field_diam": rdk.field_diam_px,
                "center": list(rdk.field_center_xy_px),
                "xy": _pack(sim["xy"]),
                "alpha": _pack(sim["alpha"]),
                "lum": _pack(sim["dot_lum"]),
            }
        )

    @app.post("/api/diagnostics")
    def api_diagnostics() -> Response:
        """Measure the current parameters and return the diagnostic panel.

        Costs roughly a second, so the page asks for this on demand rather than on
        every parameter change.
        """
        rdk, render, _ = parse_form(request.form)
        diag = compute_diagnostics(rdk, render)
        return jsonify(
            {
                "summary": diag.summary(),
                "notes": list(diag.notes),
                "png": base64.b64encode(figure_png(diag)).decode("ascii"),
            }
        )

    @app.post("/api/diagnostics.zip")
    def api_diagnostics_zip() -> Response:
        """Publication-quality figure bundle for the current parameters.

        Streamed straight back rather than written to `outputs/`: this is a download, not
        an export artifact, and it should not accumulate on disk.
        """
        rdk, render, _ = parse_form(request.form)
        render = resolve_seed(render)
        diag = compute_diagnostics(rdk, render)
        blob = figures_zip(diag, rdk=rdk, render=render)
        return _download(blob, "application/zip", f"rdk-diagnostics-{rdk.method}.zip")

    def _export_params() -> tuple[RDKParams, RenderParams]:
        """Parse the form and pin the seed, so a download is reproducible from its own
        params.json and every artifact in it describes one stimulus."""
        rdk, render, sticky = parse_form(request.form)
        _remember(rdk, sticky)
        return rdk, resolve_seed(render)

    @app.post("/export/video.mp4")
    def export_video() -> Response:
        rdk, render = _export_params()
        return _download(video_bytes(rdk, render), "video/mp4", f"rdk-{rdk.method}.mp4")

    @app.post("/export/frames.zip")
    def export_frames() -> Response:
        rdk, render = _export_params()
        return _download(
            frames_zip_bytes(rdk, render), "application/zip", f"rdk-frames-{rdk.method}.zip"
        )

    @app.post("/export/params.json")
    def export_params() -> Response:
        rdk, render = _export_params()
        return _download(
            params_json(rdk, render).encode("utf-8"),
            "application/json",
            f"rdk-params-{rdk.method}.json",
        )

    @app.post("/export/bundle.zip")
    def export_bundle() -> Response:
        """Everything about one stimulus: video, frames, parameters and diagnostics.

        `everything_zip` pins the seed before generating anything, so the diagnostic plots
        measure exactly the clip they ship beside rather than a different draw.
        """
        rdk, render = _export_params()
        return _download(
            everything_zip(rdk, render), "application/zip", f"rdk-bundle-{rdk.method}.zip"
        )

    return app


app = create_app()
