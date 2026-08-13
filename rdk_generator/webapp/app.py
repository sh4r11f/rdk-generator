from __future__ import annotations

import json
import os
import uuid
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from ..export import write_frames_zip, write_mp4
from ..methods import DEFAULT_METHOD, FIELDS, GROUP_LABELS, GROUP_ORDER, get_method, methods_payload
from ..params import RDKParams, RenderParams

_RDK_FIELDS = {f.name for f in dataclass_fields(RDKParams)}
_RENDER_FIELDS = {f.name for f in dataclass_fields(RenderParams)}

# Parameters where a non-positive value means "derive a sensible default".
_AUTO_WHEN_ZERO = ("gauss_sigma_px", "min_sep_px")


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
    app.config["SESSION_SCHEMA_VERSION"] = "v2"

    output_dir = Path(app.instance_path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _reset_if_stale() -> None:
        schema = app.config.get("SESSION_SCHEMA_VERSION")
        if session.get("_schema") != schema:
            session["_schema"] = schema
            session.pop("last_job_id", None)
            session.pop("saved_params", None)
            session["has_preview"] = False

    @app.get("/")
    def index() -> str:
        _reset_if_stale()

        saved = session.get("saved_params") or {}
        selected = session.get("last_method") or DEFAULT_METHOD
        if selected not in {m["id"] for m in methods_payload()}:
            selected = DEFAULT_METHOD

        job_id = session.get("last_job_id")
        mp4_url = None
        zip_url = None
        if job_id and bool(session.get("has_preview")):
            if (output_dir / f"{job_id}.mp4").exists() and (output_dir / f"{job_id}.zip").exists():
                mp4_url = url_for("download_mp4", job_id=job_id)
                zip_url = url_for("download_frames", job_id=job_id)

        return render_template(
            "index.html",
            methods_json=json.dumps(methods_payload()),
            saved_json=json.dumps(saved),
            selected=selected,
            group_order=list(GROUP_ORDER),
            group_labels=GROUP_LABELS,
            job_id=job_id,
            mp4_url=mp4_url,
            zip_url=zip_url,
            preview_method=session.get("preview_method"),
        )

    @app.get("/api/methods")
    def api_methods() -> Response:
        return jsonify(methods_payload())

    @app.post("/generate")
    def generate() -> Response:
        _reset_if_stale()

        rdk, render, sticky = parse_form(request.form)

        # Remember values per method so switching back restores what you had.
        saved = dict(session.get("saved_params") or {})
        saved[rdk.method] = sticky
        session["saved_params"] = saved
        session["last_method"] = rdk.method

        job_id = uuid.uuid4().hex
        write_mp4(output_dir / f"{job_id}.mp4", rdk=rdk, render=render)
        write_frames_zip(output_dir / f"{job_id}.zip", rdk=rdk, render=render)

        session["last_job_id"] = job_id
        session["has_preview"] = True
        session["preview_method"] = get_method(rdk.method).label

        return redirect(url_for("index"))

    @app.get("/download/<job_id>.mp4")
    def download_mp4(job_id: str):
        return send_file(
            output_dir / f"{job_id}.mp4", as_attachment=True, download_name=f"rdk_{job_id}.mp4"
        )

    @app.get("/download/<job_id>.zip")
    def download_frames(job_id: str):
        return send_file(
            output_dir / f"{job_id}.zip",
            as_attachment=True,
            download_name=f"rdk_frames_{job_id}.zip",
        )

    @app.get("/meta/<job_id>")
    def meta(job_id: str):
        return send_file(output_dir / f"{job_id}.json", as_attachment=False)

    return app


app = create_app()
