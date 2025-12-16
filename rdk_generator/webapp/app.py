from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_file, session, url_for

from ..export import write_frames_zip, write_mp4
from ..params import RDKParams, RenderParams


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
    app.config["SESSION_SCHEMA_VERSION"] = "v1"

    output_dir = Path(app.instance_path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> str:
        last_params = session.get("last_params") or {}

        # Ensure "first view" never auto-shows a preview.
        # If the schema version changed (or session is new), we keep params but clear preview.
        schema = app.config.get("SESSION_SCHEMA_VERSION")
        if session.get("_schema") != schema:
            session["_schema"] = schema
            session.pop("last_job_id", None)
            session["has_preview"] = False

        # Default UI preferences (while still allowing sticky overrides)
        if "prevent_overlap" not in last_params:
            last_params["prevent_overlap"] = "on"
        if "min_sep_px" not in last_params or last_params.get("min_sep_px") in (None, ""):
            try:
                dot_size = float(last_params.get("dot_size_px", 3))
            except Exception:
                dot_size = 3.0
            last_params["min_sep_px"] = round(dot_size * 1.1, 2)

        job_id = session.get("last_job_id")
        mp4_url = None
        zip_url = None
        if job_id and bool(session.get("has_preview")):
            mp4_path = output_dir / f"{job_id}.mp4"
            zip_path = output_dir / f"{job_id}.zip"
            if mp4_path.exists() and zip_path.exists():
                mp4_url = url_for("download_mp4", job_id=job_id)
                zip_url = url_for("download_frames", job_id=job_id)

        return render_template(
            "index.html",
            params=last_params,
            job_id=job_id,
            mp4_url=mp4_url,
            zip_url=zip_url,
        )

    def _parse_int(name: str, default: int) -> int:
        try:
            return int(request.form.get(name, default))
        except Exception:
            return default

    def _parse_float(name: str, default: float) -> float:
        try:
            return float(request.form.get(name, default))
        except Exception:
            return default

    @app.post("/generate")
    def generate() -> Response:
        # If a user posts to /generate before ever loading /, ensure we don't treat
        # the redirect target as a "first view" that must hide the preview.
        schema = app.config.get("SESSION_SCHEMA_VERSION")
        if session.get("_schema") != schema:
            session["_schema"] = schema

        prevent_overlap = bool(request.form.get("prevent_overlap", "") == "on")
        min_sep_px = _parse_float("min_sep_px", 0.0)
        if not prevent_overlap:
            min_sep_val = None
        else:
            if min_sep_px <= 0:
                # Sensible default when enabled
                min_sep_val = float(_parse_int("dot_size_px", 3)) * 1.1
            else:
                min_sep_val = float(min_sep_px)

        rdk = RDKParams(
            n_dots=_parse_int("n_dots", 300),
            dot_size_px=_parse_int("dot_size_px", 3),
            speed_px_per_s=_parse_float("speed_px_per_s", 120.0),
            dot_life_frames=_parse_int("dot_life_frames", 12),
            direction_deg=_parse_float("direction_deg", 0.0),
            coherence=_parse_float("coherence", 0.5),
            field_diam_px=_parse_int("field_diam_px", 300),
            field_center_xy_px=(0.0, 0.0),
            gauss_sigma_px=(
                _parse_float("gauss_sigma_px", 0.0)
                if _parse_float("gauss_sigma_px", 0.0) > 0
                else None
            ),
            reassign_life=bool(request.form.get("reassign_life", "on") == "on"),
            min_sep_px=min_sep_val,
        )

        render = RenderParams(
            width_px=_parse_int("width_px", 512),
            height_px=_parse_int("height_px", 512),
            fps=_parse_int("fps", 60),
            duration_s=_parse_float("duration_s", 1.0),
            background_lum=_parse_float("background_lum", 0.5),
            dot_contrast=_parse_float("dot_contrast", 1.0),
            seed=_parse_int("seed", 0) if request.form.get("seed", "") != "" else None,
        )

        # Remember last-used params so the UI stays sticky.
        session["last_params"] = {
            "n_dots": rdk.n_dots,
            "dot_size_px": rdk.dot_size_px,
            "speed_px_per_s": rdk.speed_px_per_s,
            "dot_life_frames": rdk.dot_life_frames,
            "direction_deg": rdk.direction_deg,
            "coherence": rdk.coherence,
            "field_diam_px": rdk.field_diam_px,
            "gauss_sigma_px": (rdk.gauss_sigma_px or 0),
            "width_px": render.width_px,
            "height_px": render.height_px,
            "fps": render.fps,
            "duration_s": render.duration_s,
            "background_lum": render.background_lum,
            "dot_contrast": render.dot_contrast,
            "seed": ("" if render.seed is None else render.seed),
            "reassign_life": ("on" if rdk.reassign_life else ""),
            "prevent_overlap": ("on" if rdk.min_sep_px is not None else ""),
            "min_sep_px": ("" if rdk.min_sep_px is None else rdk.min_sep_px),
        }

        job_id = uuid.uuid4().hex
        mp4_path = output_dir / f"{job_id}.mp4"
        zip_path = output_dir / f"{job_id}.zip"

        write_mp4(mp4_path, rdk=rdk, render=render)
        write_frames_zip(zip_path, rdk=rdk, render=render)

        session["last_job_id"] = job_id
        session["has_preview"] = True

        return redirect(url_for("index"))

    @app.get("/download/<job_id>.mp4")
    def download_mp4(job_id: str):
        path = output_dir / f"{job_id}.mp4"
        return send_file(path, as_attachment=True, download_name=f"rdk_{job_id}.mp4")

    @app.get("/download/<job_id>.zip")
    def download_frames(job_id: str):
        path = output_dir / f"{job_id}.zip"
        return send_file(path, as_attachment=True, download_name=f"rdk_frames_{job_id}.zip")

    @app.get("/meta/<job_id>")
    def meta(job_id: str):
        path = output_dir / f"{job_id}.json"
        return send_file(path, as_attachment=False)

    return app


app = create_app()
