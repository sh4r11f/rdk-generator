from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_file, url_for

from ..export import write_frames_zip, write_mp4
from ..params import RDKParams, RenderParams


def create_app() -> Flask:
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(Path(__file__).with_name("templates")),
    )

    output_dir = Path(app.instance_path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

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

        job_id = uuid.uuid4().hex
        mp4_path = output_dir / f"{job_id}.mp4"
        zip_path = output_dir / f"{job_id}.zip"

        write_mp4(mp4_path, rdk=rdk, render=render)
        write_frames_zip(zip_path, rdk=rdk, render=render)

        return redirect(url_for("result", job_id=job_id))

    @app.get("/result/<job_id>")
    def result(job_id: str) -> str:
        mp4_url = url_for("download_mp4", job_id=job_id)
        zip_url = url_for("download_frames", job_id=job_id)
        return render_template("result.html", job_id=job_id, mp4_url=mp4_url, zip_url=zip_url)

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
