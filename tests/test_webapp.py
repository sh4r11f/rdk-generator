import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rdk_generator.methods import METHODS
from rdk_generator.webapp.app import create_app, parse_form

# Exports are rendered for real rather than stubbed: they stream straight back now, so at
# this size they cost a fraction of a second and the test covers the real path.
TINY = {"n_dots": "40", "duration_s": "0.2", "width_px": "120", "height_px": "120"}


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(instance_path=str(tmp_path))
    app.config["TESTING"] = True
    test_client = app.test_client()
    test_client.instance_path = tmp_path
    return test_client


def form(**overrides):
    data = {"method": "brownian", **TINY}
    data.update(overrides)
    return data


# --- pages and registry ------------------------------------------------------


def test_index_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"RDK Studio" in r.data
    # The preview is driven client-side from the registry, not server-rendered.
    assert b"/api/preview" in r.data


def test_index_lists_every_method(client):
    body = client.get("/").get_data(as_text=True)
    for spec in METHODS.values():
        assert spec.label in body


def test_api_methods_returns_the_registry(client):
    payload = client.get("/api/methods").get_json()
    assert {m["id"] for m in payload} == set(METHODS)
    for method in payload:
        assert method["steps"] and method["references"] and method["params"]


def test_api_diagnostic_guide_is_served(client):
    guide = client.get("/api/diagnostic-guide").get_json()
    assert len(guide) == 8
    for entry in guide:
        assert entry["measures"] and entry["detects"] and entry["reading"]


# --- exports are downloads, not stored files ---------------------------------


def test_video_export_returns_an_mp4_attachment(client):
    r = client.post("/export/video.mp4", data=form())
    assert r.status_code == 200
    assert r.mimetype == "video/mp4"
    assert 'filename="rdk-brownian.mp4"' in r.headers["Content-Disposition"]
    assert r.data[4:8] == b"ftyp"


def test_frames_export_returns_a_zip_with_a_params_sidecar(client):
    r = client.post("/export/frames.zip", data=form(fps="30"))
    assert r.status_code == 200
    assert r.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.data)) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len([n for n in names if n.endswith(".png")]) == 6
        assert "params.json" in names
        assert json.loads(archive.read("params.json"))["rdk"]["method"] == "brownian"


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_every_method_exports(client, method_id):
    r = client.post("/export/video.mp4", data=form(method=method_id))
    assert r.status_code == 200
    assert f"rdk-{method_id}.mp4" in r.headers["Content-Disposition"]


def test_nothing_is_written_to_disk_by_any_request(client):
    """A serverless host has no writable filesystem shared between requests."""
    before = set(client.instance_path.rglob("*"))
    client.get("/")
    client.post("/export/video.mp4", data=form())
    client.post("/export/frames.zip", data=form())
    client.post("/api/preview", data=form())
    client.post("/api/diagnostics", data=form())
    client.post("/api/diagnostics.zip", data=form())
    assert set(client.instance_path.rglob("*")) == before


def test_export_remembers_values_per_method(client):
    client.post("/export/video.mp4", data=form(method="brownian", n_dots="123"))
    client.post("/export/video.mp4", data=form(method="white_noise", n_dots="45"))
    body = client.get("/").get_data(as_text=True)
    # Both methods' saved values survive, so switching tabs restores each one.
    assert '"n_dots": 123' in body
    assert '"n_dots": 45' in body


# --- form parsing ------------------------------------------------------------


def _parse(**overrides):
    data = {"method": "gaussian_nonoverlap"}
    data.update(overrides)
    return parse_form(data)


def test_parse_form_clamps_out_of_range_values():
    rdk, _, _ = _parse(coherence="5", n_dots="-4")
    assert rdk.coherence == 1.0
    assert rdk.n_dots == 1


def test_parse_form_ignores_unparseable_values():
    rdk, _, _ = _parse(n_dots="not a number")
    assert rdk.n_dots == 300


def test_parse_form_treats_zero_as_auto_for_derived_params():
    rdk, _, _ = _parse(gauss_sigma_px="0", min_sep_px="0")
    assert rdk.gauss_sigma_px is None
    assert rdk.min_sep_px is None


def test_parse_form_keeps_positive_derived_params():
    rdk, _, _ = _parse(gauss_sigma_px="40", min_sep_px="4.5")
    assert rdk.gauss_sigma_px == 40.0
    assert rdk.min_sep_px == 4.5


def test_parse_form_handles_blank_and_given_seeds():
    _, render, _ = _parse(seed="")
    assert render.seed is None
    _, render, _ = _parse(seed="7")
    assert render.seed == 7


def test_parse_form_rejects_invalid_choices():
    rdk, render, _ = _parse(signal_rule="diagonal", luminance_mode="sepia")
    assert rdk.signal_rule == "different"
    assert render.luminance_mode == "uniform"


def test_parse_form_ignores_params_a_method_does_not_use():
    # The Movshon-Newsome form never posts gauss_sigma_px / min_sep_px.
    rdk, _, _ = parse_form({"method": "movshon_newsome", "n_dots": "50", "n_sequences": "3"})
    assert rdk.gauss_sigma_px is None
    assert rdk.min_sep_px is None


def test_parse_form_falls_back_to_method_defaults():
    rdk, render, _ = parse_form({"method": "gaussian_nonoverlap"})
    assert rdk.n_dots == 300
    assert rdk.coherence == 0.5
    assert render.fps == 60


def test_unknown_method_falls_back_to_the_default():
    rdk, _, _ = parse_form({"method": "nonsense"})
    assert rdk.method == "gaussian_nonoverlap"


# --- live preview ------------------------------------------------------------


def _preview(client, **overrides):
    r = client.post("/api/preview", data=form(**overrides))
    assert r.status_code == 200
    return r.get_json()


def _unpack(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_preview_works_for_every_method(client, method_id):
    d = _preview(client, method=method_id, n_dots="60", duration_s="0.2", fps="30")
    assert d["method"] == method_id
    assert d["n_frames"] == 6
    assert d["n_dots"] == 60
    assert _unpack(d["xy"]).size == 6 * 60 * 2
    assert _unpack(d["alpha"]).size == 6 * 60
    assert _unpack(d["lum"]).size == 60


def test_preview_positions_stay_inside_the_aperture(client):
    d = _preview(client, n_dots="80", field_diam_px="200", duration_s="0.5")
    xy = _unpack(d["xy"]).reshape(d["n_frames"], d["n_dots"], 2)
    assert np.linalg.norm(xy, axis=2).max() <= 100.0 + 1e-3


def test_preview_reports_the_gaussian_envelope(client):
    ours = _preview(client, method="gaussian_nonoverlap", n_dots="200", field_diam_px="200")
    plain = _preview(client, method="movshon_newsome", n_dots="200", field_diam_px="200")
    # Our variant fades dots toward the rim; the canonical base does not.
    assert _unpack(ours["alpha"]).min() < 0.9
    assert np.allclose(_unpack(plain["alpha"]), 1.0)


def test_preview_truncates_long_clips_but_reports_the_full_length(client):
    d = _preview(client, duration_s="10", fps="60")
    assert d["truncated"] is True
    assert d["n_frames"] < d["total_frames"] == 600


def test_short_preview_is_not_marked_truncated(client):
    d = _preview(client, duration_s="0.5", fps="60")
    assert d["truncated"] is False
    assert d["n_frames"] == d["total_frames"] == 30


def test_preview_is_deterministic_for_a_given_seed(client):
    a = _preview(client, seed="42", duration_s="0.3")
    b = _preview(client, seed="42", duration_s="0.3")
    assert a["xy"] == b["xy"]


def test_preview_echoes_geometry_the_canvas_needs(client):
    d = _preview(
        client, width_px="320", height_px="240", dot_size_px="5", background_lum="0.25", fps="24"
    )
    assert (d["width"], d["height"]) == (320, 240)
    assert d["dot_size"] == 5
    assert d["background"] == 0.25
    assert d["fps"] == 24
    assert d["center"] == [0.0, 0.0]


# --- diagnostics -------------------------------------------------------------


def test_diagnostics_endpoint_returns_a_figure_and_summary(client):
    d = client.post("/api/diagnostics", data=form(n_dots="60")).get_json()
    assert base64.b64decode(d["png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert d["summary"]["method"] == "brownian"
    assert d["summary"]["n_dots"] == 60
    assert "visible_dot_fraction" in d["summary"]


def test_diagnostics_reports_our_variant_as_overlap_free(client):
    d = client.post(
        "/api/diagnostics",
        data=form(method="gaussian_nonoverlap", n_dots="80", field_diam_px="150"),
    ).get_json()
    assert d["summary"]["overlapping_pairs_per_frame"] == 0.0
    assert d["summary"]["visible_dot_fraction"] == 1.0


def test_diagnostics_zip_endpoint_returns_a_downloadable_bundle(client):
    r = client.post("/api/diagnostics.zip", data=form())
    assert r.status_code == 200
    assert r.mimetype == "application/zip"
    assert "rdk-diagnostics-brownian.zip" in r.headers["Content-Disposition"]

    with zipfile.ZipFile(io.BytesIO(r.data)) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "diagnostics.pdf" in names
        assert sum(1 for n in names if n.startswith("panels/")) == 16
