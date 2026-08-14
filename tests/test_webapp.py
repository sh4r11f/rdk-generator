import base64
from pathlib import Path

import numpy as np
import pytest

from rdk_generator.methods import METHODS
from rdk_generator.webapp.app import create_app, parse_form


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    """App with the exporters stubbed out, so tests never render real video."""
    from rdk_generator.webapp import app as app_mod

    captured: dict = {}

    def _fake_write_mp4(out_path, *, rdk, render):
        captured["rdk"] = rdk
        captured["render"] = render
        Path(out_path).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return Path(out_path)

    def _fake_write_zip(out_path, *, rdk, render, image_format="png"):
        Path(out_path).write_bytes(b"PK\x03\x04")
        return Path(out_path)

    # Patch on the webapp module, which is what imported these names.
    monkeypatch.setattr(app_mod, "write_mp4", _fake_write_mp4)
    monkeypatch.setattr(app_mod, "write_frames_zip", _fake_write_zip)

    app = create_app(instance_path=str(tmp_path))
    app.config["TESTING"] = True
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    test_client = app.test_client()
    test_client.captured = captured
    return test_client


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


def test_generate_renders_preview_and_keeps_values_sticky(client):
    r = client.post(
        "/generate",
        data={
            "method": "brownian",
            "n_dots": "111",
            "dot_size_px": "3",
            "speed_px_per_s": "120",
            "dot_life_frames": "12",
            "signal_rule": "same",
            "direction_deg": "0",
            "coherence": "0.5",
            "field_diam_px": "300",
            "width_px": "256",
            "height_px": "256",
            "fps": "30",
            "duration_s": "0.2",
            "background_lum": "0.5",
            "dot_contrast": "1.0",
            "luminance_mode": "uniform",
            "seed": "",
        },
        follow_redirects=True,
    )

    assert r.status_code == 200
    assert b"Download .mp4" in r.data
    assert b"Frames (.zip)" in r.data
    # Sticky values are handed back to the page for the form to restore.
    assert b'"n_dots": 111' in r.data

    rdk = client.captured["rdk"]
    assert rdk.method == "brownian"
    assert rdk.n_dots == 111
    assert rdk.signal_rule == "same"
    assert client.captured["render"].seed is None


def test_generate_accepts_a_method_without_our_extra_fields(client):
    # The MN form never posts gauss_sigma_px / min_sep_px.
    r = client.post(
        "/generate",
        data={"method": "movshon_newsome", "n_dots": "50", "n_sequences": "3"},
        follow_redirects=True,
    )
    assert r.status_code == 200
    rdk = client.captured["rdk"]
    assert rdk.method == "movshon_newsome"
    assert rdk.gauss_sigma_px is None
    assert rdk.min_sep_px is None


def test_generate_falls_back_to_method_defaults_for_missing_fields(client):
    client.post("/generate", data={"method": "gaussian_nonoverlap"}, follow_redirects=True)
    rdk = client.captured["rdk"]
    render = client.captured["render"]
    assert rdk.n_dots == 300
    assert rdk.coherence == 0.5
    assert render.fps == 60


def test_unknown_method_falls_back_to_the_default(client):
    client.post("/generate", data={"method": "nonsense"}, follow_redirects=True)
    assert client.captured["rdk"].method == "gaussian_nonoverlap"


def test_each_method_remembers_its_own_values(client):
    client.post("/generate", data={"method": "brownian", "n_dots": "123"}, follow_redirects=True)
    client.post("/generate", data={"method": "white_noise", "n_dots": "456"}, follow_redirects=True)
    body = client.get("/").get_data(as_text=True)
    # Both methods' saved values survive, so switching tabs restores each one.
    assert '"n_dots": 123' in body
    assert '"n_dots": 456' in body


# --- form parsing ------------------------------------------------------------


def _parse(**form):
    form.setdefault("method", "gaussian_nonoverlap")
    return parse_form(form)


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


# --- live preview ------------------------------------------------------------


def _preview(client, **form):
    form.setdefault("method", "gaussian_nonoverlap")
    r = client.post("/api/preview", data=form)
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
    d = _preview(client, method="brownian", n_dots="80", field_diam_px="200", duration_s="0.5")
    xy = _unpack(d["xy"]).reshape(d["n_frames"], d["n_dots"], 2)
    assert np.linalg.norm(xy, axis=2).max() <= 100.0 + 1e-3


def test_preview_reports_the_gaussian_envelope(client):
    ours = _preview(client, method="gaussian_nonoverlap", n_dots="200", field_diam_px="200")
    plain = _preview(client, method="movshon_newsome", n_dots="200", field_diam_px="200")
    # Our variant fades dots toward the rim; the canonical base does not.
    assert _unpack(ours["alpha"]).min() < 0.9
    assert np.allclose(_unpack(plain["alpha"]), 1.0)


def test_preview_truncates_long_clips_but_reports_the_full_length(client):
    d = _preview(client, method="brownian", duration_s="10", fps="60")
    assert d["truncated"] is True
    assert d["n_frames"] < d["total_frames"] == 600


def test_short_preview_is_not_marked_truncated(client):
    d = _preview(client, method="brownian", duration_s="0.5", fps="60")
    assert d["truncated"] is False
    assert d["n_frames"] == d["total_frames"] == 30


def test_preview_is_deterministic_for_a_given_seed(client):
    a = _preview(client, method="brownian", seed="42", duration_s="0.3")
    b = _preview(client, method="brownian", seed="42", duration_s="0.3")
    assert a["xy"] == b["xy"]


def test_preview_writes_nothing_to_disk(client, tmp_path):
    outputs = tmp_path / "outputs"
    before = set(outputs.iterdir())
    _preview(client, method="brownian", duration_s="0.2")
    assert set(outputs.iterdir()) == before


def test_preview_echoes_geometry_the_canvas_needs(client):
    d = _preview(
        client,
        method="brownian",
        width_px="320",
        height_px="240",
        dot_size_px="5",
        background_lum="0.25",
        fps="24",
    )
    assert (d["width"], d["height"]) == (320, 240)
    assert d["dot_size"] == 5
    assert d["background"] == 0.25
    assert d["fps"] == 24
    assert d["center"] == [0.0, 0.0]


# --- diagnostics -------------------------------------------------------------


def test_diagnostics_endpoint_returns_a_figure_and_summary(client):
    r = client.post(
        "/api/diagnostics",
        data={
            "method": "brownian",
            "n_dots": "60",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
    )
    assert r.status_code == 200
    d = r.get_json()
    assert base64.b64decode(d["png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert d["summary"]["method"] == "brownian"
    assert d["summary"]["n_dots"] == 60
    assert "visible_dot_fraction" in d["summary"]


def test_diagnostics_endpoint_writes_nothing_to_disk(client, tmp_path):
    outputs = tmp_path / "outputs"
    before = set(outputs.iterdir())
    client.post(
        "/api/diagnostics",
        data={
            "method": "brownian",
            "n_dots": "40",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
    )
    assert set(outputs.iterdir()) == before


def test_diagnostics_reports_our_variant_as_overlap_free(client):
    d = client.post(
        "/api/diagnostics",
        data={
            "method": "gaussian_nonoverlap",
            "n_dots": "80",
            "duration_s": "0.2",
            "field_diam_px": "150",
            "width_px": "160",
            "height_px": "160",
        },
    ).get_json()
    assert d["summary"]["overlapping_pairs_per_frame"] == 0.0
    assert d["summary"]["visible_dot_fraction"] == 1.0


def test_export_writes_a_diagnostics_png_and_links_it(client, tmp_path):
    r = client.post(
        "/generate",
        data={
            "method": "brownian",
            "n_dots": "40",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"Diagnostics (.png)" in r.data
    written = list((tmp_path / "outputs").glob("*_diagnostics.png"))
    assert len(written) == 1
    assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_diagnostics_png_is_downloadable(client):
    client.post(
        "/generate",
        data={
            "method": "brownian",
            "n_dots": "40",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
        follow_redirects=True,
    )
    body = client.get("/").get_data(as_text=True)
    url = body.split('href="')[1:]
    diag = next(u.split('"')[0] for u in url if "_diagnostics.png" in u)
    r = client.get(diag)
    assert r.status_code == 200
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_diagnostics_zip_endpoint_returns_a_downloadable_bundle(client):
    import io
    import zipfile

    r = client.post(
        "/api/diagnostics.zip",
        data={
            "method": "brownian",
            "n_dots": "40",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
    )
    assert r.status_code == 200
    assert r.mimetype == "application/zip"
    assert "attachment" in r.headers["Content-Disposition"]
    assert "rdk-diagnostics-brownian.zip" in r.headers["Content-Disposition"]

    with zipfile.ZipFile(io.BytesIO(r.data)) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "diagnostics.pdf" in names
        assert sum(1 for n in names if n.startswith("panels/")) == 16


def test_diagnostics_zip_writes_nothing_to_disk(client, tmp_path):
    outputs = tmp_path / "outputs"
    before = set(outputs.iterdir())
    client.post(
        "/api/diagnostics.zip",
        data={
            "method": "brownian",
            "n_dots": "40",
            "duration_s": "0.2",
            "width_px": "120",
            "height_px": "120",
        },
    )
    assert set(outputs.iterdir()) == before
