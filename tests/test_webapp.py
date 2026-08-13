from pathlib import Path

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


def test_index_loads_without_a_preview(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"RDK Generator" in r.data
    assert b"No preview yet" in r.data


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
