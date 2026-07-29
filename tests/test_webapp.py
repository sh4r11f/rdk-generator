from pathlib import Path

from rdk_generator.webapp.app import create_app


def test_index_loads():
    app = create_app()
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"RDK Generator" in r.data
    assert b"No preview yet" in r.data


def test_generate_renders_preview_on_index(monkeypatch, tmp_path: Path):
    # Patch exporters to avoid doing real video work in tests.
    from rdk_generator.webapp import app as app_mod

    def _fake_write_mp4(out_path, *, rdk, render):
        Path(out_path).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return Path(out_path)

    def _fake_write_zip(out_path, *, rdk, render, image_format="png"):
        Path(out_path).write_bytes(b"PK\x03\x04")
        return Path(out_path)

    monkeypatch.setattr(app_mod, "write_mp4", _fake_write_mp4)
    monkeypatch.setattr(app_mod, "write_frames_zip", _fake_write_zip)

    app = create_app(instance_path=str(tmp_path))
    app.config["TESTING"] = True
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    client = app.test_client()
    r = client.post(
        "/generate",
        data={
            "n_dots": "111",
            "dot_size_px": "3",
            "speed_px_per_s": "120",
            "dot_life_frames": "12",
            "direction_deg": "0",
            "coherence": "0.5",
            "field_diam_px": "300",
            "gauss_sigma_px": "0",
            "width_px": "256",
            "height_px": "256",
            "fps": "30",
            "duration_s": "0.2",
            "background_lum": "0.5",
            "dot_contrast": "1.0",
            "seed": "",
            "reassign_life": "on",
            "prevent_overlap": "on",
            "min_sep_px": "4.0",
        },
        follow_redirects=True,
    )

    assert r.status_code == 200
    assert b"Download .mp4" in r.data
    assert b"Download frames" in r.data
    # Sticky form value
    assert b'value="111"' in r.data
