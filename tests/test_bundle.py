import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from rdk_generator import RDKParams, RenderParams, build_engine, render_frames, video_bytes
from rdk_generator.bundle import everything_zip
from rdk_generator.diagnostics import PANEL_KEYS, compute_diagnostics
from rdk_generator.export import params_json, resolve_seed
from rdk_generator.methods import METHODS

RENDER = RenderParams(width_px=140, height_px=140, fps=30, duration_s=0.4, seed=None)
RDK = RDKParams(method="gaussian_nonoverlap", n_dots=60, field_diam_px=120)


def open_bundle(blob: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(blob))


# --- the seed, which is what makes a bundle self-consistent -------------------


def test_a_blank_seed_really_does_give_a_different_stimulus_each_time():
    """The reason a bundle has to pin its seed before generating anything."""
    blank = replace(RENDER, seed=None)
    first = build_engine(RDK, blank).xys
    second = build_engine(RDK, blank).xys
    assert not (first == second).all()


def test_resolve_seed_pins_a_blank_seed_and_leaves_a_given_one():
    pinned = resolve_seed(replace(RENDER, seed=None))
    assert isinstance(pinned.seed, int)
    assert resolve_seed(replace(RENDER, seed=99)).seed == 99


def test_bundle_records_the_seed_it_used():
    with open_bundle(everything_zip(RDK, replace(RENDER, seed=None))) as archive:
        assert json.loads(archive.read("params.json"))["render"]["seed"] is not None


def test_bundle_frames_are_reproducible_from_its_own_params():
    """Re-rendering from params.json returns the same stimulus, pixel for pixel.

    Frames rather than the .mp4, because encoded bytes depend on the ffmpeg build: the
    frames really do reproduce on another machine, the container need not.
    """
    blob = everything_zip(RDK, replace(RENDER, seed=None))
    with open_bundle(blob) as archive:
        recorded = json.loads(archive.read("params.json"))
        packaged = [
            np.array(Image.open(io.BytesIO(archive.read(name))))
            for name in sorted(n for n in archive.namelist() if n.startswith("frames/"))
        ]

    rebuilt = render_frames(RDKParams(**recorded["rdk"]), RenderParams(**recorded["render"]))
    assert len(rebuilt) == len(packaged)
    assert all(np.array_equal(a, b) for a, b in zip(rebuilt, packaged))


def test_bundle_video_is_deterministic_on_one_machine():
    blob = everything_zip(RDK, replace(RENDER, seed=None))
    with open_bundle(blob) as archive:
        recorded = json.loads(archive.read("params.json"))
        packaged_video = archive.read("stimulus.mp4")

    rebuilt = replace(RENDER, seed=recorded["render"]["seed"])
    assert video_bytes(RDK, rebuilt) == packaged_video


def test_bundle_diagnostics_describe_the_bundled_clip():
    """The plots must measure the video shipped beside them, not a different draw."""
    blob = everything_zip(RDK, replace(RENDER, seed=None))
    with open_bundle(blob) as archive:
        recorded = json.loads(archive.read("params.json"))
        packaged_summary = json.loads(archive.read("diagnostics/data/summary.json"))

    rebuilt = replace(RENDER, seed=recorded["render"]["seed"])
    recomputed = compute_diagnostics(RDK, rebuilt, max_frames=None).summary()
    assert recomputed == packaged_summary


def test_bundle_diagnostics_cover_every_exported_frame():
    blob = everything_zip(RDK, replace(RENDER, seed=3))
    with open_bundle(blob) as archive:
        exported = sum(1 for n in archive.namelist() if n.startswith("frames/"))
        summary = json.loads(archive.read("diagnostics/data/summary.json"))
    assert summary["n_frames"] == exported


# --- contents ----------------------------------------------------------------


def test_bundle_contains_everything():
    with open_bundle(everything_zip(RDK, replace(RENDER, seed=1))) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())

        assert {"params.json", "stimulus.mp4", "README.txt"} <= names
        assert "diagnostics/diagnostics.png" in names
        assert "diagnostics/diagnostics.pdf" in names
        assert "diagnostics/data/per_frame.csv" in names
        assert "diagnostics/data/radial_density.csv" in names
        for index, key in enumerate(PANEL_KEYS, start=1):
            assert f"diagnostics/panels/{index:02d}_{key}.png" in names
            assert f"diagnostics/panels/{index:02d}_{key}.pdf" in names

        # 0.4s at 30fps
        assert sum(1 for n in names if n.startswith("frames/")) == 12
        assert archive.read("stimulus.mp4")[4:8] == b"ftyp"
        assert archive.read("diagnostics/diagnostics.pdf")[:5] == b"%PDF-"


def test_bundle_readme_names_the_method_and_seed():
    readme = None
    with open_bundle(everything_zip(RDK, replace(RENDER, seed=11))) as archive:
        readme = archive.read("README.txt").decode()
    assert "Gaussian non-overlapping (MN)" in readme
    assert "seed 11" in readme


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_every_method_bundles(method_id):
    blob = everything_zip(
        RDKParams(method=method_id, n_dots=40, field_diam_px=100),
        RenderParams(width_px=120, height_px=120, fps=20, duration_s=0.2, seed=2),
    )
    with open_bundle(blob) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("params.json"))["method"]["id"] == method_id


# --- params.json -------------------------------------------------------------


def test_params_json_names_the_method():
    payload = json.loads(params_json(RDK, RENDER))
    assert payload["method"]["id"] == "gaussian_nonoverlap"
    assert payload["method"]["label"] == METHODS["gaussian_nonoverlap"].label
    assert payload["method"]["tagline"]
    # The dataclasses stay verbatim so the file can be fed back in.
    assert payload["rdk"]["method"] == "gaussian_nonoverlap"
    assert payload["rdk"]["n_dots"] == 60
    assert payload["render"]["fps"] == 30


def test_params_json_round_trips_into_the_dataclasses():
    payload = json.loads(params_json(RDK, resolve_seed(RENDER)))
    rebuilt_rdk = RDKParams(**payload["rdk"])
    rebuilt_render = RenderParams(**payload["render"])
    assert rebuilt_rdk == RDK
    assert build_engine(rebuilt_rdk, rebuilt_render) is not None
