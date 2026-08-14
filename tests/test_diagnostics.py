import numpy as np
import pytest

from rdk_generator import RDKParams, RenderParams
from rdk_generator.diagnostics import (
    compute_diagnostics,
    figure_png,
    write_diagnostics_png,
)
from rdk_generator.methods import METHODS

RENDER = RenderParams(width_px=160, height_px=160, fps=60, duration_s=0.5, seed=7)
OUR_IDS = ("gaussian_nonoverlap", "gaussian_nonoverlap_brownian")


def measure(method_id, **overrides):
    kwargs = dict(method=method_id, n_dots=120, field_diam_px=150)
    kwargs.update(overrides)
    return compute_diagnostics(RDKParams(**kwargs), RENDER, max_frames=20)


# --- shape and coverage ------------------------------------------------------


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_every_method_can_be_measured(method_id):
    diag = measure(method_id)
    assert diag.method == method_id
    assert diag.n_frames == 20
    assert diag.n_dots == 120
    for series in (diag.overlap_pairs, diag.occluded_dots, diag.relocations, diag.mean_luminance):
        assert series.shape == (20,)
    assert diag.nn_distances.size == 20 * 120


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_summary_is_json_friendly(method_id):
    summary = measure(method_id).summary()
    for key, value in summary.items():
        assert isinstance(value, (int, float, str, type(None))), f"{key} is {type(value)}"
        if isinstance(value, float):
            assert not np.isnan(value), f"{key} is NaN"


def test_measurement_window_is_capped_and_reported():
    diag = compute_diagnostics(
        RDKParams(method="brownian"), RenderParams(duration_s=5.0, fps=60), max_frames=10
    )
    assert diag.n_frames == 10
    assert any("10 of 300" in note for note in diag.notes)


# --- overlap: the reason our variants exist ----------------------------------


@pytest.mark.parametrize("method_id", OUR_IDS)
def test_our_variants_report_no_overlap_at_all(method_id):
    diag = measure(method_id, n_dots=150, dot_size_px=3)
    assert diag.overlap_pairs.max() == 0
    assert diag.occluded_dots.max() == 0
    assert diag.summary()["visible_dot_fraction"] == 1.0
    # Every pair clears the separation the engine settled on.
    assert diag.nn_distances.min() >= diag.min_sep_px - 1e-3


@pytest.mark.parametrize("method_id", ["brownian", "movshon_newsome", "white_noise"])
def test_canonical_methods_do_overlap(method_id):
    diag = measure(method_id, n_dots=250, dot_size_px=4, field_diam_px=150)
    assert diag.overlap_pairs.mean() > 0, "dense canonical fields must show overlap"
    assert diag.summary()["visible_dot_fraction"] < 1.0


def test_overlap_threshold_follows_dot_size():
    small = measure("brownian", n_dots=200, dot_size_px=2)
    large = measure("brownian", n_dots=200, dot_size_px=8)
    assert large.overlap_pairs.mean() > small.overlap_pairs.mean()


# --- the noise rules leave distinct fingerprints -----------------------------


def test_brownian_steps_are_speed_matched():
    diag = measure("brownian", speed_px_per_s=60, dot_life_frames=0)
    # Every non-relocated dot moves exactly one step, signal or noise alike.
    assert np.allclose(diag.step_magnitudes, 1.0, atol=0.05)


def test_white_noise_relocates_far_more_dots_than_brownian():
    wn = measure("white_noise", dot_life_frames=0)
    bm = measure("brownian", dot_life_frames=0)
    assert wn.relocations.mean() > 5 * bm.relocations.mean()


def test_movshon_newsome_updates_a_third_of_dots():
    diag = measure("movshon_newsome", coherence=1.0)
    assert 0.25 < diag.signal_fraction.mean() < 0.42


def test_step_directions_concentrate_on_the_signal():
    diag = measure("brownian", direction_deg=90, coherence=0.8, dot_life_frames=0)
    angles = diag.step_angles_deg % 360
    near_up = np.abs(angles - 90) < 10
    # Noise is uniform over 360 degrees, so a 20-degree window would hold ~5.5% by chance.
    assert near_up.mean() > 0.5


# --- density, coherence, luminance ------------------------------------------


def test_radial_density_is_flat_for_uniform_sampling():
    diag = measure("brownian", n_dots=800, field_diam_px=300)
    density = diag.radial_density
    # Equal-area rings, so a correct r = R*sqrt(u) sampler gives a flat profile.
    assert density.std() / density.mean() < 0.2


def test_gaussian_envelope_shows_up_as_falling_weighted_density():
    diag = measure("gaussian_nonoverlap", n_dots=400, field_diam_px=300)
    weighted = diag.radial_density_weighted
    assert weighted[0] > weighted[-1] * 3, "envelope should dim the rim"
    # The unweighted profile stays flat: only opacity changes, not placement.
    assert diag.radial_density.std() / diag.radial_density.mean() < 0.25


def test_signal_loss_is_attributed_to_lifetime_not_replanting():
    # A canonical method never replants, so all lost signal is lifecycle.
    diag = measure("brownian", dot_life_frames=6, coherence=0.5)
    assert np.nanmean(diag.lost_to_replanting) == 0.0
    assert np.nanmean(diag.lost_to_lifecycle) > 0.0


def test_disabling_dot_life_removes_most_signal_loss():
    aged = measure("brownian", dot_life_frames=4, field_diam_px=400)
    forever = measure("brownian", dot_life_frames=0, field_diam_px=400)
    assert np.nanmean(forever.signal_delivered) > np.nanmean(aged.signal_delivered)


def test_replanting_loss_appears_only_when_the_field_is_crowded():
    sparse = measure("gaussian_nonoverlap", n_dots=60, field_diam_px=300)
    assert np.nanmean(sparse.lost_to_replanting) == pytest.approx(0.0, abs=1e-3)

    crowded = compute_diagnostics(
        RDKParams(
            method="gaussian_nonoverlap",
            n_dots=700,
            dot_size_px=4,
            field_diam_px=160,
            coherence=1.0,
        ),
        RENDER,
        max_frames=8,
    )
    # It stays small even when packed tight, because noise dots are sacrificed first.
    assert np.nanmean(crowded.lost_to_replanting) < 0.05


def test_balanced_luminance_holds_the_frame_mean_at_background():
    render = RenderParams(
        width_px=160,
        height_px=160,
        fps=60,
        duration_s=0.5,
        seed=7,
        luminance_mode="balanced",
        background_lum=0.5,
    )
    diag = compute_diagnostics(
        RDKParams(method="brownian", n_dots=200, field_diam_px=150), render, max_frames=15
    )
    assert abs(diag.mean_luminance.mean() - 0.5) < 0.02
    assert diag.mean_luminance.std() < 0.01


def test_uniform_luminance_lifts_the_frame_mean_above_background():
    diag = measure("brownian", n_dots=300)
    assert diag.mean_luminance.mean() > diag.background_lum


# --- figure ------------------------------------------------------------------


def test_figure_png_renders_a_png():
    png = figure_png(measure("gaussian_nonoverlap"), dpi=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 10_000


def test_write_diagnostics_png_creates_the_file(tmp_path):
    out = write_diagnostics_png(
        tmp_path / "nested" / "diag.png",
        rdk=RDKParams(method="brownian", n_dots=60),
        render=RENDER,
        max_frames=8,
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_replanting_loss_metric_can_detect_real_loss(monkeypatch):
    """Guards the assertion above: a zero there must mean "no loss", not "blind metric".

    Swapping in a resolver that replants every dot in conflict, rather than sparing the
    ones carrying signal, should light the metric up.
    """
    from dataclasses import replace

    from rdk_generator.core import GaussianNonOverlapMNEngine
    from rdk_generator.methods import METHODS as REGISTRY

    class NaiveResolver(GaussianNonOverlapMNEngine):
        def _resolve_violations(self, max_iters: int = 3) -> None:
            for _ in range(max_iters):
                sep_sq = self.min_sep_px**2
                diff = self.xys[:, None, :] - self.xys[None, :, :]
                dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
                np.fill_diagonal(dist_sq, np.inf)
                violating = np.any(dist_sq < sep_sq, axis=1)
                if not violating.any():
                    return
                self._respawn(violating, reason="replanted")

    spec = REGISTRY["gaussian_nonoverlap"]
    monkeypatch.setitem(REGISTRY, "gaussian_nonoverlap", replace(spec, engine=NaiveResolver))

    diag = compute_diagnostics(
        RDKParams(
            method="gaussian_nonoverlap",
            n_dots=700,
            dot_size_px=4,
            field_diam_px=160,
            coherence=1.0,
        ),
        RENDER,
        max_frames=8,
    )
    assert np.nanmean(diag.lost_to_replanting) > 0.2
