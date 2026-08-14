import numpy as np
import pytest

from rdk_generator.core import (
    BrownianRDKEngine,
    FrameRenderer,
    GaussianNonOverlapMNEngine,
    GaussianNonOverlapRDKEngine,
    MovshonNewsomeRDKEngine,
    RandomDirectionRDKEngine,
    WhiteNoiseRDKEngine,
)

# A field roomy enough that only dots near its rim can exit during a test.
WIDE_FIELD = 1200
# Dots starting this far inside the rim cannot reach it in one step at test speeds.
INTERIOR_MARGIN = 20.0

# Positions are float32, so at these coordinate magnitudes a 1 px step cannot be
# resolved more finely than ~1e-4. Still far tighter than a respawn's hundreds of px.
STEP_TOL = 1e-3


def build(cls, **overrides):
    kwargs = dict(
        n_dots=200,
        coherence=0.5,
        direction_deg=0,
        speed_px_per_s=60,
        field_diam_px=200,
        fps=60,
        seed=11,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def step_and_measure(engine):
    """Advance one frame; return per-dot displacement and an "could not have exited" mask.

    Dots that leave the aperture are respawned at a random position, which registers as a
    huge displacement. Tests that care about step *size* must exclude them, so the mask
    marks dots that started far enough inside the rim to be unable to reach it this frame.
    """
    before = engine.xys.copy()
    engine.step()
    moved = np.linalg.norm(engine.xys - before, axis=1)
    interior = np.linalg.norm(before, axis=1) < engine.radius - INTERIOR_MARGIN
    return moved, interior


def min_pairwise_distance(xys: np.ndarray) -> float:
    dist = np.linalg.norm(xys[:, None, :] - xys[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))


NON_OVERLAP_ENGINES = [GaussianNonOverlapMNEngine, GaussianNonOverlapRDKEngine]

ALL_ENGINES = [
    BrownianRDKEngine,
    WhiteNoiseRDKEngine,
    RandomDirectionRDKEngine,
    MovshonNewsomeRDKEngine,
    *NON_OVERLAP_ENGINES,
]


def _extra(cls):
    return {"min_sep_px": 4.0} if cls in NON_OVERLAP_ENGINES else {}


# --- shared invariants -------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_ENGINES)
def test_dots_stay_inside_aperture(cls):
    engine = build(cls, n_dots=150, field_diam_px=200, **_extra(cls))
    for _ in range(20):
        engine.step()
        radii = np.linalg.norm(engine.xys, axis=1)
        assert radii.max() <= engine.radius + 1e-3


@pytest.mark.parametrize("cls", ALL_ENGINES)
def test_same_seed_reproduces_positions(cls):
    a = build(cls, n_dots=80, **_extra(cls))
    b = build(cls, n_dots=80, **_extra(cls))
    for _ in range(6):
        a.step()
        b.step()
    assert np.array_equal(a.xys, b.xys)


@pytest.mark.parametrize("cls", [BrownianRDKEngine, WhiteNoiseRDKEngine, RandomDirectionRDKEngine])
def test_coherence_is_an_exact_count(cls):
    engine = build(cls, n_dots=1000, coherence=0.25)
    assert engine.is_signal.sum() == 250


# --- per-method motion rules -------------------------------------------------


def test_brownian_noise_is_speed_matched():
    # Every dot, signal or noise, moves exactly one step per frame.
    engine = build(BrownianRDKEngine, field_diam_px=WIDE_FIELD, dot_life_frames=0)
    moved, interior = step_and_measure(engine)
    assert np.allclose(moved[interior], 1.0, atol=STEP_TOL)


def test_white_noise_relocates_noise_dots_only():
    engine = build(
        WhiteNoiseRDKEngine, field_diam_px=WIDE_FIELD, dot_life_frames=0, signal_rule="same"
    )
    signal = engine.is_signal.copy()
    moved, interior = step_and_measure(engine)
    assert np.allclose(moved[signal & interior], 1.0, atol=STEP_TOL)
    # Relocation is across the whole aperture, so noise jumps are far larger.
    assert np.median(moved[~signal]) > 100


def test_random_direction_keeps_each_heading_for_life():
    engine = build(
        RandomDirectionRDKEngine,
        field_diam_px=WIDE_FIELD,
        dot_life_frames=0,
        signal_rule="same",
    )
    noise = ~engine.is_signal

    before = engine.xys.copy()
    engine.step()
    first = engine.xys - before
    stayed = np.linalg.norm(before, axis=1) < engine.radius - INTERIOR_MARGIN

    before = engine.xys.copy()
    engine.step()
    second = engine.xys - before
    stayed &= np.linalg.norm(before, axis=1) < engine.radius - INTERIOR_MARGIN

    keep = noise & stayed
    assert np.allclose(first[keep], second[keep], atol=STEP_TOL)
    # ...and headings differ between dots.
    assert np.unique(np.round(first[keep], 4), axis=0).shape[0] > 1


def test_random_direction_redraws_heading_on_respawn():
    engine = build(RandomDirectionRDKEngine, n_dots=50, dot_life_frames=1, signal_rule="same")
    before = engine.dot_dir.copy()
    engine.step()  # every dot expires and respawns
    assert not np.allclose(before, engine.dot_dir)


def test_movshon_newsome_updates_one_sequence_per_frame():
    engine = build(
        MovshonNewsomeRDKEngine,
        n_dots=900,
        coherence=1.0,
        field_diam_px=WIDE_FIELD,
        dot_life_frames=0,
    )
    updated = []
    never_near_rim = np.ones(engine.n, dtype=bool)
    for _ in range(6):
        moved, interior = step_and_measure(engine)
        updated.append(moved > 1e-6)
        never_near_rim &= interior
    updated = np.array(updated)[:, never_near_rim]

    # One third of dots move on any given frame...
    assert 0.28 < updated.mean() < 0.39
    # ...and each dot moves exactly twice in six frames (every third frame).
    assert set(np.unique(updated.sum(axis=0))) == {2}


def test_movshon_newsome_scales_step_by_sequence_count():
    engine = build(
        MovshonNewsomeRDKEngine,
        coherence=1.0,
        n_sequences=3,
        speed_px_per_s=60,
        fps=60,
        field_diam_px=WIDE_FIELD,
        dot_life_frames=0,
    )
    moved, interior = step_and_measure(engine)
    # A dot updates every 3rd frame, so it travels 3 frames' worth when it does.
    stepped = interior & (moved > 1e-6)
    assert stepped.any()
    assert np.allclose(moved[stepped], 3.0, atol=STEP_TOL)


def test_movshon_newsome_coherence_is_probabilistic():
    engine = build(MovshonNewsomeRDKEngine, n_dots=3000, coherence=0.5, field_diam_px=WIDE_FIELD)
    engine.step()
    active = engine.sequence == 0
    realised = engine.is_signal[active].mean()
    assert 0.4 < realised < 0.6
    # Exact-count selection would give precisely 0.5; a Bernoulli draw essentially never does.
    assert engine.is_signal.sum() != round(active.sum() * 0.5)


# --- signal rule and dot life ------------------------------------------------


def test_signal_rule_same_keeps_membership_fixed():
    engine = build(BrownianRDKEngine, signal_rule="same", dot_life_frames=0)
    original = engine.is_signal.copy()
    for _ in range(10):
        engine.step()
    assert np.array_equal(original, engine.is_signal)


def test_signal_rule_different_reshuffles_membership():
    engine = build(BrownianRDKEngine, signal_rule="different", dot_life_frames=0)
    original = engine.is_signal.copy()
    engine.step()
    assert not np.array_equal(original, engine.is_signal)
    assert engine.is_signal.sum() == original.sum()


def test_rejects_unknown_signal_rule():
    with pytest.raises(ValueError, match="signal_rule"):
        build(BrownianRDKEngine, signal_rule="sideways")


def test_zero_dot_life_disables_ageing():
    engine = build(BrownianRDKEngine, dot_life_frames=0, field_diam_px=WIDE_FIELD)
    for _ in range(30):
        moved, interior = step_and_measure(engine)
        # No dot ever teleports: the only large jumps would come from ageing.
        assert np.allclose(moved[interior], 1.0, atol=STEP_TOL)


def test_dot_life_expiry_respawns_dots():
    engine = build(BrownianRDKEngine, n_dots=60, dot_life_frames=1, field_diam_px=WIDE_FIELD)
    # Every dot expires on the first step, so displacements are relocations, not steps.
    moved, _ = step_and_measure(engine)
    assert np.median(moved) > 100


# --- Gaussian non-overlapping variant ---------------------------------------


def test_non_overlap_respects_min_sep_initially_and_after_steps():
    engine = build(
        GaussianNonOverlapRDKEngine,
        min_sep_px=4.0,
        n_dots=150,
        dot_life_frames=6,
        field_diam_px=200,
    )
    assert engine.min_sep_px == pytest.approx(4.0), "this density should not need relaxing"
    assert min_pairwise_distance(engine.xys) >= 3.95
    for _ in range(5):
        engine.step()
        assert min_pairwise_distance(engine.xys) >= 3.95


def test_non_overlap_inherits_brownian_speed_matching():
    engine = build(
        GaussianNonOverlapRDKEngine,
        min_sep_px=4.0,
        n_dots=60,
        dot_life_frames=0,
        field_diam_px=WIDE_FIELD,
    )
    # Sparse enough that no dot needs replanting, so every dot takes a clean step.
    moved, interior = step_and_measure(engine)
    assert np.allclose(moved[interior], 1.0, atol=STEP_TOL)


def test_gaussian_envelope_defaults_to_quarter_field_and_fades_with_radius():
    engine = build(GaussianNonOverlapRDKEngine, min_sep_px=4.0, field_diam_px=200)
    assert engine.gauss_sigma == pytest.approx(50.0)

    engine.xys = np.array([[0.0, 0.0], [50.0, 0.0], [99.0, 0.0]], dtype=np.float32)
    engine.n = 3
    alpha = engine.compute_opacity()
    assert alpha[0] == pytest.approx(1.0)
    assert alpha[0] > alpha[1] > alpha[2]
    assert alpha[1] == pytest.approx(np.exp(-0.5), abs=1e-6)


def test_canonical_methods_have_no_envelope():
    engine = build(BrownianRDKEngine)
    assert engine.gauss_sigma == 0.0
    assert np.all(engine.compute_opacity() == 1.0)


def test_min_sep_must_be_positive():
    with pytest.raises(ValueError, match="min_sep_px"):
        build(GaussianNonOverlapRDKEngine, min_sep_px=0.0)


# --- luminance ---------------------------------------------------------------


def test_uniform_luminance_is_a_single_value():
    engine = build(BrownianRDKEngine, background_lum=0.5, dot_contrast=1.0)
    assert np.unique(engine.dot_lum).tolist() == [1.0]


def test_balanced_luminance_uses_two_levels_around_background():
    engine = build(
        BrownianRDKEngine, luminance_mode="balanced", background_lum=0.5, dot_contrast=1.0
    )
    levels = np.unique(engine.dot_lum)
    assert len(levels) == 2
    assert engine.dot_lum.mean() == pytest.approx(0.5, abs=0.01)


def test_rejects_unknown_luminance_mode():
    with pytest.raises(ValueError, match="luminance_mode"):
        build(BrownianRDKEngine, luminance_mode="sepia")


# --- renderer ----------------------------------------------------------------


def test_renderer_outputs_uint8_of_requested_shape():
    engine = build(BrownianRDKEngine, n_dots=200, field_diam_px=150)
    renderer = FrameRenderer(
        width_px=256,
        height_px=256,
        background_lum=0.4,
        field_center_xy_px=(0.0, 0.0),
        dot_size_px=3,
    )
    frame = renderer.render(engine.xys, engine.dot_lum, engine.compute_opacity())
    assert frame.dtype == np.uint8
    assert frame.shape == (256, 256)


def test_balanced_luminance_keeps_frame_mean_near_background():
    bg = 0.6
    engine = build(
        BrownianRDKEngine,
        n_dots=50,
        speed_px_per_s=0,
        field_diam_px=120,
        luminance_mode="balanced",
        background_lum=bg,
    )
    renderer = FrameRenderer(
        width_px=128,
        height_px=128,
        background_lum=bg,
        field_center_xy_px=(0.0, 0.0),
        dot_size_px=2,
    )
    frame = renderer.render(engine.xys, engine.dot_lum, engine.compute_opacity())
    # Not exact: dot masks quantise pixel coverage. This is a sanity bound.
    assert abs(frame.mean() / 255.0 - bg) < 0.15


def test_renderer_flips_y_for_pixel_coordinates():
    renderer = FrameRenderer(
        width_px=64,
        height_px=64,
        background_lum=0.0,
        field_center_xy_px=(0.0, 0.0),
        dot_size_px=2,
    )
    # A dot above the field centre must land in the upper half of the image.
    frame = renderer.render(
        np.array([[0.0, 20.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    lit_rows = np.flatnonzero(frame.max(axis=1) > 0)
    assert lit_rows.max() < 32


# --- the modifications compose onto either base ------------------------------


@pytest.mark.parametrize("cls", NON_OVERLAP_ENGINES)
def test_both_variants_respect_min_sep_initially_and_after_steps(cls):
    engine = build(cls, min_sep_px=4.0, n_dots=150, field_diam_px=200)
    assert engine.min_sep_px == pytest.approx(4.0), "this density should not need relaxing"
    assert min_pairwise_distance(engine.xys) >= 3.95
    for _ in range(5):
        engine.step()
        assert min_pairwise_distance(engine.xys) >= 3.95


@pytest.mark.parametrize("cls", NON_OVERLAP_ENGINES)
def test_both_variants_get_the_default_envelope(cls):
    engine = build(cls, min_sep_px=4.0, field_diam_px=200)
    assert engine.gauss_sigma == pytest.approx(50.0)
    assert engine.compute_opacity().min() < 1.0


def test_mn_variant_inherits_the_interleaving():
    engine = build(
        GaussianNonOverlapMNEngine,
        min_sep_px=4.0,
        n_dots=300,
        coherence=1.0,
        field_diam_px=WIDE_FIELD,
    )
    assert isinstance(engine, MovshonNewsomeRDKEngine)
    assert engine.n_sequences == 3
    # Each dot still updates exactly every third frame despite the extra machinery.
    updated, never_near_rim = [], np.ones(engine.n, dtype=bool)
    for _ in range(6):
        moved, interior = step_and_measure(engine)
        updated.append(moved > 1e-6)
        never_near_rim &= interior
    assert set(np.unique(np.array(updated)[:, never_near_rim].sum(axis=0))) == {2}


def test_brownian_variant_inherits_exact_count_coherence():
    engine = build(GaussianNonOverlapRDKEngine, min_sep_px=4.0, n_dots=400, coherence=0.25)
    assert engine.is_signal.sum() == 100


def test_replanting_spares_signal_dots():
    """A crowded field forces replanting; coherent displacements should survive it."""
    engine = build(
        GaussianNonOverlapMNEngine,
        min_sep_px=4.0,
        n_dots=700,
        coherence=1.0,
        speed_px_per_s=120,
        field_diam_px=300,
        dot_life_frames=0,
    )
    coherent_step = 120 / 60 * engine.n_sequences
    lost = total = 0
    for _ in range(20):
        before = engine.xys.copy()
        engine.step()
        moved = np.linalg.norm(engine.xys - before, axis=1)
        interior = np.linalg.norm(before, axis=1) < engine.radius - 20
        signal = engine.is_signal & interior
        total += int(signal.sum())
        lost += int((signal & ~np.isclose(moved, coherent_step, atol=1e-2)).sum())
    assert total > 0
    # Noise dots are sacrificed first, so signal steps come through intact.
    assert lost / total < 0.01, f"{lost}/{total} signal steps were displaced"


def test_replanting_still_resolves_every_violation():
    # Even when sparing signal dots conflicts with the constraint, the final pass wins.
    engine = build(
        GaussianNonOverlapMNEngine,
        min_sep_px=4.0,
        n_dots=700,
        coherence=1.0,
        field_diam_px=300,
        dot_life_frames=0,
    )
    for _ in range(10):
        engine.step()
        assert min_pairwise_distance(engine.xys) >= engine.min_sep_px - 1e-3
