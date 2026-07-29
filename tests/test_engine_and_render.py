import numpy as np

from rdk_generator.core import FrameRenderer, NonOverlappingRDKEngine, RDKEngine


def test_engine_coherence_approx():
    e = RDKEngine(
        n_dots=1000,
        dot_life_frames=10,
        coherence=0.25,
        direction_deg=0,
        speed_px_per_s=120,
        field_diam_px=200,
        gauss_sigma_px=None,
        reassign_life=True,
        fps=60,
        seed=1,
        background_lum=0.5,
        dot_contrast=1.0,
        dot_low_lum=None,
        dot_high_lum=None,
    )
    frac = e.is_signal.mean()
    assert abs(frac - 0.25) < 0.02


def test_renderer_outputs_uint8_shape():
    e = RDKEngine(
        n_dots=200,
        dot_life_frames=5,
        coherence=0.5,
        direction_deg=90,
        speed_px_per_s=60,
        field_diam_px=150,
        gauss_sigma_px=50,
        reassign_life=False,
        fps=30,
        seed=2,
        background_lum=0.4,
        dot_contrast=1.0,
        dot_low_lum=None,
        dot_high_lum=None,
    )
    r = FrameRenderer(
        width_px=256,
        height_px=256,
        background_lum=0.4,
        field_center_xy_px=(0.0, 0.0),
        dot_size_px=3,
    )
    frame = r.render(e.xys, e.dot_lum, e.compute_opacity())
    assert frame.dtype == np.uint8
    assert frame.shape == (256, 256)


def test_frame_mean_close_to_background_reasonable():
    # Not exact because dot mask affects pixel coverage; just sanity.
    bg = 0.6
    e = RDKEngine(
        n_dots=50,
        dot_life_frames=10,
        coherence=0.5,
        direction_deg=0,
        speed_px_per_s=0,
        field_diam_px=120,
        gauss_sigma_px=40,
        reassign_life=False,
        fps=60,
        seed=3,
        background_lum=bg,
        dot_contrast=1.0,
        dot_low_lum=None,
        dot_high_lum=None,
    )
    r = FrameRenderer(
        width_px=128, height_px=128, background_lum=bg, field_center_xy_px=(0.0, 0.0), dot_size_px=2
    )
    frame = r.render(e.xys, e.dot_lum, e.compute_opacity())
    mean = frame.mean() / 255.0
    assert abs(mean - bg) < 0.15


def _min_pairwise_distance(xys: np.ndarray) -> float:
    diff = xys[:, None, :] - xys[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))


def test_non_overlapping_engine_respects_min_sep_initial_and_steps():
    e = NonOverlappingRDKEngine(
        min_sep_px=4.0,
        n_dots=150,
        dot_life_frames=6,
        coherence=0.5,
        direction_deg=0,
        speed_px_per_s=120,
        field_diam_px=200,
        gauss_sigma_px=60,
        reassign_life=True,
        fps=60,
        seed=123,
        background_lum=0.5,
        dot_contrast=1.0,
        dot_low_lum=None,
        dot_high_lum=None,
    )

    # Allow a tiny numerical tolerance.
    assert _min_pairwise_distance(e.xys) >= 3.95

    for _ in range(5):
        e.step()
        assert _min_pairwise_distance(e.xys) >= 3.95
