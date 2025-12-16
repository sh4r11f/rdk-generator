import numpy as np

from rdk_generator.core import FrameRenderer, RDKEngine


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
    r = FrameRenderer(width_px=256, height_px=256, background_lum=0.4, field_center_xy_px=(0.0, 0.0), dot_size_px=3)
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
    r = FrameRenderer(width_px=128, height_px=128, background_lum=bg, field_center_xy_px=(0.0, 0.0), dot_size_px=2)
    frame = r.render(e.xys, e.dot_lum, e.compute_opacity())
    mean = frame.mean() / 255.0
    assert abs(mean - bg) < 0.15
