import json

import pytest

from rdk_generator import RDKParams, RenderParams, build_engine
from rdk_generator.methods import (
    DEFAULT_METHOD,
    FIELDS,
    GROUP_ORDER,
    METHODS,
    get_method,
    list_methods,
    methods_payload,
)

ALL_IDS = sorted(METHODS)


def test_registry_covers_the_documented_methods():
    assert set(METHODS) == {
        "movshon_newsome",
        "brownian",
        "white_noise",
        "random_direction",
        "gaussian_nonoverlap",
    }
    assert DEFAULT_METHOD in METHODS


@pytest.mark.parametrize("method_id", ALL_IDS)
def test_every_declared_param_is_a_known_field(method_id):
    spec = METHODS[method_id]
    for name in spec.param_names:
        assert name in FIELDS, f"{method_id} declares unknown param {name}"
    assert len(set(spec.param_names)) == len(spec.param_names), "duplicate param"


@pytest.mark.parametrize("method_id", ALL_IDS)
def test_every_param_belongs_to_a_known_group(method_id):
    for name in METHODS[method_id].param_names:
        assert FIELDS[name].group in GROUP_ORDER


@pytest.mark.parametrize("method_id", ALL_IDS)
def test_defaults_sit_inside_declared_bounds(method_id):
    spec = METHODS[method_id]
    for name in spec.param_names:
        spec_field = FIELDS[name]
        value = spec.default_for(name)
        if spec_field.kind in ("int", "float") and value != "":
            if spec_field.min is not None:
                assert value >= spec_field.min, f"{method_id}.{name} below min"
            if spec_field.max is not None:
                assert value <= spec_field.max, f"{method_id}.{name} above max"
        if spec_field.kind == "choice":
            assert value in {v for v, _ in spec_field.choices}


@pytest.mark.parametrize("method_id", ALL_IDS)
def test_every_method_documents_itself(method_id):
    spec = METHODS[method_id]
    assert spec.tagline and spec.noise_rule
    assert spec.steps, "needs a 'how it is generated' description"
    assert spec.references, "needs at least one citation"
    for ref in spec.references:
        assert ref.citation.strip()
        assert ref.url.startswith("https://")


@pytest.mark.parametrize("method_id", ALL_IDS)
def test_method_builds_an_engine_from_default_params(method_id):
    engine = build_engine(RDKParams(method=method_id), RenderParams(seed=0))
    assert engine.method_id == method_id
    engine.step()


def test_movshon_newsome_defaults_to_no_dot_ageing():
    # Canonical MN has no lifetime; the dataclass default must not leak in.
    engine = build_engine(RDKParams(method="movshon_newsome"), RenderParams(seed=0))
    assert engine.dot_life == 0


def test_other_methods_keep_the_standard_dot_life():
    engine = build_engine(RDKParams(method="brownian"), RenderParams(seed=0))
    assert engine.dot_life == 12


def test_explicit_dot_life_overrides_the_method_default():
    engine = build_engine(
        RDKParams(method="movshon_newsome", dot_life_frames=7), RenderParams(seed=0)
    )
    assert engine.dot_life == 7


def test_min_sep_defaults_to_scaled_dot_size():
    engine = build_engine(
        RDKParams(method="gaussian_nonoverlap", dot_size_px=6), RenderParams(seed=0)
    )
    assert engine.min_sep_px == pytest.approx(6.6)


def test_unknown_method_falls_back_to_the_default():
    assert get_method("not-a-method").id == DEFAULT_METHOD
    assert get_method(None).id == DEFAULT_METHOD
    assert get_method("").id == DEFAULT_METHOD


def test_only_our_variant_is_marked_non_canonical():
    non_canonical = [m.id for m in METHODS.values() if not m.canonical]
    assert non_canonical == ["gaussian_nonoverlap"]


def test_list_methods_puts_canonical_ones_first():
    ordered = list_methods()
    assert len(ordered) == len(METHODS)
    assert ordered[-1].id == "gaussian_nonoverlap"


def test_payload_is_json_serialisable_and_carries_merged_defaults():
    payload = methods_payload()
    round_tripped = json.loads(json.dumps(payload))
    assert [m["id"] for m in round_tripped] == [m.id for m in list_methods()]

    mn = next(m for m in round_tripped if m["id"] == "movshon_newsome")
    life = next(p for p in mn["params"] if p["name"] == "dot_life_frames")
    assert life["default"] == 0, "payload must carry the per-method override"


def test_gaussian_variant_exposes_its_two_modifications():
    params = set(METHODS["gaussian_nonoverlap"].param_names)
    assert {"gauss_sigma_px", "min_sep_px"} <= params


@pytest.mark.parametrize("method_id", [m for m in ALL_IDS if m != "gaussian_nonoverlap"])
def test_canonical_methods_do_not_expose_our_modifications(method_id):
    params = set(METHODS[method_id].param_names)
    assert not ({"gauss_sigma_px", "min_sep_px"} & params)


def test_movshon_newsome_has_no_signal_rule_choice():
    # Its probabilistic re-selection is inherently a "different" rule.
    assert "signal_rule" not in METHODS["movshon_newsome"].param_names
