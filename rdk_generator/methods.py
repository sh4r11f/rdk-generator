"""Registry of RDK algorithms.

One place that knows, for every method: which engine implements it, which parameters it
takes, how it generates its frames, and what to cite. Both `export` and the webapp read
from here, so adding a method means adding a `MethodSpec` and nothing else.

The prose in this module is user-facing — the webapp renders it verbatim. See
docs/methods.md for the longer treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

from .core import (
    BaseRDKEngine,
    BrownianRDKEngine,
    GaussianNonOverlapRDKEngine,
    MovshonNewsomeRDKEngine,
    RandomDirectionRDKEngine,
    WhiteNoiseRDKEngine,
)


@dataclass(frozen=True, slots=True)
class Reference:
    """A citation, with a link that resolves without hard-coding a DOI."""

    citation: str
    title: str

    @property
    def url(self) -> str:
        return "https://scholar.google.com/scholar?q=" + quote_plus(self.title)

    def to_dict(self) -> dict[str, str]:
        return {"citation": self.citation, "url": self.url}


@dataclass(frozen=True, slots=True)
class ParamField:
    """One tunable parameter, with everything the UI needs to render an input for it."""

    name: str
    label: str
    kind: str  # "int" | "float" | "bool" | "choice"
    group: str  # "motion" | "field" | "method" | "render"
    default: Any
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "default": self.default,
            "help": self.help,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": [{"value": v, "label": lab} for v, lab in self.choices],
        }


GROUP_LABELS: dict[str, str] = {
    "motion": "Motion",
    "field": "Dots & field",
    "method": "Method options",
    "render": "Rendering & export",
}

GROUP_ORDER: tuple[str, ...] = ("motion", "field", "method", "render")

_FIELD_LIST: tuple[ParamField, ...] = (
    # -- motion --
    ParamField(
        name="coherence",
        label="Coherence",
        kind="float",
        group="motion",
        default=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        help="Fraction of dots carrying the common motion signal. Not comparable across methods.",
    ),
    ParamField(
        name="direction_deg",
        label="Direction (deg)",
        kind="float",
        group="motion",
        default=0.0,
        step=1.0,
        help="PsychoPy convention: 0 is rightward, 90 is up.",
    ),
    ParamField(
        name="speed_px_per_s",
        label="Speed (px/s)",
        kind="float",
        group="motion",
        default=120.0,
        min=0.0,
        step=1.0,
        help="Applies to signal dots and, where the noise rule is speed-matched, to "
        "noise dots too.",
    ),
    # -- dots & field --
    ParamField(
        name="n_dots",
        label="Dots",
        kind="int",
        group="field",
        default=300,
        min=1,
        step=1,
        help="Total dot count in the aperture.",
    ),
    ParamField(
        name="dot_size_px",
        label="Dot size (px)",
        kind="int",
        group="field",
        default=3,
        min=1,
        step=1,
        help="Diameter of each drawn dot.",
    ),
    ParamField(
        name="field_diam_px",
        label="Field diameter (px)",
        kind="int",
        group="field",
        default=300,
        min=1,
        step=1,
        help="Diameter of the circular aperture.",
    ),
    # -- method options --
    ParamField(
        name="dot_life_frames",
        label="Dot life (frames)",
        kind="int",
        group="method",
        default=12,
        min=0,
        step=1,
        help="Frames before a dot dies and respawns at a random position. 0 disables ageing.",
    ),
    ParamField(
        name="signal_rule",
        label="Signal rule",
        kind="choice",
        group="method",
        default="different",
        choices=(
            ("different", "different — redraw the signal subset every frame"),
            ("same", "same — one fixed signal subset for the whole run"),
        ),
        help="Whether an individual dot can be tracked along a long coherent trajectory.",
    ),
    ParamField(
        name="n_sequences",
        label="Interleaved sequences",
        kind="int",
        group="method",
        default=3,
        min=1,
        max=6,
        step=1,
        help="Dots are split into this many groups; one group updates per frame. The "
        "canonical value is 3.",
    ),
    ParamField(
        name="gauss_sigma_px",
        label="Envelope sigma (px)",
        kind="float",
        group="method",
        default=0.0,
        min=0.0,
        step=1.0,
        help="Width of the Gaussian opacity envelope. 0 leaves it at the default of a "
        "quarter of the field diameter.",
    ),
    ParamField(
        name="min_sep_px",
        label="Min dot separation (px)",
        kind="float",
        group="method",
        default=0.0,
        min=0.0,
        step=0.1,
        help="Minimum centre-to-centre distance. 0 uses the recommended 1.1 x dot size.",
    ),
    # -- rendering & export --
    ParamField(
        name="width_px",
        label="Width (px)",
        kind="int",
        group="render",
        default=512,
        min=16,
        step=1,
    ),
    ParamField(
        name="height_px",
        label="Height (px)",
        kind="int",
        group="render",
        default=512,
        min=16,
        step=1,
    ),
    ParamField(
        name="fps",
        label="FPS",
        kind="int",
        group="render",
        default=60,
        min=1,
        step=1,
        help="Frame rate. Also sets the per-frame step size for a given speed.",
    ),
    ParamField(
        name="duration_s",
        label="Duration (s)",
        kind="float",
        group="render",
        default=1.0,
        min=0.1,
        step=0.1,
    ),
    ParamField(
        name="background_lum",
        label="Background luminance",
        kind="float",
        group="render",
        default=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        help="0 is black, 1 is white.",
    ),
    ParamField(
        name="dot_contrast",
        label="Dot contrast",
        kind="float",
        group="render",
        default=1.0,
        min=0.0,
        max=2.0,
        step=0.01,
        help="Absolute luminance span, not Michelson contrast: dots sit at background "
        "+/- half this. For white dots on black use background 0 with contrast 2.",
    ),
    ParamField(
        name="luminance_mode",
        label="Dot luminance",
        kind="choice",
        group="render",
        default="uniform",
        choices=(
            ("uniform", "uniform — every dot the same shade (classic)"),
            ("balanced", "balanced — dark/bright mix, mean equals background"),
        ),
        help="Balanced keeps mean field luminance at background, so no luminance "
        "flicker accompanies the motion. Available to every method.",
    ),
    ParamField(
        name="seed",
        label="Seed (optional)",
        kind="int",
        group="render",
        default="",
        step=1,
        help="Leave blank for a fresh random stimulus each time.",
    ),
)

FIELDS: dict[str, ParamField] = {f.name: f for f in _FIELD_LIST}

_MOTION_AND_FIELD: tuple[str, ...] = (
    "coherence",
    "direction_deg",
    "speed_px_per_s",
    "n_dots",
    "dot_size_px",
    "field_diam_px",
)

_RENDER: tuple[str, ...] = (
    "width_px",
    "height_px",
    "fps",
    "duration_s",
    "background_lum",
    "dot_contrast",
    "luminance_mode",
    "seed",
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Everything the app knows about one algorithm."""

    id: str
    label: str
    short: str
    engine: type[BaseRDKEngine]
    tagline: str
    noise_rule: str
    steps: tuple[str, ...]
    notes: tuple[str, ...]
    references: tuple[Reference, ...]
    method_params: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    canonical: bool = True

    @property
    def param_names(self) -> tuple[str, ...]:
        return _MOTION_AND_FIELD + self.method_params + _RENDER

    def default_for(self, name: str) -> Any:
        """Default for `name`, with this method's override applied if it has one."""
        if name in self.defaults:
            return self.defaults[name]
        return FIELDS[name].default

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "short": self.short,
            "tagline": self.tagline,
            "noise_rule": self.noise_rule,
            "steps": list(self.steps),
            "notes": list(self.notes),
            "canonical": self.canonical,
            "references": [r.to_dict() for r in self.references],
            "params": [
                {**FIELDS[name].to_dict(), "default": self.default_for(name)}
                for name in self.param_names
            ],
        }


_REF_BRITTEN = Reference(
    citation=(
        "Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992). "
        "The analysis of visual motion: a comparison of neuronal and psychophysical "
        "performance. Journal of Neuroscience, 12(12), 4745-4765."
    ),
    title="The analysis of visual motion: a comparison of neuronal and psychophysical performance",
)
_REF_NEWSOME = Reference(
    citation=(
        "Newsome, W. T., & Pare, E. B. (1988). A selective impairment of motion perception "
        "following lesions of the middle temporal visual area (MT). Journal of "
        "Neuroscience, 8(6), 2201-2211."
    ),
    title=(
        "A selective impairment of motion perception following lesions of the middle "
        "temporal visual area MT"
    ),
)
_REF_SHADLEN = Reference(
    citation=(
        "Shadlen, M. N., & Newsome, W. T. (2001). Neural basis of a perceptual decision in "
        "the parietal cortex (area LIP) of the rhesus monkey. Journal of Neurophysiology, "
        "86(4), 1916-1936."
    ),
    title=(
        "Neural basis of a perceptual decision in the parietal cortex area LIP of the rhesus monkey"
    ),
)
_REF_SCASE = Reference(
    citation=(
        "Scase, M. O., Braddick, O. J., & Raymond, J. E. (1996). What is noise for the "
        "motion system? Vision Research, 36(16), 2579-2586."
    ),
    title="What is noise for the motion system",
)
_REF_PILLY = Reference(
    citation=(
        "Pilly, P. K., & Seitz, A. R. (2009). What a difference a parameter makes: a "
        "psychophysical comparison of random dot motion algorithms. Vision Research, "
        "49(13), 1599-1612."
    ),
    title=(
        "What a difference a parameter makes: a psychophysical comparison of random dot "
        "motion algorithms"
    ),
)
_REF_MORGAN = Reference(
    citation=(
        "Morgan, M. J., & Ward, R. (1980). Conditions for motion flow in dynamic visual "
        "noise. Vision Research, 20(5), 431-435."
    ),
    title="Conditions for motion flow in dynamic visual noise",
)
_REF_WATAMANIUK = Reference(
    citation=(
        "Watamaniuk, S. N. J., & Sekuler, R. (1992). Temporal and spatial integration in "
        "dynamic random-dot stimuli. Vision Research, 32(12), 2341-2347."
    ),
    title="Temporal and spatial integration in dynamic random-dot stimuli",
)
_REF_PEIRCE = Reference(
    citation=(
        "Peirce, J. W. (2007). PsychoPy - psychophysics software in Python. Journal of "
        "Neuroscience Methods, 162(1-2), 8-13."
    ),
    title="PsychoPy psychophysics software in Python",
)
_REF_YELLOTT = Reference(
    citation=(
        "Yellott, J. I. (1983). Spectral consequences of photoreceptor sampling in the "
        "rhesus retina. Science, 221(4608), 382-385."
    ),
    title="Spectral consequences of photoreceptor sampling in the rhesus retina",
)

_SHARED_STEPS: tuple[str, ...] = (
    "Place every dot uniformly over the area of the circular aperture "
    "(r = R x sqrt(u), so the centre is not over-sampled).",
)

_LIFETIME_STEP = (
    "Age each dot by one frame; dots that reach the end of their life respawn at a new "
    "random position. Initial ages are staggered so deaths spread evenly over time."
)

_EXIT_STEP = "Respawn any dot that has left the aperture at a new random position inside it."


METHODS: dict[str, MethodSpec] = {
    "movshon_newsome": MethodSpec(
        id="movshon_newsome",
        label="Movshon-Newsome",
        short="MN",
        engine=MovshonNewsomeRDKEngine,
        tagline="The three-sequence classic behind the MT and decision-making literature.",
        noise_rule="Relocated to a random position when its sequence updates.",
        steps=_SHARED_STEPS
        + (
            "Assign each dot to one of 3 interleaved sequences.",
            "On each frame, update only the sequence whose turn it is, so a dot moves "
            "every 3rd frame and travels 3 frames' worth of distance when it does.",
            "For each updating dot, draw independently: with probability equal to the "
            "coherence it is displaced in the signal direction, otherwise it is relocated "
            "to a random position in the aperture.",
            _EXIT_STEP,
        ),
        notes=(
            "Coherence is a per-dot probability here, not an exact count, so the realised "
            "fraction varies frame to frame.",
            "Dot lifetime is not part of the original algorithm - the probabilistic "
            "re-selection already limits trajectories - so it is off by default.",
            "The three-way interleaving is the part most often dropped in reimplementations, "
            "and it is what sets the effective displacement of the motion signal.",
        ),
        references=(_REF_NEWSOME, _REF_BRITTEN, _REF_SHADLEN, _REF_PILLY),
        method_params=("n_sequences", "dot_life_frames"),
        defaults={"dot_life_frames": 0},
    ),
    "brownian": MethodSpec(
        id="brownian",
        label="Brownian (random walk)",
        short="BM",
        engine=BrownianRDKEngine,
        tagline="Speed-matched random-walk noise; the common psychophysics default.",
        noise_rule="Steps at signal speed in a fresh random direction every frame.",
        steps=_SHARED_STEPS
        + (
            _LIFETIME_STEP,
            "Choose the signal subset as an exact count of coherence x dots, redrawn each "
            "frame under the 'different' rule or fixed once under 'same'.",
            "Step signal dots along the common direction.",
            "Step every noise dot the same distance, but each in its own freshly drawn "
            "random direction.",
            _EXIT_STEP,
        ),
        notes=(
            "Signal and noise dots move at identical speeds, so local speed gives away "
            "nothing about which dots carry the signal.",
            "Matches PsychoPy's DotStim(noiseDots='walk').",
        ),
        references=(_REF_SCASE, _REF_PILLY, _REF_MORGAN, _REF_PEIRCE),
        method_params=("dot_life_frames", "signal_rule"),
    ),
    "white_noise": MethodSpec(
        id="white_noise",
        label="White noise (random position)",
        short="WN",
        engine=WhiteNoiseRDKEngine,
        tagline="Noise dots teleport anywhere each frame - dynamic visual noise.",
        noise_rule="Relocated to a random position in the aperture every frame.",
        steps=_SHARED_STEPS
        + (
            _LIFETIME_STEP,
            "Choose the signal subset as an exact count of coherence x dots, redrawn each "
            "frame under the 'different' rule or fixed once under 'same'.",
            "Step signal dots along the common direction.",
            "Relocate every noise dot to a new uniformly random position in the aperture.",
            _EXIT_STEP,
        ),
        notes=(
            "Noise dots carry no motion signal at all, which is why this typically yields "
            "the lowest coherence thresholds of the three noise rules.",
            "Matches PsychoPy's DotStim(noiseDots='position').",
        ),
        references=(_REF_SCASE, _REF_PILLY, _REF_WATAMANIUK, _REF_PEIRCE),
        method_params=("dot_life_frames", "signal_rule"),
    ),
    "random_direction": MethodSpec(
        id="random_direction",
        label="Random direction",
        short="RD",
        engine=RandomDirectionRDKEngine,
        tagline="Every noise dot keeps one heading for life - looks like transparent motion.",
        noise_rule="Moves at signal speed along a heading fixed when the dot was born.",
        steps=_SHARED_STEPS
        + (
            "Give every dot a random heading at birth.",
            _LIFETIME_STEP + " Respawning is the only thing that redraws a heading.",
            "Choose the signal subset as an exact count of coherence x dots, redrawn each "
            "frame under the 'different' rule or fixed once under 'same'.",
            "Step signal dots along the common direction and every noise dot along its own "
            "fixed heading.",
            _EXIT_STEP,
        ),
        notes=(
            "Noise dots travel in straight lines just like signal dots, so the display reads "
            "as a transparent superposition of directions rather than signal buried in noise.",
            "Dot lifetime matters more here than in any other method: it is the only thing "
            "that ever re-randomises a noise direction.",
            "Matches PsychoPy's DotStim(noiseDots='direction').",
        ),
        references=(_REF_SCASE, _REF_PILLY, _REF_PEIRCE),
        method_params=("dot_life_frames", "signal_rule"),
    ),
    "gaussian_nonoverlap": MethodSpec(
        id="gaussian_nonoverlap",
        label="Gaussian non-overlapping",
        short="Ours",
        engine=GaussianNonOverlapRDKEngine,
        tagline="The Brownian method with a soft aperture and evenly spaced dots.",
        noise_rule="Steps at signal speed in a fresh random direction every frame "
        "(inherited from Brownian).",
        canonical=False,
        steps=(
            "Place every dot by rejection sampling so no two centres are closer than the "
            "minimum separation - blue-noise placement rather than uniform.",
            _LIFETIME_STEP + " Respawns also respect the minimum separation.",
            "Run the Brownian method's motion unchanged: exact-count signal selection, "
            "signal dots along the common direction, noise dots stepping the same distance "
            "in fresh random directions.",
            _EXIT_STEP,
            "Replant any dots that ended the step too close to a neighbour.",
            "Scale each dot's opacity by a Gaussian in its distance from the field centre, "
            "alpha = exp(-r^2 / 2 sigma^2), so the aperture has no hard edge.",
        ),
        notes=(
            "Motion is identical to the Brownian method, so coherence is directly comparable "
            "between the two. The modifications are placement and rendering only.",
            "The Gaussian envelope removes the aperture contour and the abrupt onset and "
            "offset dots get as they cross a hard edge.",
            "Even spacing keeps the visible dot count stable and stops apparent contrast "
            "varying with chance clustering.",
            "Cost: separation is enforced by teleporting violators, which can displace a "
            "signal dot mid-trajectory, so effective coherence sits marginally below nominal. "
            "Prefer plain Brownian when coherence must be exact.",
        ),
        references=(_REF_SCASE, _REF_PILLY, _REF_YELLOTT, _REF_MORGAN),
        method_params=("dot_life_frames", "signal_rule", "gauss_sigma_px", "min_sep_px"),
    ),
}

DEFAULT_METHOD = "gaussian_nonoverlap"


def list_methods() -> list[MethodSpec]:
    """All methods, canonical ones first, in registry order."""
    return sorted(METHODS.values(), key=lambda m: (not m.canonical, 0))


def get_method(method_id: str | None) -> MethodSpec:
    """Look up a method, falling back to the default for unknown or missing ids."""
    if not method_id:
        return METHODS[DEFAULT_METHOD]
    return METHODS.get(str(method_id), METHODS[DEFAULT_METHOD])


def methods_payload() -> list[dict[str, Any]]:
    """JSON-serialisable registry for the webapp."""
    return [m.to_dict() for m in list_methods()]
