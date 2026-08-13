# RDK Algorithms

A random dot kinematogram (RDK) shows a field of dots in which some fraction — the
**coherence** — carries a common motion signal while the rest carry noise. Almost every
RDK ever published agrees on that sentence and disagrees on everything after it. The
disagreements matter: Pilly & Seitz (2009) showed that coherence thresholds measured with
different algorithms differ several-fold for the same observer, so "50% coherence" is
meaningful only once you name the algorithm that produced it.

This document describes the four canonical algorithms implemented here, the two orthogonal
rules that cut across them, and the one non-canonical variant this repository adds.

## The two orthogonal rules

Before the algorithms, two choices that apply to most of them. Scase, Braddick & Raymond
(1996) isolated these and showed they change the stimulus independently of the noise rule.

**Signal rule — which dots carry the signal.**

- `same`: one fixed subset of dots carries the signal for the whole trial. A coherent dot
  traces a long, continuous trajectory, so an observer can in principle track a single dot.
- `different`: the signal subset is redrawn every frame. No dot has a trajectory longer
  than one frame-pair, so the percept must come from pooled local motion rather than
  tracking. This is the more common choice and it is our default.

**Dot lifetime.** Each dot lives a fixed number of frames, then dies and respawns at a
uniformly random position in the aperture. Limited lifetime (Morgan & Ward, 1980) prevents
observers from tracking any individual dot for long and keeps dot density uniform. Initial
lifetimes are staggered uniformly across `1..dot_life_frames` so deaths are spread evenly
over time rather than arriving in synchronized cohorts. Setting the lifetime to a value
greater than the trial length effectively disables it.

Two conventions are shared by every method here. Dots are sampled uniformly **over the
area** of the circular aperture (`r = R·√u`, not `r = R·u`, which would over-sample the
centre), and dots that leave the aperture respawn at a new uniformly random position inside
it rather than wrapping to the opposite edge. Wrapping preserves coherent trajectories
across the boundary and is used in some implementations, notably the Shadlen-lab
`dots.m` lineage; uniform respawn is what Britten et al. (1992) and PsychoPy do, and using
it everywhere keeps the five methods comparable to each other.

## The canonical algorithms

### 1. Movshon–Newsome (`movshon_newsome`)

The algorithm behind the classic MT physiology and perceptual-decision literature —
Newsome & Paré (1988), Britten et al. (1992), and the accumulation-to-bound work that grew
out of it (Shadlen & Newsome, 2001).

Dots are divided into three interleaved sequences. On each frame only one sequence is
redrawn, so any given dot updates every third frame and its coherent displacement is
`speed × 3 / fps`. When a sequence is redrawn, each of its dots independently either
carries the signal with probability equal to the coherence — displaced in the signal
direction — or is relocated to a uniformly random position in the aperture. Coherence is
therefore a **per-dot probability**, not an exact count, and signal identity is re-drawn at
every update, which makes this inherently a `different`-rule stimulus.

The three-way interleaving is the distinctive part and the part most often dropped in
reimplementations. It sets the effective spatial and temporal displacement of the motion
signal, and it means a "noise" dot is uncorrelated with its own past as well as with the
signal. Dot lifetime is not part of the original algorithm — the probabilistic
re-selection already limits trajectories — so lifetime is disabled by default here.

### 2. Brownian / random-walk noise (`brownian`)

Scase et al.'s "random walk" noise; the `BM` algorithm of Pilly & Seitz (2009).

Signal dots step in the common direction. Each noise dot steps the **same distance** but in
a fresh uniformly random direction, redrawn every frame — a random walk at signal speed.
Because signal and noise dots are matched in speed, local speed carries no information
about which dots are signal, and the noise is spatially and temporally smooth in a way
white noise is not.

This is the closest match to PsychoPy's `DotStim(noiseDots='walk')` and is the base our own
method builds on.

### 3. White noise / random position (`white_noise`)

Scase et al.'s "random position" noise; the `WN` algorithm of Pilly & Seitz (2009).

Signal dots step in the common direction. Each noise dot is relocated to a uniformly random
position anywhere in the aperture on every frame, so noise dots have no motion signal at
all — they are dynamic visual noise. This produces the largest apparent displacement for
noise dots and, by breaking any local speed match, tends to yield the lowest coherence
thresholds of the three noise rules.

Equivalent to PsychoPy's `DotStim(noiseDots='position')`.

### 4. Random direction (`random_direction`)

Scase et al.'s "random direction" noise; PsychoPy's `noiseDots='direction'`.

Signal dots step in the common direction. Each noise dot is assigned a random direction
**once**, at birth, and keeps it for its entire lifetime, moving at signal speed. Noise dots
therefore travel in straight lines like signal dots, and the display looks like a
transparent superposition of many directions rather than a signal buried in noise. Dot
lifetime matters more here than anywhere else: it is the only thing that ever re-randomizes
a noise dot's direction, so a long lifetime makes the noise nearly static in direction.

## Our variant: Gaussian non-overlapping

Not a new algorithm. It is a canonical method, **unchanged**, plus two placement and
rendering modifications. Every motion rule of the base is inherited exactly, so a run at
coherence *c* is directly comparable to that base at coherence *c*.

The modifications live in a mixin that composes onto any engine, so the repository ships
two variants:

- **`gaussian_nonoverlap`** (the default) is built on **Movshon–Newsome**, the algorithm
  the MT and perceptual-decision literature is written in.
- **`gaussian_nonoverlap_brownian`** is built on **Brownian**. Its coherence is an exact
  count rather than a per-frame Bernoulli draw, and because its dots move a pixel or two
  per frame instead of teleporting, the separation constraint almost never fires. Prefer it
  when coherence must be exact.

**Modification 1 — Gaussian envelope.** Per-dot opacity falls off with distance from the
field centre, `alpha = exp(−r² / 2σ²)`, with σ defaulting to a quarter of the field
diameter. The canonical methods use a hard-edged circular aperture, which introduces a
sharp luminance boundary and gives dots an abrupt onset and offset as they cross it. The
Gaussian window removes that edge, so dots fade in and out smoothly and the stimulus
carries no aperture contour. Setting σ to zero or leaving it unset disables the envelope
and recovers a hard aperture.

**Modification 2 — minimum separation.** Dot centres are kept at least `min_sep_px` apart
(recommended ≈ 1.1 × dot size) by rejection sampling at spawn and respawn, with violations
after a motion step resolved by replanting the offending dots. Canonical RDKs let dots
overlap freely, which means the number of *visible* dots fluctuates with chance clustering
and effective density is lower than the nominal dot count. Enforcing a separation gives the
field blue-noise spatial statistics — the same argument Yellott (1983) made for retinal
photoreceptor sampling — so density is even, no dot is ever hidden behind another, and
apparent contrast does not vary with local clustering.

**Known cost of modification 2.** The constraint is enforced by teleportation, not by
collision physics. When a moving dot ends a step too close to a neighbour, one of the two is
replanted elsewhere in the aperture — and replanting a *signal* dot mid-trajectory perturbs
the motion. On the Movshon–Newsome base this matters more than it looks, because one
coherent displacement there stands in for three frames of signal rather than one.

Violation resolution therefore chooses its victim rather than moving every dot in conflict:
noise dots are sacrificed first, and a signal dot is only replanted when its conflict cannot
be cleared any other way. A final pass moves everything still in conflict, which always
resolves, so the separation guarantee holds regardless. Measured on a deliberately crowded
field (700 dots, 4 px separation, 300 px aperture), naively replanting every violator
corrupts roughly 5% of coherent displacements; preferring noise dots brings that to zero,
and the repository tests hold it under 1%.

Use `gaussian_nonoverlap_brownian` if you want the modifications with the least
interference of all: its dots step rather than teleport, so conflicts are rarer to begin
with, and its coherence is an exact dot count.

## What happened to luminance balancing

Earlier versions of this repository treated luminance balancing — assigning each dot one of
two luminances straddling the background, in the exact proportion that makes mean field
luminance equal background luminance — as part of the house method. It is no longer part of
any method's definition, because none of the canonical algorithms specify it and bundling it
made our variant non-comparable to its own base.

It survives as a **render-level option available to every method equally**
(`luminance_mode`), so a balanced Movshon–Newsome stimulus is as easy to produce as a
balanced Brownian one:

- `uniform` (default): all dots share one luminance derived from `dot_contrast`, the
  classic bright-dots-on-a-darker-background arrangement.
- `balanced`: the two-level scheme described above. Useful when a luminance-flicker artifact
  would contaminate a measurement — EEG and fMRI work in particular — since the field's mean
  luminance stays at background and only the motion is detectable.

`dot_contrast` is an **absolute luminance span**, not Michelson contrast: the dot
luminances are `background ± dot_contrast/2`, clipped to the displayable range.

## Choosing between them

| If you want | Use |
|---|---|
| Comparability with the MT / decision-making literature | `movshon_newsome` |
| Speed-matched noise, the common psychophysics default | `brownian` |
| Maximum noise displacement, lowest thresholds | `white_noise` |
| Transparent-motion appearance, direction-defined noise | `random_direction` |
| Even spacing and a soft aperture on the classic paradigm | `gaussian_nonoverlap` |
| The same, with exact coherence and minimal interference | `gaussian_nonoverlap_brownian` |

Coherence values are **not** interchangeable across rows of that table, with one exception:
each of our variants is directly comparable to the canonical method it is built on, because
it inherits that method's motion untouched.

## References

- Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992). The analysis of
  visual motion: a comparison of neuronal and psychophysical performance. *Journal of
  Neuroscience*, 12(12), 4745–4765.
- Morgan, M. J., & Ward, R. (1980). Conditions for motion flow in dynamic visual noise.
  *Vision Research*, 20(5), 431–435.
- Newsome, W. T., & Paré, E. B. (1988). A selective impairment of motion perception following
  lesions of the middle temporal visual area (MT). *Journal of Neuroscience*, 8(6),
  2201–2211.
- Peirce, J. W. (2007). PsychoPy — psychophysics software in Python. *Journal of Neuroscience
  Methods*, 162(1–2), 8–13.
- Pilly, P. K., & Seitz, A. R. (2009). What a difference a parameter makes: a psychophysical
  comparison of random dot motion algorithms. *Vision Research*, 49(13), 1599–1612.
- Scase, M. O., Braddick, O. J., & Raymond, J. E. (1996). What is noise for the motion
  system? *Vision Research*, 36(16), 2579–2586.
- Shadlen, M. N., & Newsome, W. T. (2001). Neural basis of a perceptual decision in the
  parietal cortex (area LIP) of the rhesus monkey. *Journal of Neurophysiology*, 86(4),
  1916–1936.
- Watamaniuk, S. N. J., & Sekuler, R. (1992). Temporal and spatial integration in dynamic
  random-dot stimuli. *Vision Research*, 32(12), 2341–2347.
- Yellott, J. I. (1983). Spectral consequences of photoreceptor sampling in the rhesus
  retina. *Science*, 221(4608), 382–385.
