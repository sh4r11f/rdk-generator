# Stimulus Diagnostics

What a stimulus *is* and what its parameters *asked for* are different things, and the gap
is invisible when you watch the movie. Dots occlude each other, so fewer are visible than
you specified. Coherence is a per-frame Bernoulli draw in the Movshon–Newsome family, so
the realised signal fraction wanders around the nominal value. Dots that die of old age,
leave the aperture, or get replanted by the minimum-separation constraint deliver no
coherent motion that frame, whatever they were flagged as. A sampling bug shows up as a
radial density gradient that nobody notices by eye.

`rdk_generator.diagnostics` runs a stimulus and measures all of it. The webapp exposes it
under **Diagnostics** (on demand — it costs about a second, unlike the live preview), and
every export writes the same panel as `<job>_diagnostics.png`.

## The numbers worth reading first

`Diagnostics.summary()` returns these; the webapp shows the first six as tiles.

| Metric | What it tells you |
|---|---|
| `overlapping_pairs_per_frame` | Pairs of dots closer than one dot diameter — dots drawn on top of each other. |
| `visible_dot_fraction` | Fraction of dots not substantially hidden behind a neighbour. Your *effective* dot count. |
| `min_nn_distance_px` | The closest any two dots ever came. For our variants this must equal the minimum separation. |
| `signal_delivered` | Of the dots flagged as carrying signal, the fraction that actually made a clean coherent displacement. |
| `lost_to_replanting` | Signal dots displaced by the separation constraint. This is what our modification costs. |
| `luminance_sd` | Frame-to-frame variation in mean luminance. Non-zero means a luminance flicker rides along with the motion. |
| `lost_to_lifecycle` | Signal dots that instead aged out or left the aperture. By design, and present in every method. |
| `relocations_per_frame` | Dots teleported rather than moved. A direct readout of the noise rule. |

## The eight panels

**1. Dot spacing.** Histogram of nearest-neighbour distances, with the dot diameter marked.
Everything left of that line is an overlap. Canonical methods have a long left tail —
uniform random placement puts dots on top of each other regularly. Our variants have a hard
wall at the minimum separation and nothing to its left.

**2. Occlusion over time.** Overlapping pairs and mostly-hidden dots per frame. Tells you
whether occlusion is steady or comes in bursts, and confirms the effective dot count is
stable rather than fluctuating with chance clustering.

**3. Step size.** Histogram of per-frame displacements, with relocations excluded and the
expected coherent step marked. This is the clearest fingerprint of the noise rule.
Speed-matched noise (Brownian) puts *everything* on the line. White noise puts only the
signal dots there. Movshon–Newsome puts its updating third at three frames' worth of travel.

**4. Step direction.** Polar histogram of displacement directions. A spike at the signal
direction sitting on a pedestal of noise. The pedestal is flat for Brownian, lumpy for
random-direction (each dot keeps one heading), and nearly absent for white noise, whose
noise dots teleport rather than move.

**5. Radial density.** Dots per unit area against distance from the field centre, in
equal-area rings. **This should be flat.** A slope means the placement sampler is wrong —
the classic bug is `r = R·u` instead of `r = R·√u`, which over-samples the centre. The
dashed opacity-weighted line falls off for our variants and tracks the solid line for the
canonical four, which is the Gaussian envelope made visible.

**6. Relocations per frame.** How much of the field is teleporting rather than moving.
White noise relocates most of its noise dots every frame; Brownian relocates only what ages
out or exits. A spiky trace means synchronised death cohorts, which staggered initial
lifetimes are supposed to prevent.

**7. Coherence delivery.** Three traces against the nominal coherence: how many dots were
flagged as signal, how many actually delivered a coherent displacement, and how many were
lost specifically to replanting. The gap between the first two is normal — a dot that aged
out or left the aperture contributed no coherent motion — and the panel subtitle reports how
much of it is that. The *replanting* trace is the one that should sit at zero; if it does
not, the field is too crowded for the requested separation.

**8. Frame luminance.** Mean luminance of each rendered frame against the background. With
`luminance_mode="balanced"` this should sit flat on the background line: mean luminance
carries no information, so only motion does. With `uniform` it sits above the background and
that is expected — just be aware the dots add a luminance signal that co-varies with dot
count.

## How we know the measurements are right

Each metric is checked against a value derived without reference to the diagnostics code,
and `tests/test_diagnostics.py` pins the results.

- **Overlap counting.** For *n* uniform points in a disc of radius *R*, the expected number
  of pairs closer than *s* is `C(n,2)·(s/R)²`. Measured counts match to within 2% across
  dot counts, dot sizes and field sizes.
- **Radial density.** Integrating density × ring area over all rings returns the dot count
  exactly. The sampler on its own is flat to the Poisson noise floor.
- **Step size and direction.** Every measured step equals the algorithm's expected step for
  all eight methods. At coherence 1 every angle is the requested direction exactly; at
  coherence 0 the mean resultant length is under 0.01, i.e. uniform.
- **Relocations.** Teleports are read from the engine's own record rather than inferred
  from displacement size. Inferring them misfiles every respawn that lands near its old
  position — 1.7% of them in a small aperture — and contaminates the step histogram.
- **Coherence delivery.** Against a resolver that replants every dot in conflict the
  replanting trace reads 0.71, so the zero our variants report is a measured zero rather
  than a blind metric.

**One finding worth knowing.** The radial density profile is flat across the interior but
dips 6–15% in the outermost ring. That is not a measurement artefact and not a bug: dots
crossing the rim respawn uniformly across the *whole* aperture, so the edge annulus is
continuously drained. It is a real consequence of the uniform-respawn convention, and
wraparound would not show it.

## Using it

```python
from rdk_generator import RDKParams, RenderParams
from rdk_generator.diagnostics import compute_diagnostics, write_diagnostics_png

diag = compute_diagnostics(RDKParams(method="gaussian_nonoverlap"), RenderParams(seed=1))
print(diag.summary()["visible_dot_fraction"])

write_diagnostics_png(
    "out/diagnostics.png", rdk=RDKParams(method="brownian"), render=RenderParams(seed=1)
)
```

Measurement is `O(n_dots²)` per frame and renders every frame it measures, so it runs on a
bounded window — 90 frames by default, reported in the figure when the clip is longer. Pass
`max_frames=None` to measure everything.

## A worked comparison

The same parameters (150 dots, 300 px aperture, 3 px dots) measured two ways:

| | Ours (MN) | White noise |
|---|---|---|
| overlapping pairs / frame | 0.0 | 17.8 |
| dots visible | 100% | 97.0% |
| closest pair ever | 3.31 px | 0.19 px |

That is the entire argument for the non-overlap modification, in three rows.

The same tooling also settled how violations should be resolved. Replanting *every* dot in
conflict destroys about 71% of coherent displacements on a deliberately crowded field (700
dots, 4 px separation, 160 px aperture); sacrificing noise dots first brings that to zero.
`tests/test_diagnostics.py` pins both numbers, so the metric cannot quietly go blind.
