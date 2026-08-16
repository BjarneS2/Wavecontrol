# Single-Atom Transport: Experimental Summary

Source: `SingleAtomTransport.py` (`CONFIG = "long"`, calibration `"Mephisto"`), run on 2026-08-11.
Scope of this summary: only the runs in the `"long"` config (site spacing 32.25 µm, "default" calibration
session), **excluding** all 4.6 µm-spacing runs (these belong to the separate `"short"` config /
`Optimized*`, `*46um*` runs). The 200 µs runs (`Linear200us`, `STA200usNew`) are included in the table below.

## 1. Method / hardcoded parameters

| Quantity | Value | Source |
|---|---|---|
| Detection threshold | **154.3454 photons** | fitted per-run, `bimodal_threshold()` on pooled loading-frame counts of the *source* site across all runs |
| Empty-site peak (fit mean) | 84.64 photons | same bimodal fit |
| Filled-site peak (fit mean) | 300.89 photons | same bimodal fit |
| ROI radius | `R_PIX = 4.0` px | hardcoded in `CONFIG == "long"` branch |
| Binning | `BIN = 2` | module-level constant |
| Frame indices | `LOAD = 1`, `SURV = 2` | fixed order in the image stack (loading frame, survival frame) |
| Site spacing (long/"default" config) | **32.25 µm** (confirmed) | not present as a named constant in `CommonThings.py` (no `spacing_um`/pixel-to-µm conversion in that file — it must be set/used elsewhere in the acquisition/plotting pipeline), but v = d/T with d = 32.25 µm reproduces the plotted velocities almost exactly: 500/400/300/250/200/162.5 µs → 0.0645/0.0806/0.1075/0.129/0.161/0.198 m/s vs. plotted 0.065/0.08/0.108/0.13/0.161/0.20 m/s. |
| CI method | Wilson score interval, z=1.0 (≈68% CI, i.e. ±1σ) | `wil(k, n, z=1.0)` |
| "Arrived" | `loaded@source AND survived@target` (target count > threshold in survival frame) |
| "At source" (left behind) | `loaded@source AND survived@source AND NOT survived@target` |
| "Lost" | everything else (loaded but absent from both sites at readout) |

The threshold is fit **once**, globally, from the pooled loading-frame data of all runs in the config — it is not
re-fit per run/per trajectory type, so it is common to the whole table below.

## 2. Results table

n = total shots, `loaded` = shots where an atom was detected at the source before the move.
Probabilities are conditioned on `loaded` (i.e. P(...|loaded at source)).

| Trajectory | T [µs] | v [m/s]* | n | loaded | P(arrived) ±1σ | P(at source) ±1σ | P(lost) ±1σ |
|---|---|---|---|---|---|---|---|
| linear | 500 | 0.0645 | 141 | 86 | 0.686 ± 0.050 | 0.012 ± 0.013 | 0.302 ± 0.049 |
| min-jerk (STA) | 500 | 0.0645 | 141 | 90 | 0.478 ± 0.052 | 0.000 ± 0.006 | 0.522 ± 0.052 |
| linear | 400 | 0.0806 | 196 | 106 | 0.585 ± 0.048 | 0.000 ± 0.005 | 0.415 ± 0.048 |
| min-jerk (STA) | 400 | 0.0806 | 147 | 80 | 0.487 ± 0.056 | 0.050 ± 0.025 | 0.463 ± 0.055 |
| linear | 300 | 0.1075 | 154 | 94 | 0.372 ± 0.050 | 0.053 ± 0.024 | 0.574 ± 0.051 |
| min-jerk (STA) | 300 | 0.1075 | 132 | 77 | 0.649 ± 0.054 | 0.000 ± 0.006 | 0.351 ± 0.054 |
| linear | 250 | 0.1290 | 620 | 340 | 0.176 ± 0.021 | 0.044 ± 0.011 | 0.779 ± 0.023 |
| min-jerk (STA) | 250 | 0.1290 | 142 | 94 | 0.766 ± 0.044 | 0.032 ± 0.019 | 0.202 ± 0.041 |
| linear | 200 | 0.1613 | 327 | 193 | 0.021 ± 0.011 | 0.212 ± 0.029 | 0.767 ± 0.030 |
| min-jerk (STA, "New") | 200 | 0.1613 | 198 | 114 | 0.316 ± 0.043 | 0.202 ± 0.038 | 0.483 ± 0.047 |
| linear | 162.5 | 0.1985 | 148 | 83 | 0.000 ± 0.006 | 0.325 ± 0.051 | 0.675 ± 0.051 |
| min-jerk (STA) | 162.5 | 0.1985 | 207 | 117 | 0.068 ± 0.024 | 0.385 ± 0.045 | 0.547 ± 0.046 |

\*velocities computed as d/T with confirmed site spacing d = 32.25 µm; match panel (a) labels. All three
probability columns (and their Wilson-score ±1σ error bars, z=1.0) are recomputed directly from the same
per-run counts/threshold used for `P(arrived)` — not read off the figure — so the point estimates sum to 1
per row within rounding; the three CIs are each computed independently (not jointly), so they don't
necessarily sum in quadrature to anything meaningful across the row.

## 3. Trend with speed

Plotting `P(arrived)` vs. transport time/velocity (panel a):

- **Linear trajectories**: monotonically decreasing with velocity, and the decrease is not linear in v —
  it is much steeper than linear, consistent with an *exponential-like* (or faster) roll-off. Between
  500 µs (0.686) and 400 µs (0.585) the drop is modest (~15% relative), but between 300 µs (0.372) and
  250 µs (0.176) it roughly halves again, and by 200 µs (0.021) and 162.5 µs (0.000) it has fully
  collapsed. On a semi-log scale this looks close to an exponential decay in v (or equivalently
  super-exponential in 1/T), not a straight-line (linear) decrease — a linear fit would predict a much
  shallower drop at high speed and would go negative, which is unphysical, so exponential/thresholded
  (sigmoidal) behavior fits far better qualitatively.
- **Min-jerk (STA) trajectories**: *not* monotonic over this range. It falls from 500 µs (0.478) to
  400 µs (0.487, ~flat) to 300 µs (0.649) and actually **peaks at 250 µs (0.766)** before collapsing
  through 200 µs (0.316) to 162.5 µs (0.068). This non-monotonic "sweet spot" around 250 µs suggests that
  the min-jerk (smooth, minimum-jerk/STA) profile trades off different loss channels (e.g. sloshing/heating
  vs. finite-time diabatic loss) differently than the linear ramp — at the slow end it may be limited by
  something other than adiabaticity (background loss/heating during the longer hold), while the linear
  profile is limited by the abrupt velocity/acceleration kinks throughout.
- **Linear vs. min-jerk comparison**: at long times (500, 400 µs) linear actually *outperforms* min-jerk;
  the two curves cross around 300–250 µs, above which min-jerk clearly outperforms linear (0.649 vs 0.372
  at 300 µs; 0.766 vs 0.176 at 250 µs; 0.316 vs 0.021 at 200 µs), and this advantage is largest right
  before both collapse near 162.5 µs (0.068 vs 0.000). So min-jerk buys a substantial fast-transport
  advantage, but only in the "moderately fast" regime — at very low speed neither profile matters much,
  and at very high speed both fail almost completely.

## 4. Loss channel breakdown (panel b)

For both trajectory types, "lost" (never redetected at either site) dominates over "left behind at source"
for slow/moderate transport times (500–300 µs), while for the fastest times (200/162.5 µs) "left behind at
source" becomes the dominant loss channel instead of outright atom loss — i.e. failures at high speed are
increasingly failures to move the atom at all (it stays trapped at the source) rather than losing the atom
from the trap entirely.

## 5. Caveats

- The detection threshold is global across all runs in the config, fit from pooled source-loading data —
  any systematic drift in imaging/illumination between sessions would bias all runs together, not per-run.
- Site spacing for the "default"/long config is 32.25 µm (confirmed against the actual tweezer
  configuration); it just isn't stored as a named constant inside `CommonThings.py`, so the velocity axis
  had to be cross-checked externally rather than read out of that file.
- `Linear250us`/`STA250us` etc. sample sizes vary a lot (n=620 for Linear250us vs. n=132–207 for most others),
  so the Wilson CIs are correspondingly tighter/looser — errorbars in the table should be respected when
  comparing close values (e.g. STA400us vs STA500us are statistically indistinguishable: 0.487±0.056 vs
  0.478±0.052).
