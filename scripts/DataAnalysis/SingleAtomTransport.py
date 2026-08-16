"""
For these experiments, we only need essentially one figure. This file will do the analysis and produce said figure.
The figure will contain of 3 plots next to each other: left the loaded source and survived target, middle the
probability of arrival over speed (for linear and sta transport), right the probability as stacked histograms
for lost, arrived and at source atoms.

@author: Bjarne Schümann
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # else for pdf use "PDF"
import CommonThings as C
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BIN, LOAD, SURV = 2, 1, 2
CONFIG = "long"  # "short"  # or anything else to run long config version

if CONFIG == "short":
    CAL, R_PIX, CROP = "46umMephisto", 3.0, ((128, 160), (58, 92))
    RUNS = [
        ("Linear46um50us", 50, "linear"),
        ("STA46um50us", 50, "min-jerk"),
        ("Optimized50us", 50, "optimised"),
        ("Linear46um30us", 30, "linear"),
        ("Optimized30us", 30, "optimised"),
        ("OptimizedWithAmp30us", 30, "optimised + amp"),
        ("OptimizedWithAmpHalfFinalOffset30us", 30, "opt + amp, half offset"),
    ]
    SHOW = "Optimized50us"  # run to show left panel
elif CONFIG == "long":
    CAL, R_PIX, CROP = "Mephisto", 4.0, ((128, 160), (55, 155))
    RUNS = [
        ("Linear500us", 500, "linear"),
        ("STA500us", 500, "min-jerk"),
        ("Linear400us", 400, "linear"),
        ("STA400us", 400, "min-jerk"),
        ("Linear300us", 300, "linear"),
        ("STA300us", 300, "min-jerk"),
        ("Linear250us", 250, "linear"),
        ("STA250us", 250, "min-jerk"),
        ("Linear200us", 200, "linear"),
        ("STA200usNew", 200, "min-jerk"),
        ("Linear162_5us", 162.5, "linear"),
        ("STA162_5us", 162.5, "min-jerk"),
    ]
    SHOW = "STA250us"  # run to show left panel
else:
    print("Either pick long or short for CONFIG...")


def wil(k, n, z=1.0):
    """Wilson score interval for a binomial proportion.
    Returns the proportion and the half-width of the confidence interval.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    dd = 1 + z * z / n
    _ = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return p, h


def plot(save=True):
    """Builds the 3-panel transfer figure for the module-level CONFIG. Returns the Figure."""
    images = str(C.TRANSPORT_IMAGES)
    cal = C.load_stack(images, f"tweezerLoad1x2-{CAL}")  # get calibration images
    ref = cal[:, LOAD].std(0) + cal[:, SURV].std(0)  # get reference std for each pixel
    loc = C.locate_sites(ref)  # locate the two sites in the reference image
    masks = C.site_masks(ref.shape, loc, R_PIX)
    ini, fin = 0, 1  # indices for the two sites
    raw = {
        n: C.load_stack(images, f"tweezerLoad1x2-{n}") for n, _, _ in RUNS
    }  # stack all images for each run
    cts = {
        n: (
            C.roi_mask_counts(C.to_photons(raw[n][:, LOAD], BIN), masks),
            C.roi_mask_counts(C.to_photons(raw[n][:, SURV], BIN), masks),
        )
        for n in raw
    }
    THRESHOLD, fitp = C.bimodal_threshold(np.concatenate([cts[n][0][ini] for n in raw]))
    print(
        f"Threshold is {THRESHOLD:.4f} photons (empty {min(fitp[1], fitp[2]):.4f}, filled {max(fitp[1], fitp[2]):.4f})"
    )

    rows = []
    for name, T, kind in RUNS:
        L, S = cts[name]
        lo = L[ini] > THRESHOLD
        af, ai = S[fin] > THRESHOLD, S[ini] > THRESHOLD
        n, nl = L.shape[1], int(lo.sum())
        arr = int((lo & af).sum())
        src = int((lo & ai & ~af).sum())
        pa, ea = wil(arr, nl)
        rows.append(
            dict(  # noqa: C408
                run=name,
                T=T,
                kind=kind,
                n=n,
                nl=nl,
                p_arr=pa,
                e_arr=ea,
                p_src=src / nl,
                p_lost=(nl - arr - src) / nl,
            )
        )
        print(
            f"{name:<38}{T:>4}us  n={n:4d} loaded={nl:4d}  arrived {pa:.3f}+-{ea:.3f}"
        )

    """ Now onto plotting the results. """

    (x0, x1), (y0, y1) = CROP
    L, S = cts[SHOW]
    sel = (L[ini] > THRESHOLD) & (L[fin] <= THRESHOLD)  # source loaded, target empty
    ok = sel & (S[fin] > THRESHOLD)  # ... and delivered
    before = C.to_photons(raw[SHOW][sel, LOAD], BIN).mean(0)[y0:y1, x0:x1]
    after = C.to_photons(raw[SHOW][ok, SURV], BIN).mean(0)[y0:y1, x0:x1]
    vmax = max(before.max(), after.max())

    fig = plt.figure(figsize=(11.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.6], wspace=0.3)
    for k, (img, ttl) in enumerate(
        [
            (before, f"loading  (n={sel.sum()})"),
            (after, f"survival, delivered  (n={ok.sum()})"),
        ]
    ):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(
            img,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            extent=(x0, x1, y1, y0),
            interpolation="nearest",
        )
        for i, s in enumerate(loc):
            ax.add_patch(Circle(s, R_PIX, fill=False, ec="w", lw=1.2, ls="--"))  # type: ignore
            ax.text(
                s[0] + R_PIX + 1,
                s[1],
                "source" if i == ini else "target",
                color="w",
                va="center",
                fontsize=8,
            )
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("x [px]")
        if k == 0:
            ax.set_ylabel("y [px]")

    ax = fig.add_subplot(gs[0, 2])
    idx = np.arange(len(rows))
    a = [r["p_arr"] for r in rows]
    s = [r["p_src"] for r in rows]
    l = [r["p_lost"] for r in rows]
    col = {"linear": "tab:blue", "min-jerk": "tab:red"}
    ax.bar(
        idx,
        a,
        0.68,
        color=[col.get(r["kind"], "tab:green") for r in rows],
        label="arrived",
    )
    ax.bar(idx, s, 0.68, bottom=a, color="orange", label="at source")
    ax.bar(idx, l, 0.68, bottom=np.add(a, s), color="grey", label="lost")
    ax.errorbar(
        idx, a, yerr=[r["e_arr"] for r in rows], fmt="none", ecolor="k", capsize=3
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{r['kind']}\n{r['T']}" + r"$\,\mu$s" for r in rows], fontsize=7
    )
    ax.set_ylabel("fraction of loaded atoms")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    if save:
        C.save_figure(fig, C.CAT_TRANSPORT, CONFIG, "transfer", dpi=300)
    return fig


if __name__ == "__main__":
    sys.exit(0 if plot() else 1)
