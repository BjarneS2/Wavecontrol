"""
This script analyzes the dta from the single atom transport experiment.
Here we show in a heatmap K (atoms loaded) over sites S (hop length) over
which the atoms are transported given K loaded atoms and S sites to move
over. The color indicates the survival probability for a given K and S.
It essentially shows if K atoms are loaded for specific atoms that need
to be moved over S sites, how many of those atoms survive the transport
process - independent of their starting and final position. Inside the
squares of the heatmap, the survival probability is also shown as a number
in addition to the color coding, the number of runs that were observed with
the specific K and S, and also given and error value for the survival
probability (as a means to show the uncertainty/trustworthy-ness of the
numerical values).

@author: Bjarne Schümann
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import CommonThings as C
import matplotlib.pyplot as plt

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D
MIN_N = 8


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> list of built Dataset objects."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    names = (
        sorted(groups)
        if not runs
        else [k for k in sorted(groups) if any(r in k for r in runs)]
    )
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return run_dss


def _heatmap_figure(by_kd):
    """K times hop-length: heatmap of survival, each cell annotated with p, n and the
    Wilson half-width."""
    cells = {kd: c for kd, c in by_kd.items() if c[1] >= MIN_N}
    ks = sorted({k for k, _ in cells})
    ds_ = sorted({d for _, d in cells})
    H = np.full((len(ks), len(ds_)), np.nan)
    for (k, d), (a_, n_) in cells.items():
        H[ks.index(k), ds_.index(d)] = a_ / n_
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(ds_)), ds_), ax.set_yticks(range(len(ks)), ks)  # type: ignore
    ax.set_xlabel("S: hop length [sites]"), ax.set_ylabel("K atoms loaded")  # type: ignore
    # ax.set_title(f"mover survival vs K and hop length (cells with n>={MIN_N})")
    for (k, d), (a_, n_) in cells.items():
        i, j = ks.index(k), ds_.index(d)
        p, lo, hi = C.wilson(  # noqa
            a_, n_
        )  # for plus minus just use high as pm maybe should do abs() compare high low and selet highest ...
        ax.text(
            j,
            i,
            f"{p:.2f}\n±{hi - p:.2f}\nn={n_}",
            ha="center",
            va="center",
            fontsize=6,
        )
    fig.colorbar(im, ax=ax, label="survival")
    fig.tight_layout()
    return fig


def plot(runs=None, save=True, images=None, name=None):
    """K x hop-length survival heatmap, pooled over the selected runs. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    surv = C.survival_tables(None, run_dss)
    fig = _heatmap_figure(surv["by_kd"])
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "heatmap_survival_hopping")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        for ds in _load(a.runs, a.images):
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    plot(runs=a.runs, save=not a.no_save, images=a.images)
    if a.no_save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
