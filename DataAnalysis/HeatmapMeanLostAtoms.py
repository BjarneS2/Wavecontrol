"""
This script analyzes the data from the single atom transport experiment.
Here we show in a heatmap K (atoms loaded) over M (atoms that need to move)
where the color indicates the mean number of atoms lost for a given K and M.
The individual squares of the heatmap will also contain the number of mean
lost atoms plus minus the standard deviation (representin error bars) as well
as the number of runs that were used to calculate those values (or rather that
were taken with the specific K and M).

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


def _heatmap_figure(loss):
    """KxM heatmap of mean atoms lost, each cell annotated with mean +- std and n."""
    cells = loss["cells"]
    ks = sorted({k for k, _ in cells})
    ms = sorted({m for _, m in cells})
    mean = np.full((len(ks), len(ms)), np.nan)
    std = np.full_like(mean, np.nan)
    cnt = np.zeros_like(mean)
    for (k, m), c in cells.items():
        i, j = ks.index(k), ms.index(m)
        mean[i, j] = (c["obs"] * np.arange(len(c["obs"]))).sum() / max(1, c["n"])
        std[i, j] = np.sqrt(c["var_obs"]) if np.isfinite(c["var_obs"]) else np.nan
        cnt[i, j] = c["n"]
    fig, ax = plt.subplots(figsize=(8, 5.8))
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(ms)), ms), ax.set_yticks(range(len(ks)), ks)  # type: ignore
    ax.set_xlabel("M atoms that had to move"), ax.set_ylabel("K atoms loaded")  # type: ignore
    # ax.set_title("mean atoms lost per (K, M) cell")
    for i in range(len(ks)):
        for j in range(len(ms)):
            if cnt[i, j]:
                txt = f"{mean[i, j]:.1f}"
                if np.isfinite(std[i, j]):
                    txt += f"±{std[i, j]:.1f}"
                txt += f"\nn={int(cnt[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, label="mean atoms lost")
    fig.tight_layout()
    return fig


def plot(runs=None, save=True, images=None, name=None):
    """KxM heatmap of mean atoms lost, pooled over the selected runs. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    loss = C.loss_distributions(run_dss)
    fig = _heatmap_figure(loss)
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "heatmap_mean_lost_atoms")
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
