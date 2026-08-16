"""
This script analyzes the data from the single atom transport experiment.
This plot shows the target blocks size (on x) and the actual continuous
line of atoms achieved / probability of achieving that line.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D
KBANDS = [(1, 3), (4, 6), (7, 9)]


def fig_success_vs_k(per_ds, pooled, kbands=(), title=""):
    """P(defect-free block | K), per run | pooled, plus any requested K bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 2 if kbands else 1
    fig, axes = plt.subplots(1, ncol, figsize=(8.0 * ncol, 4.4), squeeze=False)
    ax = axes[0][0]
    cmap = matplotlib.colormaps["turbo"]
    colors = cmap(np.linspace(0, 1, max(len(per_ds), 1)))
    for (name, tab), color in zip(per_ds.items(), colors):
        ks = [k for k in sorted(tab) if tab[k][1] > 0]
        if not ks:
            continue
        ax.plot(
            ks,
            [tab[k][0] / tab[k][1] for k in ks],
            marker="s",
            ms=4,
            alpha=0.7,
            lw=1,
            color=color,
            label=name.replace("tweezerLoad1x11-", ""),
        )
    ks = [k for k in sorted(pooled) if pooled[k][1] > 0]
    y, el, eh = [], [], []
    for k in ks:
        pp, lo, hi = A.wilson(*pooled[k])
        y.append(pp)
        el.append(pp - lo)
        eh.append(hi - pp)
    ax.errorbar(
        ks,
        y,
        yerr=[el, eh],
        marker="o",
        color="k",
        lw=2.4,
        capsize=4,
        label="pooled" if len(per_ds) > 1 else None,
    )
    for k, yy in zip(ks, y):
        ax.annotate(
            f"{pooled[k][1]}",
            (k, yy),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color="0.1",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )
    ax.set_xlabel("K atoms loaded")
    ax.set_ylabel("P(target block defect-free in frame 2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(ks)
    # ax.set_title("per K" + (" (n above each point)" if ks else ""), fontsize=10)
    ax.legend(fontsize=6, ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3)

    if kbands:
        ax2 = axes[0][1]
        names = list(per_ds)
        xs = np.arange(len(names))
        for j, (lo_, hi_) in enumerate(kbands):
            p, e = [], []
            for nm in names:
                tab = per_ds[nm]
                a_ = sum(tab[k][0] for k in tab if lo_ <= k <= hi_)
                n_ = sum(tab[k][1] for k in tab if lo_ <= k <= hi_)
                pp, l_, h_ = A.wilson(a_, n_)  # noqa
                p.append(pp if n_ else np.nan), e.append(max(h_ - pp, 0) if n_ else 0)  # type: ignore
            off = (j - (len(kbands) - 1) / 2) * 0.22
            ax2.errorbar(
                np.array(p),
                xs + off,
                xerr=e,
                fmt="o",
                ms=5,
                capsize=3,
                lw=1,
                label=f"K in [{lo_}, {hi_}]",
            )
        ax2.set_yticks(xs)
        ax2.set_yticklabels(
            [n.replace("tweezerLoad1x11-", "") for n in names], fontsize=8
        )
        ax2.set_xlabel("P(defect-free block)")
        ax2.set_xlim(-0.02, 1.02)
        # ax2.set_title("per run, pooled over each K band", fontsize=10)
        ax2.legend(fontsize=8, framealpha=0.95)
        ax2.grid(axis="x", alpha=0.3)
    # fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
    return fig


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> list of built Dataset objects (no calibration needed here)."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    names = (
        sorted(groups)
        if not runs
        else [k for k in sorted(groups) if any(r in k for r in runs)]
    )
    names = [n for n in names if "RampOut" not in n]
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return run_dss


def plot(runs=None, save=True, kmax=10, images=None, name=None):
    """P(defect-free target block | K), per run and pooled. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    per_ds, pooled = C.success_vs_k(run_dss, kmax=kmax)
    kbands = KBANDS if runs else ()
    fig = fig_success_vs_k(per_ds, pooled, kbands=kbands)
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "expectations_of_delivery")
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
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
