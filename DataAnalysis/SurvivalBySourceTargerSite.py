"""
This script analyzes the data from the single atom transport experiment.
Figure should show for the 80us run in terms of a deeper analysis the survival
of the atoms given their source site, two curves showing if they are stationary
or moving and then the overall probability. (y axis probabilities, x axis sites).
Then the second plot of that figure (on the right) should show the dependence on
the target site, again displayed in the same way. (of course stationary are the
same for both, so one could in principle combine those into one plot/figure).

I might do both and make the user be able to decide themselves what to do - stationary
one curve and then 2 curves of starting and final sites in e.g. red and blue for a nice
contrast, while stationary is a dark grey or black in the background. Of course all
with error bars.

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
DEFAULT_RUNS = [
    "bestsort-80us",
    "80usAGAIN",
]  # the "80us run", pooled -- excludes lin80us
MIN_N = 15


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> list of built Dataset objects."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    names = [k for k in sorted(groups) if any(r in k for r in runs)]  # type: ignore
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return run_dss


def _site_breakdown(run_dss, n_sites):
    """Per-site [n_alive, n_total], split into stationary / moved-by-source / moved-by-target."""
    stat = np.zeros((n_sites, 2), int)
    src_moved = np.zeros((n_sites, 2), int)
    tgt_moved = np.zeros((n_sites, 2), int)
    for ds in run_dss:
        m2 = ds.occ(C.SURV_FRAME)
        for i, p in enumerate(ds.plans):
            for a in range(p.k):
                site, tgt, d = int(p.sites[a]), int(p.targets[a]), int(p.dist[a])
                alive = int(bool(m2[i, tgt]))
                if d == 0:
                    stat[site] += (alive, 1)
                else:
                    src_moved[site] += (alive, 1)
                    tgt_moved[tgt] += (alive, 1)
    return stat, src_moved, tgt_moved


def _errorbar(ax, arr, color, label, zorder=3):
    xs = [i for i in range(len(arr)) if arr[i][1] >= MIN_N]
    if not xs:
        return
    y, el, eh = [], [], []
    for i in xs:
        p, lo, hi = C.wilson(*arr[i])
        y.append(p)
        el.append(p - lo)
        eh.append(hi - p)
    ax.errorbar(
        xs,
        y,
        yerr=[el, eh],
        marker="o",
        capsize=3,
        color=color,
        label=label,
        zorder=zorder,
    )


def _two_panel_figure(
    stat,
    src_moved,
    tgt_moved,
    stat_color="0.4",
    moved_color="tab:red",
    overall_color="k",
):
    """Left: survival by source site. Right: survival by target site. Each with
    stationary / moving / overall curves (colors are parameters, per the docstring)."""
    n = len(stat)
    overall_src = stat + src_moved
    overall_tgt = stat + tgt_moved
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, moved, overall, xlabel, tag in (
        (ax1, src_moved, overall_src, "source site", "(a)"),
        (ax2, tgt_moved, overall_tgt, "target site", "(b)"),
    ):
        _errorbar(ax, stat, stat_color, "stationary", zorder=2)
        _errorbar(ax, moved, moved_color, "moving", zorder=3)
        _errorbar(ax, overall, overall_color, "overall", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("survival")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(n))
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.3)
        ax.text(
            0.02,
            0.96,
            tag,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )
    ax1.legend(fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig


def _combined_figure(
    stat,
    src_moved,
    tgt_moved,
    stat_color="0.25",
    src_color="tab:red",
    tgt_color="tab:blue",
):
    """Alternate single-panel view: stationary in the background, source- and
    target-site moving curves overlaid in contrasting colors."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _errorbar(ax, stat, stat_color, "stationary", zorder=2)
    _errorbar(ax, src_moved, src_color, "moving, by source site", zorder=3)
    _errorbar(ax, tgt_moved, tgt_color, "moving, by target site", zorder=3)
    ax.set_xlabel("site")
    ax.set_ylabel("survival")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(len(stat)))
    ax.set_title("Survival by source and target site, combined")
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot(runs=None, save=True, combined=False, images=None, name=None):
    """The 80us-run source/target site survival breakdown. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    n_sites = run_dss[0].cal.n
    stat, src_moved, tgt_moved = _site_breakdown(run_dss, n_sites)
    fig = (
        _combined_figure(stat, src_moved, tgt_moved)
        if combined
        else _two_panel_figure(stat, src_moved, tgt_moved)
    )
    if save:
        name = name or (
            "_".join(runs) if runs is not None else "_".join(DEFAULT_RUNS)
        )
        title = "survival_by_source_target_site" + ("_combined" if combined else "")
        C.save_figure(fig, CATEGORY, name, title)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings pooled into one run set"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument(
        "--combined", action="store_true", help="the alternate single-panel view"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        for ds in _load(a.runs if a.runs is not None else DEFAULT_RUNS, a.images):
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    plot(runs=a.runs, save=not a.no_save, combined=a.combined, images=a.images)
    if a.no_save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
