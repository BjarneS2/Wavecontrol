"""
This script analyzes the data from the single atom trasnport experiment.
Here we show a plot where on the Y axis we will sort the runs of the
experiment by the number of atoms that were loaded (K) and then sort also
according to the target window (starting position/index of the). The X axis
will show the site index. We will show this plot once in a black and white
version for the loading image and then use it again for comparison sakes to
the second plot where the survival image will be overlaid by a colored (red)
target site window.

This script should only be ran and used for single runs of the experiment.

@author: Bjarne Schümann
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import CommonThings as C
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Rectangle

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = [
    "tweezerLoad1x11-sortbest"
]  # a run with real K diversity, as a sane default


def _resolve_run(runs, images=None, min_shots=20):
    """--runs substrings -> the single matching Dataset, with plans already built."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    if len(runs) == 1 and runs[0] in groups:
        matches = [runs[0]]
    else:
        matches = [k for k in sorted(groups) if any(r in k for r in runs)]
    if len(matches) < 1:
        raise ValueError(f"--runs {runs!r} matched no runs")
    run_dss = []
    for name in matches:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    if len(run_dss) != 1:
        raise ValueError(
            f"--runs {runs!r} matched {len(run_dss)} runs {[d.name for d in run_dss]} "
            "-- this script needs exactly one (or a configured merge group)"
        )
    return run_dss[0]


def _raster_figure(ds):
    """Shots sorted by (K, t0): frame 1 in black/white, frame 2 with the target window in red."""
    order = sorted(range(ds.n_shots), key=lambda i: (ds.plans[i].k, ds.plans[i].t0))
    m1 = ds.occ(C.LOAD_FRAME)[order]
    m2 = ds.occ(C.SURV_FRAME)[order]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, min(14, 0.05 * ds.n_shots + 2.5)), sharey=True
    )
    for ax, (M, ttl) in zip(axes, ((m1, "frame 1: loaded"), (m2, "frame 2: survived"))):
        ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_xlabel("site")
        ax.set_xticks(range(ds.cal.n))
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("shot (sorted by K, then by target block)")
    for row, i in enumerate(order):
        p = ds.plans[i]
        if p.k:
            axes[1].add_patch(
                Rectangle(
                    (p.t0 - 0.5, row - 0.5),
                    p.k,
                    1,
                    fill=False,
                    edgecolor="tab:red",
                    lw=0.6,
                )
            )
    axes[1].set_title("frame 2: survived  -- red = target block", fontsize=10)
    fig.suptitle(
        f"{ds.name} -- {ds.n_shots} shots, sorted by K then target window", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _row_layout(ordered_plans, gap=5, bottom_pad=2):
    """ordered_plans (already sorted by K, then t0) -> (display row per shot, [K, row_start,
    row_end] per K block, total display rows incl. the inter-block gaps and bottom padding)."""
    rows, groups = [], []
    r = 0
    prev_k = None
    for p in ordered_plans:
        if prev_k is not None and p.k != prev_k:
            r += gap
        if p.k != prev_k:
            groups.append([p.k, r, r])
        groups[-1][2] = r
        rows.append(r)
        prev_k = p.k
        r += 1
    return rows, groups, r + bottom_pad


def _survival_rgb(m2, rows, ordered_plans, n_sites, n_rows):
    """Frame-2 image as RGB: black = survived inside its target block, blue = survived
    outside it."""
    rgb = np.ones((n_rows, n_sites, 3))
    for pos, disp_row in enumerate(rows):
        p = ordered_plans[pos]
        blk = np.zeros(n_sites, bool)
        if p.k:
            blk[p.t0 : p.t0 + p.k] = True
        alive = m2[pos]
        rgb[disp_row, alive & blk] = (0, 0, 0)
        rgb[disp_row, alive & ~blk] = (0.12, 0.46, 0.70)
    return rgb


def _draw_k_brackets(ax, groups, side="left"):
    """Square-bracket + 'K=n' label per contiguous block, standing in for the (otherwise
    uninformative) numeric y-ticks. side picks which edge of ax the bracket sits outside of."""
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    edge = 0.0 if side == "left" else 1.0
    sign = -1 if side == "left" else 1
    x0, x1, xt = edge + sign * 0.03, edge + sign * 0.015, edge + sign * 0.045
    ha = "right" if side == "left" else "left"
    for k, r0, r1 in groups:
        y0, y1 = r0 - 0.5, r1 + 0.5
        ax.plot(
            [x0, x0], [y0, y1], color="black", lw=0.8, transform=trans, clip_on=False
        )
        ax.plot(
            [x0, x1], [y0, y0], color="black", lw=0.8, transform=trans, clip_on=False
        )
        ax.plot(
            [x0, x1], [y1, y1], color="black", lw=0.8, transform=trans, clip_on=False
        )
        ax.text(
            xt,
            (y0 + y1) / 2,
            f"K={k}",
            transform=trans,
            ha=ha,
            va="center",
            fontsize=7,
        )
    ax.set_yticks([])


def _raster_figure_v2(ds):
    """Same shots as _raster_figure, but with a gap between K-blocks, a K bracket instead
    of numeric y-ticks, bottom padding, no title, and stray (out-of-block) frame-2 counts
    marked in blue."""
    order = sorted(range(ds.n_shots), key=lambda i: (ds.plans[i].k, ds.plans[i].t0))
    ordered_plans = [ds.plans[i] for i in order]
    m1 = ds.occ(C.LOAD_FRAME)[order]
    m2 = ds.occ(C.SURV_FRAME)[order]
    rows, groups, n_rows = _row_layout(ordered_plans)
    n_sites = ds.cal.n

    M1 = np.zeros((n_rows, n_sites), bool)
    for pos, disp_row in enumerate(rows):
        M1[disp_row] = m1[pos]
    rgb2 = _survival_rgb(m2, rows, ordered_plans, n_sites, n_rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, min(14, 0.05 * n_rows + 2.5)), sharey=True
    )
    axes[0].imshow(M1, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].imshow(rgb2, aspect="auto", interpolation="nearest")
    for ax, ttl in zip(axes, ("frame 1: loaded", "frame 2: survived")):
        ax.set_xlabel("site")
        ax.set_xticks(range(n_sites))
        ax.set_title(ttl, fontsize=10)
    for pos, disp_row in enumerate(rows):
        p = ordered_plans[pos]
        if p.k:
            axes[1].add_patch(
                Rectangle(
                    (p.t0 - 0.5, disp_row - 0.5),
                    p.k,
                    1,
                    fill=False,
                    edgecolor="tab:red",
                    lw=0.6,
                )
            )
    axes[1].set_title(
        "frame 2: survived -- red = target block, blue = outside it", fontsize=10
    )
    _draw_k_brackets(axes[0], groups, side="left")
    _draw_k_brackets(axes[1], groups, side="right")
    fig.tight_layout()
    return fig


def plot(runs=None, save=True, images=None):
    """The single-run K/target-window-sorted raster, current and v2. Returns (fig, fig_v2)."""
    ds = _resolve_run(runs if runs is not None else DEFAULT_RUNS, images)
    fig = _raster_figure(ds)
    fig_v2 = _raster_figure_v2(ds)
    if save:
        C.save_figure(fig, CATEGORY, ds.name, "shots_sorted_by_k_and_target_window")
        C.save_figure(
            fig_v2,
            CATEGORY,
            ds.name,
            "shots_sorted_by_k_and_target_window_v2",
            bbox_inches="tight",
        )
    return fig, fig_v2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting exactly one run"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--list", action="store_true", help="list all runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        images = a.images or str(C.SORTING1D_IMAGES)
        for k in sorted(C.discover_runs(images, 20, verbose=False)):
            print(f"  {k}")
        return 0

    plot(runs=a.runs, save=not a.no_save, images=a.images)
    if a.no_save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
