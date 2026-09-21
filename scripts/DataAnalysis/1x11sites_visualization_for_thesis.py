"""
1x11 tweezer array, experimental sorting data: four bare image strips for the thesis.

    (1) the full array, calibration mean image
    (2) one shot with 7 atoms loaded, occupancy marked (solid / dotted rings)
    (3) the same shot, plus the target window and the planned move order
    (4) the survival frame of the same shot: the sorted array inside the window

Sites are 4.6 um apart, which sets the scale bar in the lower left corner of every
panel. No titles, no legends -- only the cropped strip.

@author: Bjarne Schuemann
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patheffects import withStroke

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "Sorting"))

import CommonThings as C
from sorter import BINNING, LOAD_FRAME, _counts_to_photons, load_calibration, plan_moves

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0

CALIB_NPZ = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\calibration\sorter_calibration.npz"
SHOT_FILE = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\tweezerLoad1x11-bestsort-80us_3_20260729-121718.npy"
SURVIVAL_FRAME = 2

SPACING_UM = 4.6
BAR_UM = 4.6
AXIS_MARGIN_PX = 10  # along the array
PERP_MARGIN_PX = 14  # across the array -> strip thickness

BOX_EDGE = "#b8720f"
ARROW = "#2b2b2b"

MARKER_S = 210
DARK_STROKE = [withStroke(linewidth=2.2, foreground="black", alpha=0.55)]
WHITE_STROKE = [withStroke(linewidth=2.6, foreground="black", alpha=0.6)]


def load_illustration_shot(shot_file=SHOT_FILE, calib_npz=CALIB_NPZ):
    cal = load_calibration(calib_npz)
    frames = np.asarray(np.load(shot_file, allow_pickle=True)[()]["Images"])
    img = _counts_to_photons(frames[LOAD_FRAME], BINNING)
    mask = cal.occupancy(img)
    sites, moves = plan_moves(mask)
    final_img = _counts_to_photons(frames[SURVIVAL_FRAME], BINNING)
    final_mask = cal.occupancy(final_img)
    return cal, img, mask, sites, moves, final_img, final_mask


def crop_geometry(cal):
    """Crop box in pixels plus the extent of that box in trap-index units.

    x runs along the array (one unit = one site spacing), y across it, centred on the
    trap row, so the strip is drawn true to scale with aspect='equal'."""
    axis_px = cal.locations[:, 1]
    perp_px = cal.locations[:, 0]
    spacing = float(np.median(np.diff(axis_px)))
    perp_c = float(np.mean(perp_px))

    r0 = int(np.floor(axis_px.min() - AXIS_MARGIN_PX))
    r1 = int(np.ceil(axis_px.max() + AXIS_MARGIN_PX))
    c0 = int(np.floor(perp_c - PERP_MARGIN_PX))
    c1 = int(np.ceil(perp_c + PERP_MARGIN_PX))

    # imshow extents are pixel edges, the site positions are pixel centres
    x0 = (r0 - 0.5 - axis_px[0]) / spacing
    x1 = (r1 - 0.5 - axis_px[0]) / spacing
    y0 = (c0 - 0.5 - perp_c) / spacing
    y1 = (c1 - 0.5 - perp_c) / spacing
    return (r0, r1, c0, c1), (x0, x1, y0, y1), spacing, perp_c


def site_xy(cal, spacing, perp_c):
    """Site centres in the index coordinates used by the plot."""
    x = (cal.locations[:, 1] - cal.locations[0, 1]) / spacing
    y = (cal.locations[:, 0] - perp_c) / spacing
    return x, y


def _ring(ax, x, y, filled):
    art = ax.scatter(
        [x],
        [y],
        s=MARKER_S,
        facecolors="none",
        edgecolors="white",
        linestyles="-" if filled else ":",
        linewidths=1.9 if filled else 1.3,
        zorder=7,
    )
    art.set_path_effects(DARK_STROKE)


def _target_box(ax, lo, hi, ls="--"):
    box = Rectangle(
        (lo - 0.55, -0.62),
        hi - lo + 1.1,
        1.24,
        fill=False,
        edgecolor=BOX_EDGE,
        lw=2.0,
        ls=ls,
        zorder=5,
    )
    box.set_path_effects(WHITE_STROKE)
    ax.add_patch(box)


def _moves(ax, moves, x_site, y_site):
    for k, (sc, dc) in enumerate(moves, 1):
        rad = 0.35 if dc > sc else -0.35
        arrow = FancyArrowPatch(
            (x_site[sc], y_site[sc]),
            (x_site[dc], y_site[dc]),
            connectionstyle="arc3,rad=%.2f" % rad,
            arrowstyle="-|>",
            mutation_scale=12,
            color="white",
            lw=1.7,
            zorder=8,
        )
        arrow.set_path_effects(WHITE_STROKE)
        ax.add_patch(arrow)
        ax.text(
            (x_site[sc] + x_site[dc]) / 2,
            0.96 * np.sign(rad),
            str(k),
            color=ARROW,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="none"),
        )


def _scale_bar(ax, x0, x1, y0, y1):
    """Bar of BAR_UM in the lower right corner (index units: 1 unit = SPACING_UM)."""
    length = BAR_UM / SPACING_UM
    x = x1 - 0.035 * (x1 - x0) - length
    y = y1 - 0.16 * (y1 - y0)
    ax.add_patch(
        Rectangle(
            (x, y),
            length,
            0.045 * (y1 - y0),
            facecolor="white",
            edgecolor="none",
            zorder=10,
        )
    )
    ax.text(
        x + length / 2,
        y - 0.055 * (y1 - y0),
        rf"${BAR_UM:g}\,\mu\mathrm{{m}}$",
        color="white",
        fontsize=16.0,
        ha="center",
        va="bottom",
        zorder=10,
        path_effects=[withStroke(linewidth=1.6, foreground="black")],
    )


def strip(cal, img, vlim=None, mask=None, window=None, moves=None, box_ls="--"):
    (r0, r1, c0, c1), (x0, x1, y0, y1), spacing, perp_c = crop_geometry(cal)
    crop = img[r0:r1, c0:c1].T
    vmin, vmax = vlim if vlim else (np.percentile(crop, 45), np.percentile(crop, 99.5))

    width = x1 - x0
    height = y1 - y0
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * height / width))
    ax.imshow(
        crop,
        cmap="viridis",
        origin="upper",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, y1, y0),
        interpolation="nearest",
    )

    xs, ys = site_xy(cal, spacing, perp_c)
    if window is not None:
        _target_box(ax, int(window[0]), int(window[-1]), ls=box_ls)
    if mask is not None:
        for i in range(cal.n):
            _ring(ax, xs[i], ys[i], bool(mask[i]))
    if moves is not None:
        _moves(ax, moves, xs, ys)

    _scale_bar(ax, x0, x1, y0, y1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def save(fig, name):
    out_dir = C.FIGURES_DIR / "Miscellaneous"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.svg"
    fig.savefig(path, dpi=600, facecolor="white")
    print(f"  saved {path}")


def main():
    cal, img, mask, sites, moves, final_img, final_mask = load_illustration_shot()
    print(
        f"  {cal.n} sites, {int(mask.sum())} atoms loaded, window {sites[0]}-{sites[-1]}, "
        f"{len(moves)} moves, {int(final_mask.sum())} atoms after sorting"
    )

    (r0, r1, c0, c1), _, _, _ = crop_geometry(cal)
    crop = img[r0:r1, c0:c1]
    vlim = (np.percentile(crop, 45), np.percentile(crop, 99.5))

    save(strip(cal, cal.mean_img), "array1x11_mean")
    save(strip(cal, img, vlim=vlim, mask=mask), "array1x11_loaded")
    save(
        strip(cal, img, vlim=vlim, mask=mask, window=sites, moves=moves),
        "array1x11_window_moves",
    )
    save(
        strip(cal, final_img, vlim=vlim, mask=final_mask, window=sites, box_ls="-"),
        "array1x11_sorted",
    )

    plt.show()


if __name__ == "__main__":
    main()
