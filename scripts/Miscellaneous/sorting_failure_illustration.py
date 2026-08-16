"""
Image creation for chapter 3 illustrating a failed sorting attempt: one
target site close to the center of the fill window ends up empty after
the move sequence.

Picked from ./Data/tweezerImagesSorting1D (the 80us-linear run) a shot
where the planned window has moves from both sides and exactly one
target site (dead center) is missing an atom in the survival frame.

Plot: 3 panels side by side sharing one real image crop
    (left)   loading frame - encircle every trap, green if filled, red
             if empty, plus the classification into the fill window.
    (center) move plan - numbered arrows, source -> destination, same
             style as sorting_steps_illustration.py panel (c) but
             without the descriptive callouts, slimmer aspect ratio.
    (right)  survival frame - the real, measured outcome, with the
             failed (still-empty) target site highlighted.

Reuse code from sorter.py & the data analysis scripts.

@author: Bjarne Schuemann
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patheffects import withStroke

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sorting"
    ),
)
from sorter import BINNING, LOAD_FRAME, _counts_to_photons, load_calibration, plan_moves

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0

CALIB_NPZ = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\calibration\sorter_calibration.npz"
SHOT_FILE = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\tweezerLoad1x11-sortbest-lin80us_55_20260729-130925.npy"
SURVIVAL_FRAME = 2  # frame taken while the sorted block is held (post-move)

CROP_MARGIN_PX = 12

GREEN = "#2f9e58"
RED = "#d1495b"
AMBER = "#e0972c"
BOX_EDGE = "#b8720f"
SLATE = "#5c6fa8"
BLUE = "#3477eb"
GRAY = "#9a9a9a"
ARROW = "#2b2b2b"
TEXT = "#232323"

STROKE = [withStroke(linewidth=2.3, foreground="white")]
CALLOUT_BBOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82)

MARKER_S = 260


def load_illustration_shot(shot_file=SHOT_FILE, calib_npz=CALIB_NPZ):
    cal = load_calibration(calib_npz)
    frames = np.asarray(np.load(shot_file, allow_pickle=True)[()]["Images"])
    img = _counts_to_photons(frames[LOAD_FRAME], BINNING)
    mask = cal.occupancy(img)
    sites, moves = plan_moves(mask)
    final_img = _counts_to_photons(frames[SURVIVAL_FRAME], BINNING)
    final_mask = cal.occupancy(final_img)
    return cal, img, mask, sites, moves, final_img, final_mask


def _site_ring(ax, x, edgecolor, ls="-", lw=1.9, y=0.0, z=6):
    art = ax.scatter(
        [x],
        [y],
        s=MARKER_S,
        facecolors="none",
        edgecolors=edgecolor,
        linestyles=ls,
        linewidths=lw,
        zorder=z,
    )
    art.set_path_effects(STROKE)


def _site_disc(ax, x, color, y=0.0, z=7, alternative=False):
    if alternative:
        ax.scatter(
            [x],
            [y],
            s=MARKER_S * 0.82,
            facecolors="none",
            edgecolors="white",
            linewidths=0.9,
            zorder=z,
        )
    else:
        ax.scatter(
            [x],
            [y],
            s=MARKER_S * 0.82,
            facecolors=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=z,
        )


def _target_box(ax, lo, hi, ls="--", half_h=0.62):
    box = Rectangle(
        (lo - 0.5, -half_h),
        hi - lo + 1,
        2 * half_h,
        fill=False,
        edgecolor=BOX_EDGE,
        lw=2.0,
        ls=ls,
        zorder=4,
    )
    box.set_path_effects(STROKE)
    ax.add_patch(box)


def _classify_window(i, mask, lo, hi):
    inside = lo <= i <= hi
    if inside and mask[i]:
        return "in_target"
    if inside and not mask[i]:
        return "to_fill"
    if mask[i]:
        return "to_vacate"
    return "unused"


def _crop_extent(cal):
    axis_px = cal.locations[:, 1]
    perp_px = cal.locations[:, 0]
    spacing = np.median(np.diff(axis_px))

    def index_of(px):
        return (px - axis_px[0]) / spacing

    r0 = int(np.floor(axis_px.min() - CROP_MARGIN_PX))
    r1 = int(np.ceil(axis_px.max() + CROP_MARGIN_PX))
    c0 = int(np.floor(perp_px.min() - CROP_MARGIN_PX))
    c1 = int(np.ceil(perp_px.max() + CROP_MARGIN_PX))
    return r0, r1, c0, c1, index_of(r0), index_of(r1)


def plot_sorting_failure(cal, img, mask, sites, moves, final_img, final_mask):
    n = cal.n
    n_atoms = int(mask.sum())
    lo, hi = int(sites[0]), int(sites[-1])
    missing = [i for i in range(lo, hi + 1) if not final_mask[i]]
    r0, r1, c0, c1, x0, x1 = _crop_extent(cal)
    crop = img[r0:r1, c0:c1].T
    final_crop = final_img[r0:r1, c0:c1].T
    vmin, vmax = np.percentile(crop, 45), np.percentile(crop, 99.5)

    spacing = np.median(np.diff(cal.locations[:, 1]))
    yh = (c1 - c0) / spacing / 2  # square-pixel half-height, in trap-index units
    box_half = yh * 0.653  # keep the box's proportion to the crop from before

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.2, 2.5),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.08},
    )
    ax_load, ax_mov, ax_surv = axes

    # (left) loading frame ----------------------------------------------------------
    ax_load.imshow(
        crop,
        cmap="viridis",
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, -yh, yh),
    )
    ax_load.set_aspect("equal", adjustable="datalim")
    _target_box(ax_load, lo, hi, half_h=box_half)
    for i in range(n):
        if mask[i]:
            _site_disc(ax_load, i, GREEN, alternative=True)
        else:
            _site_ring(ax_load, i, GRAY, ls=":", lw=1.2)
    ax_load.set_title("loading", fontsize=9.5, color=TEXT, pad=4)

    # (center) move plan --------------------------------------------------------------
    _target_box(ax_mov, lo, hi)
    for i in range(n):
        cat = _classify_window(i, mask, lo, hi)
        if cat == "in_target":
            _site_disc(ax_mov, i, GREEN)
        elif cat == "to_fill":
            _site_ring(ax_mov, i, SLATE, ls="--")
        elif cat == "to_vacate":
            _site_disc(ax_mov, i, AMBER)
        else:
            _site_ring(ax_mov, i, GRAY, ls=":", lw=1.2)
    for k, (sc, dc) in enumerate(moves, 1):
        rad = 0.55 if dc > sc else -0.55
        arrow = FancyArrowPatch(
            (sc, 0),
            (dc, 0),
            connectionstyle="arc3,rad=%.2f" % rad,
            arrowstyle="-|>",
            mutation_scale=13,
            color=ARROW,
            lw=1.8,
            zorder=8,
        )
        arrow.set_path_effects(STROKE)
        ax_mov.add_patch(arrow)
        ax_mov.text(
            (sc + dc) / 2,
            0.275 * np.sign(rad),
            str(k),
            color=ARROW,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="none"),
        )
    ax_mov.set_title("planned moves", fontsize=9.5, color=TEXT, pad=4)

    # (right) survival frame ------------------------------------------------------------
    ax_surv.imshow(
        final_crop,
        cmap="viridis",
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, -yh, yh),
    )
    ax_surv.set_aspect("equal", adjustable="datalim")
    glow = Rectangle(
        (lo - 0.5, -box_half),
        hi - lo + 1,
        2 * box_half,
        facecolor=BLUE,
        alpha=0.10,
        edgecolor="none",
        zorder=1,
    )
    ax_surv.add_patch(glow)
    _target_box(ax_surv, lo, hi, ls="-", half_h=box_half)
    for i in range(n):
        if final_mask[i]:
            _site_disc(ax_surv, i, BLUE, alternative=True)
        else:
            _site_ring(ax_surv, i, GRAY, ls=":", lw=1.2)
    for i in missing:
        _site_ring(ax_surv, i, RED, ls="-", lw=2.3, z=8)
    ax_surv.set_title("survival", fontsize=9.5, color=TEXT, pad=4)

    # cosmetics -----------------------------------------------------------------------
    # ax_load / ax_surv keep the ylim matplotlib derives from aspect="equal" +
    # adjustable="datalim" so their box is the same size as ax_mov's, full width.
    ax_mov.set_ylim(-0.95, 0.95)
    xlim = (-0.5 - 0.4, n - 1 + 0.5 + 0.4)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    for ax in axes:
        ax.set_xticks(range(n))
        ax.tick_params(bottom=True, labelbottom=True, length=3, labelsize=7.5)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.86, bottom=0.14)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, "sorting_failure.pdf"), facecolor="white")
    fig.savefig(
        os.path.join(out_dir, "sorting_failure.png"), dpi=300, facecolor="white"
    )
    print("saved sorting_failure.png / .pdf")

    plt.show()
    return fig


if __name__ == "__main__":
    cal, img, mask, sites, moves, final_img, final_mask = load_illustration_shot()
    plot_sorting_failure(cal, img, mask, sites, moves, final_img, final_mask)
