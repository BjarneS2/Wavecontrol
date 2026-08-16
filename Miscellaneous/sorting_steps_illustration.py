"""
Image creation for chapter 3 about the sorting algorithm illustrating
how the individual steps are evaluated in my code.

Taken some experimental run from ./Data/ from tweezerImagesSorting1D
e.g. from the 80us run. I would wanna look for a run that was successfull
with the sorting and has like 5 or more atoms, where at least 3 atoms had
to move. Better even if not from just one, but from two sides.

Plot: 4 panels stacked on the same real image crop, each adding one layer
of the sorting pipeline:
    (a) detect occupancy   - encircle every trap, green if filled, red if empty.
        Uses frame 1 ("loading") of the shot.
    (b) choose fill window - the move-minimising target block: green/slate
        inside it, amber/dotted outside.
    (c) plan move order     - numbered arrows, source -> destination.
    (d) sorted array        - frame 2 ("survival") of the SAME shot, taken
        while the sorted block is held: the real, measured outcome, not a
        schematic.

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
SHOT_FILE = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\tweezerLoad1x11-bestsort-80us_3_20260729-121718.npy"
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


def _target_box(ax, lo, hi, ls="--"):
    box = Rectangle(
        (lo - 0.5, -0.62),
        hi - lo + 1,
        1.24,
        fill=False,
        edgecolor=BOX_EDGE,
        lw=2.0,
        ls=ls,
        zorder=4,
    )
    box.set_path_effects(STROKE)
    ax.add_patch(box)


def _callout(ax, x, y, xt, yt, text, color, text_color=None):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(xt, yt),
        fontsize=7.8,
        style="italic",
        color=text_color or color,
        ha="center",
        va="center",
        zorder=9,
        arrowprops=dict(
            arrowstyle="-", lw=0.9, color=color, shrinkA=2, shrinkB=6, alpha=0.85
        ),
        bbox=CALLOUT_BBOX,
    )


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


def plot_sorting_steps(cal, img, mask, sites, moves, final_img, final_mask):
    n = cal.n
    n_atoms = int(mask.sum())
    lo, hi = int(sites[0]), int(sites[-1])
    r0, r1, c0, c1, x0, x1 = _crop_extent(cal)
    crop = img[r0:r1, c0:c1].T
    final_crop = final_img[r0:r1, c0:c1].T
    vmin, vmax = np.percentile(crop, 45), np.percentile(crop, 99.5)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(8.6, 5.4),
        gridspec_kw={"height_ratios": [1.3, 1, 1, 1.3], "hspace": 0.12},
    )
    ax_cls, ax_win, ax_mov, ax_fin = axes

    # (a) cutout + classification -------------------------------------------------
    ax_cls.imshow(
        crop,
        cmap="viridis",
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, 0, 1),
    )
    for i in range(n):
        if mask[i]:
            _site_disc(ax_cls, i, GREEN, y=0.5, alternative=True)
        else:
            _site_ring(ax_cls, i, GRAY, ls=":", lw=1.2, y=0.5)

    i_full = int(np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2])
    i_empty = int(np.flatnonzero(~mask.astype(bool))[0])
    _callout(ax_cls, i_full, 0.5, i_full + 0.05, 0.86, "occupied", GREEN, text_color="black")
    _callout(ax_cls, i_empty, 0.5, i_empty - 0.05, 0.14, "empty", GRAY, text_color="black")

    # (b) target window -------------------------------------------------------------
    _target_box(ax_win, lo, hi)
    cats_b = {}
    for i in range(n):
        cat = _classify_window(i, mask, lo, hi)
        cats_b.setdefault(cat, i)
        if cat == "in_target":
            _site_disc(ax_win, i, GREEN)
        elif cat == "to_fill":
            _site_ring(ax_win, i, SLATE, ls="--")
        elif cat == "to_vacate":
            _site_disc(ax_win, i, AMBER)
        else:
            _site_ring(ax_win, i, GRAY, ls=":", lw=1.2)

    _callout(
        ax_win,
        cats_b["in_target"],
        0,
        cats_b["in_target"],
        0.78,
        "already filled",
        GREEN,
    )
    _callout(
        ax_win, cats_b["to_fill"], 0, cats_b["to_fill"], -0.78, "must be filled", SLATE
    )
    _callout(
        ax_win, cats_b["to_vacate"], 0, cats_b["to_vacate"], 0.78, "will vacate", AMBER
    )

    # (c) move ordering ---------------------------------------------------------------
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
        rad = 0.35 if dc > sc else -0.35
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
            0.5 * np.sign(rad),
            str(k),
            color=ARROW,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="none"),
        )

    n_right = sum(1 for sc, dc in moves if dc > sc)
    n_left = len(moves) - n_right
    caption = "%d move%s right, %d move%s left, executed in this order" % (
        n_right,
        "" if n_right == 1 else "s",
        n_left,
        "" if n_left == 1 else "s",
    )
    ax_mov.text(
        0.5,
        0.92,
        caption,
        transform=ax_mov.get_yaxis_transform(),
        fontsize=7.8,
        style="italic",
        color="black",
        ha="center",
        va="top",
    )

    # (d) sorted result (real "survival" frame of the same shot) --------------------
    ax_fin.imshow(
        final_crop,
        cmap="viridis",
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, -0.95, 0.95),
    )
    glow = Rectangle(
        (lo - 0.5, -0.62),
        hi - lo + 1,
        1.24,
        facecolor=BLUE,
        alpha=0.10,
        edgecolor="none",
        zorder=1,
    )
    ax_fin.add_patch(glow)
    _target_box(ax_fin, lo, hi, ls="-")
    for i in range(n):
        if final_mask[i]:
            _site_disc(ax_fin, i, BLUE, alternative=True)
        else:
            _site_ring(ax_fin, i, GRAY, ls=":", lw=1.2)
    n_final = int(final_mask.sum())
    ax_fin.text(
        0.5,
        0.80,
        "sorted: %d/%d target sites filled after %d moves"
        % (n_final, n_atoms, len(moves)),
        transform=ax_fin.get_yaxis_transform(),
        fontsize=8.3,
        style="italic",
        color="black",
        ha="center",
        va="center",
        bbox=CALLOUT_BBOX,
        zorder=9,
    )

    # cosmetics -----------------------------------------------------------------------
    row_labels = [
        (ax_cls, r"$\mathbf{(a)}$" + "\ndetect\noccupancy"),
        (ax_win, r"$\mathbf{(b)}$" + "\nchoose\nwindow"),
        (ax_mov, r"$\mathbf{(c)}$" + "\nplan\nmoves"),
        (ax_fin, r"$\mathbf{(d)}$" + "\nsorted\narray"),
    ]
    for ax, label in row_labels:
        ax.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=9.5,
            linespacing=1.35,
            color=TEXT,
        )

    ax_cls.set_ylim(0, 1)
    for ax in (ax_win, ax_mov, ax_fin):
        ax.set_ylim(-0.95, 0.95)
    xlim = (-0.5 - 0.4, n - 1 + 0.5 + 0.4)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    ax_cls.set_xticks(range(n))
    ax_cls.xaxis.set_ticks_position("top")
    ax_cls.xaxis.set_label_position("top")
    ax_cls.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=3,
        labelsize=8.5,
    )
    ax_cls.set_xlabel("trap index", fontsize=9.5)

    ax_fin.set_xticks(range(n))
    ax_fin.tick_params(bottom=True, labelbottom=True, length=3, labelsize=8.5)
    # ax_fin.set_xlabel("trap index", fontsize=9.5)
    for ax in (ax_win, ax_mov):
        ax.set_xticks([])

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.90, bottom=0.085)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, "sorting_steps.pdf"), facecolor="white")
    fig.savefig(os.path.join(out_dir, "sorting_steps.png"), dpi=300, facecolor="white")
    print("saved sorting_steps.png / .pdf")

    plt.show()
    return fig


if __name__ == "__main__":
    cal, img, mask, sites, moves, final_img, final_mask = load_illustration_shot()
    plot_sorting_steps(cal, img, mask, sites, moves, final_img, final_mask)
