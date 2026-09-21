"""
8x8 tweezer array: average the loading frames of a run, localize the 64 sites and
show one random single shot -- once bare and once with the occupancy labelling
(filled / empty), in the plotting style of the AWGController illustrations.

Site spacing is 1.2 um, which sets the scale bar in the lower left corner.

@author: Bjarne Schümann
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import CommonThings as C

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0

IMAGES_8X8 = C.DATA_DIR / "tweezerImages8x8"
RUN = "tweezerLoad8x8-2"
SPACING_UM = 4.6
BAR_UM = 5.0
GRID = 8
R_PIX = 3.5
CROP_MARGIN_PX = 9
BAR_PAD_PX = 14  # extra strip at the bottom that carries the scale bar

GREEN = "#2f9e58"
GRAY = "#9a9a9a"
TEXT = "#232323"

STROKE = [withStroke(linewidth=2.3, foreground="white")]
CALLOUT_BBOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82)
MARKER_S = 150


def _cluster(v, w, n):
    """1D positions with weights -> the n strongest cluster centres (the grid lines)."""
    order = np.argsort(v)
    v, w = np.asarray(v, float)[order], np.asarray(w, float)[order]
    cuts = np.flatnonzero(np.diff(v) > 2.0) + 1
    groups = [(g, gw) for g, gw in zip(np.split(v, cuts), np.split(w, cuts))]
    groups = sorted(groups, key=lambda t: t[1].sum(), reverse=True)[:n]
    return np.sort([np.average(g, weights=gw) for g, gw in groups])


def locate_grid(mean_img, grid=GRID):
    """Mean loading image -> (grid**2, 2) site centres as (x, y), row-major."""
    sm = ndimage.gaussian_filter(mean_img, 1.0)
    peaks = (sm == ndimage.maximum_filter(sm, 5)) & (sm > np.percentile(sm, 97))
    peaks[:6] = peaks[-6:] = False
    peaks[:, :6] = peaks[:, -6:] = False
    ys, xs = np.nonzero(peaks)
    w = sm[ys, xs]
    xc, yc = _cluster(xs, w, grid), _cluster(ys, w, grid)

    locs = []
    for y in yc:
        for x in xc:
            r0, r1 = int(round(y)) - 4, int(round(y)) + 5
            c0, c1 = int(round(x)) - 4, int(round(x)) + 5
            box = sm[r0:r1, c0:c1] - sm[r0:r1, c0:c1].min()
            dy, dx = ndimage.center_of_mass(box)
            locs.append((c0 + dx, r0 + dy))
    return np.asarray(locs, float), float(np.median(np.diff(xc)))


def load_run(folder=IMAGES_8X8, run=RUN):
    raw = C.load_stack(str(folder), run)
    im = C.to_photons(raw, 2)
    load = im[:, C.LOAD_FRAME]
    mean_img = load.mean(0)
    locs, spacing_px = locate_grid(mean_img)
    masks = C.site_masks(mean_img.shape, locs, R_PIX)
    counts = C.roi_mask_counts(load, masks).T  # (n_shots, n_sites)
    try:
        thr, _ = C.bimodal_threshold(counts.ravel())
    except Exception:  # noqa: BLE001
        thr = C.valley_threshold(counts.ravel())
    print(
        f"  run {run}: {len(load)} shots, {len(locs)} sites, "
        f"spacing {spacing_px:.2f} px = {SPACING_UM} um, threshold {thr:.2f} photons, "
        f"mean filling {np.mean(counts > thr):.3f}"
    )
    return load, mean_img, locs, spacing_px, counts, float(thr)


def _crop(locs, shape):
    x0 = max(int(np.floor(locs[:, 0].min() - CROP_MARGIN_PX)), 0)
    x1 = min(int(np.ceil(locs[:, 0].max() + CROP_MARGIN_PX)), shape[1])
    y0 = max(int(np.floor(locs[:, 1].min() - CROP_MARGIN_PX)), 0)
    y1 = min(int(np.ceil(locs[:, 1].max() + CROP_MARGIN_PX + BAR_PAD_PX)), shape[0])
    return x0, x1, y0, y1


def _scale_bar(ax, x0, x1, y0, y1, spacing_px):
    """Bar of BAR_UM in the lower left corner, in cropped pixel coordinates."""
    length = BAR_UM * spacing_px / SPACING_UM
    x = x0 + 0.05 * (x1 - x0)
    y = y1 - 0.35 * BAR_PAD_PX
    bar = Rectangle(
        (x, y),
        length,
        0.1 * BAR_PAD_PX,
        facecolor="white",
        edgecolor="black",
        lw=0.6,
        zorder=10,
    )
    ax.add_patch(bar)
    ax.text(
        x + length / 2,
        y - 0.1 * BAR_PAD_PX,
        rf"${BAR_UM:g}\,\mu$m",
        color="white",
        fontsize=8.5,
        ha="center",
        va="bottom",
        zorder=10,
        path_effects=[withStroke(linewidth=2.0, foreground="black")],
    )


def plot_shot(img, locs, spacing_px, mask=None, title=None, vlim=None):
    x0, x1, y0, y1 = _crop(locs, img.shape)
    crop = img[y0:y1, x0:x1]
    vmin, vmax = vlim if vlim else (np.percentile(crop, 20), np.percentile(crop, 99.7))

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(
        crop,
        cmap="viridis",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(x0, x1, y1, y0),
        interpolation="nearest",
    )

    if mask is not None:
        for (x, y), filled in zip(locs, mask):
            if filled:
                # art = ax.scatter(
                #     [x],
                #     [y],
                #     s=MARKER_S,
                #     facecolors="none",
                #     edgecolors=GREEN,
                #     linestyles=":",
                #     linewidths=1.8,
                #     zorder=6,
                # )
                # art.set_path_effects(STROKE)
                continue
            else:
                ax.scatter(
                    [x],
                    [y],
                    s=MARKER_S,
                    facecolors="none",
                    edgecolors=GRAY,
                    linestyles=":",
                    linewidths=1.2,
                    zorder=5,
                )
        n_full = int(np.sum(mask))
        # ax.text(
        #     0.5,
        #     1.015,
        #     # f"{n_full}/{len(locs)} sites occupied",
        #     transform=ax.transAxes,
        #     fontsize=9.0,
        #     style="italic",
        #     color=TEXT,
        #     ha="center",
        #     va="bottom",
        # )

    _scale_bar(ax, x0, x1, y0, y1, spacing_px)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    # if title:
    #     ax.set_title(title, fontsize=10, color=TEXT, pad=16 if mask is not None else 6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    return fig


def save(fig, name):
    out_dir = C.FIGURES_DIR / "Array8x8"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=600, facecolor="white")
    print(f"  saved {out_dir / name}.svg / .png")


def main(seed=None):
    load, mean_img, locs, spacing_px, counts, thr = load_run()
    rng = np.random.default_rng(seed)
    k = int(rng.integers(len(load)))
    mask = counts[k] > thr
    print(f"  showing shot {k} with {int(mask.sum())} atoms")

    x0, x1, y0, y1 = _crop(locs, mean_img.shape)
    crop = load[k][y0:y1, x0:x1]
    vlim = (np.percentile(crop, 20), np.percentile(crop, 99.7))

    fig_mean = plot_shot(
        mean_img,
        locs,
        spacing_px,  # , title=f"{RUN}: mean of {len(load)} loading frames"
    )
    save(fig_mean, "array8x8_mean")

    fig_raw = plot_shot(
        load[k], locs, spacing_px
    )  # title=f"single shot #{k}", vlim=vlim)
    save(fig_raw, "array8x8_shot_raw")

    fig_lab = plot_shot(
        load[k],
        locs,
        spacing_px,
        mask=mask,
        # title=f"single shot #{k}",
        vlim=vlim,
    )
    save(fig_lab, "array8x8_shot_labelled")

    plt.show()


if __name__ == "__main__":
    main()
