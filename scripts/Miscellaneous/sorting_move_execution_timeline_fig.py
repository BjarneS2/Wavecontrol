"""
Variant of sorting_move_execution_fig.py that additionally tells the imaging
story: which of the three images of the shot is taken when, relative to the
AWG move execution. Same run/style as sorting_move_execution_fig.py, but the
top row now shows all three raw images (frame 0/1/2) instead of just the
loading frame, and the time axis underneath is broken into four segments:

    (1) "before"      -- image 1 (frame 0), taken well before the sequence,
                          exact timing not tracked -> break to the right of it.
    (2) "just before"  -- image 2 (frame 1, loading/occupancy read), taken a
                          few us before the moves start -> another break,
                          since this is a different timescale than (1).
    (3) "move"         -- the actual AWG move execution (continuous with (4)).
    (4) "hold"         -- the block is held in place; image 3 (frame 2,
                          survival) is taken here, no break to (3) since both
                          are on the same, known timescale.

@author: Bjarne Schümann
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sorting"
    ),
)
from sorter import (
    build_sort_trajectory,
    convert_position_to_freq,
    plan_moves,
    load_calibration,
    _counts_to_photons,
    BINNING,
    LOAD_FRAME,
    MAX_TOTAL_CH0,
    F_START_HZ,
    UM_PER_MHZ,
)
from sorting_steps_illustration import (
    CALIB_NPZ,
    _site_disc,
    _site_ring,
    GREEN,
    AMBER,
    GRAY,
    BLUE,
    BOX_EDGE,
    STROKE,
)
from sorting_move_execution_fig import freq_to_pos, pos_to_freq

SHOT_FILE = r"c:\dev\GitHub\AWGController\Data\tweezerImagesSorting1D\tweezerLoad1x11-bestsort-80us_19_20260729-121849.npy"
SURVIVAL_FRAME = 2

STEP_TIME_S = 80e-6
WPPS = 30
TRAJECTORY = "sta"
LEAD_FRAC = 0.3  # unused directly here, kept for parity with the other figure

# widths (before, just_before, move, hold) and which gaps are a break
COL_WIDTHS = np.array([0.20, 0.16, 0.30, 0.34])
GAP_BREAK = 0.030
GAP_NONE = 0.006
GAPS = [GAP_BREAK, GAP_BREAK, GAP_NONE]

NUM_BADGE = dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor="k", lw=0.8)

TOP_CROP_MARGIN_PX = 5  # tighter than sorting_steps_illustration's CROP_MARGIN_PX


def _crop_extent_px(cal, margin=TOP_CROP_MARGIN_PX):
    """Like sorting_steps_illustration._crop_extent, but returns pixel-unit bounds
    (not trap-index units) so the crop can be shown with square, undistorted pixels."""
    axis_px = cal.locations[:, 1]
    perp_px = cal.locations[:, 0]
    r0 = int(np.floor(axis_px.min() - margin))
    r1 = int(np.ceil(axis_px.max() + margin))
    c0 = int(np.floor(perp_px.min() - margin))
    c1 = int(np.ceil(perp_px.max() + margin))
    site_x_px = axis_px - r0
    return r0, r1, c0, c1, site_x_px


def load_illustration_shot(shot_file=SHOT_FILE, calib_npz=CALIB_NPZ):
    cal = load_calibration(calib_npz)
    frames = np.asarray(np.load(shot_file, allow_pickle=True)[()]["Images"])
    img0 = _counts_to_photons(frames[0], BINNING)
    img1 = _counts_to_photons(frames[LOAD_FRAME], BINNING)
    img2 = _counts_to_photons(frames[SURVIVAL_FRAME], BINNING)
    mask = cal.occupancy(img1)
    sites, moves = plan_moves(mask)
    final_mask = cal.occupancy(img2)
    return cal, img0, img1, img2, mask, sites, moves, final_mask


def _even_cols(left, right, n, gap):
    w = (right - left - gap * (n - 1)) / n
    lefts = [left + i * (w + gap) for i in range(n)]
    return lefts, w


def _col_geometry(left, right):
    n_gaps = sum(GAPS)
    scale = (right - left - n_gaps) / COL_WIDTHS.sum()
    widths = COL_WIDTHS * scale
    lefts = [left]
    for i in range(3):
        lefts.append(lefts[-1] + widths[i] + GAPS[i])
    return lefts, widths


def _row_geometry(top, bottom, ratios, row_gap):
    n_gaps = row_gap * (len(ratios) - 1)
    scale = (top - bottom - n_gaps) / sum(ratios)
    heights = [r * scale for r in ratios]
    tops, bottoms = [], []
    cur = top
    for h in heights:
        tops.append(cur)
        bottoms.append(cur - h)
        cur -= h + row_gap
    return bottoms, heights


def _break_marks(ax_left, ax_right, d=0.025):
    kw = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, d), **kw)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw["transform"] = ax_right.transAxes
    ax_right.plot((-d, d), (-d, d), **kw)
    ax_right.plot((-d, d), (1 - d, 1 + d), **kw)


def _image_axis(fig, rect, crop, vmin, vmax):
    """crop is (Nc, Nr): Nr pixels along the trap line, Nc across it. Plotted with
    aspect="equal" and pixel-unit extent so the traps show up as square pixels,
    letterboxed inside `rect` rather than stretched to fill it."""
    Nc, Nr = crop.shape
    ax = fig.add_axes(rect)
    ax.imshow(
        crop,
        cmap="viridis",
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=(0, Nr, 0, Nc),
    )
    ax.set_xlim(0, Nr)
    ax.set_ylim(0, Nc)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _badge(ax, x, y, label):
    ax.text(
        x,
        y,
        label,
        fontsize=8.5,
        fontweight="bold",
        color="k",
        ha="center",
        va="center",
        zorder=10,
        bbox=NUM_BADGE,
    )


def plot_move_execution_timeline(cal, img0, img1, img2, mask, sites, moves, final_mask):
    n = cal.n
    lo, hi = int(sites[0]), int(sites[-1])
    moving_src = {sc for sc, dc in moves}

    t, P, occ_cols = build_sort_trajectory(
        mask, cal.positions_um, moves, STEP_TIME_S, WPPS, TRAJECTORY
    )
    K = P.shape[0]
    freqs_kt = (
        convert_position_to_freq(P, f_start_hz=F_START_HZ, um_per_MHz=UM_PER_MHZ) / 1e6
    )
    t_ms = t * 1e3
    t_move_end = t_ms[-1]
    t_hold_end = t_move_end + 0.55 * t_move_end
    t_marker3 = t_move_end + 0.28 * t_move_end

    t_before = (-1.05, -0.55)
    t_marker1 = -0.80
    t_just_before = (-0.28, 0.0)
    t_marker2 = -0.06

    share_before = MAX_TOTAL_CH0 / n
    share_after = MAX_TOTAL_CH0 / K
    power_before = share_before**2
    power_after = share_after**2

    r0, r1, c0, c1, site_x_px = _crop_extent_px(cal)
    crop0 = img0[r0:r1, c0:c1].T
    crop1 = img1[r0:r1, c0:c1].T
    crop2 = img2[r0:r1, c0:c1].T
    vmin, vmax = np.percentile(crop1, 45), np.percentile(crop1, 99.5)

    fig = plt.figure(figsize=(9.6, 6.0))
    lefts, widths = _col_geometry(0.085, 0.93)

    # ---- top row: the three raw images, equal size, evenly spread over the full
    # width -- independent of the time-segment columns used below. Sized to just
    # fit the (very short, letterboxed) images themselves, not the tall ratio-slot
    # the freq/amp rows use, so there is no dead space below them. -----------------
    img_lefts, img_w = _even_cols(0.02, 0.98, 3, 0.05)
    img_row_top = 0.9522
    img_row_h = 0.0833
    row_bottoms, row_heights = _row_geometry(
        img_row_top - img_row_h - 0.025, 0.1162, (1.35, 0.75), 0.0615
    )

    ax_img1 = _image_axis(
        fig,
        (img_lefts[0], img_row_top - img_row_h, img_w, img_row_h),
        crop0,
        vmin,
        vmax,
    )
    ax_img2 = _image_axis(
        fig,
        (img_lefts[1], img_row_top - img_row_h, img_w, img_row_h),
        crop1,
        vmin,
        vmax,
    )
    ax_img3 = _image_axis(
        fig,
        (img_lefts[2], img_row_top - img_row_h, img_w, img_row_h),
        crop2,
        vmin,
        vmax,
    )
    site_y_px = crop0.shape[0] / 2.0

    for i in range(n):
        if mask[i]:
            _site_disc(ax_img2, site_x_px[i], GREEN, y=site_y_px, alternative=True)
        else:
            _site_ring(ax_img2, site_x_px[i], GRAY, ls=":", lw=1.0, y=site_y_px)
    for i in range(n):
        if final_mask[i]:
            _site_disc(ax_img3, site_x_px[i], BLUE, y=site_y_px, alternative=True)
        else:
            _site_ring(ax_img3, site_x_px[i], GRAY, ls=":", lw=1.0, y=site_y_px)

    ax_img1.text(
        0.5,
        1.10,
        "(1)",
        transform=ax_img1.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax_img2.text(
        0.5,
        1.10,
        "(2)",
        transform=ax_img2.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax_img3.text(
        0.5,
        1.10,
        "(3)",
        transform=ax_img3.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # ---- freq + amp rows: four time segments -----------------------------------
    seg_ranges = [t_before, t_just_before, (0.0, t_move_end), (t_move_end, t_hold_end)]
    freq_axes = [
        fig.add_axes((lefts[j], row_bottoms[0], widths[j], row_heights[0]))
        for j in range(4)
    ]
    amp_axes = [
        fig.add_axes((lefts[j], row_bottoms[1], widths[j], row_heights[1]))
        for j in range(4)
    ]

    freq_lo = min(freqs_kt.min(), (cal.freqs_hz[lo : hi + 1] / 1e6).min())
    freq_hi = max(freqs_kt.max(), (cal.freqs_hz[lo : hi + 1] / 1e6).max())
    pad_f = 0.08 * (freq_hi - freq_lo)

    for j, ax in enumerate(freq_axes):
        for i in range(n):
            ax.axhline(cal.freqs_hz[i] / 1e6, color="#e6e6e6", lw=0.8, zorder=0)
        ax.axhspan(
            cal.freqs_hz[lo] / 1e6,
            cal.freqs_hz[hi] / 1e6,
            color=BOX_EDGE,
            alpha=0.10,
            zorder=0,
        )
        lo_t, hi_t = seg_ranges[j]
        if j == 3:
            hi_t = hi_t + 0.16 * (hi_t - lo_t)
        ax.set_xlim(lo_t, hi_t)
        ax.set_ylim(freq_lo - pad_f, freq_hi + pad_f)
        if j > 0:
            ax.set_yticks([])

    for k in range(K):
        col = AMBER if occ_cols[k] in moving_src else GREEN
        freq0 = freqs_kt[k, 0]
        freq_end = freqs_kt[k, -1]
        freq_axes[0].plot(t_before, [freq0, freq0], color=col, lw=1.8, zorder=3)
        freq_axes[1].plot(t_just_before, [freq0, freq0], color=col, lw=1.8, zorder=3)
        freq_axes[2].plot(t_ms, freqs_kt[k], color=col, lw=1.8, zorder=3)
        freq_axes[3].plot(
            [t_move_end, t_hold_end], [freq_end, freq_end], color=col, lw=1.8, zorder=3
        )
        if occ_cols[k] in moving_src:
            freq_axes[2].plot(t_move_end, freq_end, "o", color=BLUE, ms=5, zorder=4)

    for k in range(K):
        col = AMBER if occ_cols[k] in moving_src else GREEN
        freq_axes[3].text(
            t_hold_end + 0.05 * (t_hold_end - t_move_end),
            freqs_kt[k, -1],
            str(occ_cols[k]),
            color=col,
            fontsize=7,
            va="center",
            ha="left",
        )

    freq_axes[0].set_ylabel("frequency [MHz]")
    ax_pos = freq_axes[3].secondary_yaxis("right", functions=(freq_to_pos, pos_to_freq))
    ax_pos.set_ylabel("position [$\\mu$m]")
    for ax in freq_axes:
        ax.set_xticks([])

    for j in range(3):
        (_break_marks if GAPS[j] == GAP_BREAK else lambda a, b: None)(
            freq_axes[j], freq_axes[j + 1]
        )
        (_break_marks if GAPS[j] == GAP_BREAK else lambda a, b: None)(
            amp_axes[j], amp_axes[j + 1]
        )

    for j, ax in enumerate(amp_axes):
        ax.set_xlim(*seg_ranges[j])
        ax.set_ylim(-0.03 * power_after, power_after * 1.2)
        if j > 0:
            ax.set_yticks([])

    for i in range(n):
        if not mask[i]:
            for j in (0, 1):
                amp_axes[j].plot(
                    seg_ranges[j],
                    [power_before, power_before],
                    color=GRAY,
                    lw=1.2,
                    ls="--",
                    zorder=1,
                )
            amp_axes[2].plot(
                [0, 0], [power_before, 0.0], color=GRAY, lw=1.2, ls="--", zorder=1
            )
    for i in range(n):
        if mask[i] and i not in moving_src:
            for j in (0, 1):
                amp_axes[j].plot(
                    seg_ranges[j],
                    [power_before, power_before],
                    color=GREEN,
                    lw=3.2,
                    zorder=2,
                )
            amp_axes[2].plot(
                [0, t_move_end],
                [power_after, power_after],
                color=GREEN,
                lw=3.2,
                zorder=2,
            )
            amp_axes[3].plot(
                [t_move_end, t_hold_end],
                [power_after, power_after],
                color=GREEN,
                lw=3.2,
                zorder=2,
            )
    for i in range(n):
        if mask[i] and i in moving_src:
            for j in (0, 1):
                amp_axes[j].plot(
                    seg_ranges[j],
                    [power_before, power_before],
                    color=AMBER,
                    lw=1.6,
                    zorder=3,
                )
            amp_axes[2].plot(
                [0, 0, t_move_end],
                [power_before, power_after, power_after],
                color=AMBER,
                lw=1.6,
                zorder=3,
            )
            amp_axes[3].plot(
                [t_move_end, t_hold_end],
                [power_after, power_after],
                color=AMBER,
                lw=1.6,
                zorder=3,
            )

    amp_axes[0].set_ylabel("power per trap [a.u.]")
    for ax in amp_axes:
        ax.set_xticks([])
    amp_axes[0].set_xlabel("time")
    amp_axes[0].xaxis.set_label_coords(1.0 + GAP_BREAK / (2 * widths[0]) + 1.75, -0.10)

    # ---- the "1"/"2"/"3" markers, tying the images to the timeline --------------
    for ax in (freq_axes[0], amp_axes[0]):
        ax.axvline(t_marker1, color="k", ls="-", lw=1.1, zorder=5)
    for ax in (freq_axes[1], amp_axes[1]):
        ax.axvline(t_marker2, color="k", ls="-", lw=1.1, zorder=5)
    for ax in (freq_axes[3], amp_axes[3]):
        ax.axvline(t_marker3, color="k", ls="-", lw=1.1, zorder=5)

    _badge(freq_axes[0], t_marker1, freq_hi + pad_f * 0.55, "1")
    _badge(freq_axes[1], t_marker2, freq_hi + pad_f * 0.55, "2")
    _badge(freq_axes[3], t_marker3, freq_hi + pad_f * 0.55, "3")

    fig.align_ylabels([freq_axes[0], amp_axes[0]])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(
        os.path.join(out_dir, "sorting_move_exe_timeline.pdf"), facecolor="white"
    )
    fig.savefig(
        os.path.join(out_dir, "sorting_move_exe_timeline.png"),
        dpi=300,
        facecolor="white",
    )
    print("saved sorting_move_exe_timeline.png / .pdf")

    plt.show()
    return fig


if __name__ == "__main__":
    cal, img0, img1, img2, mask, sites, moves, final_mask = load_illustration_shot()
    plot_move_execution_timeline(cal, img0, img1, img2, mask, sites, moves, final_mask)
