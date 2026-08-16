"""
This figure is used in chapter 3 of my thesis and should visualize how the
moves are executed on the AWG (spectrum card). On the top image the same run
as in sorting_steps_illustration.py should be taken, showing the mask by
encircling the identified atoms and the arrows where they should go towards.
Underneath I wanna show the plan - the frequency/position over time and the
amplitude over time.

Essentially the plot should look like this:
    - 1. Figure on top just the cut out region of interest where all the atoms
         sit. The real image.
    - 2. Right underneath the plan for the sorting moves. On the y-axis I want
         to show the frequency and position, but also the site index. Ideally I
         do not want legends in any of the plots (also 3.). Time axis shared with
         the bottom image.
    - 3. Bottom image; showing the amplitude over time for the sum_amps case.
         Sharing time (x) axis with the plot from 2.. There should not be a legend
         as it should be self explanatory.

Since the plot is very similar to what is already capable in the scripts I can
probably reuse a lot from there. It should just be scientific and in general good
looking for the thesis, therefore I should not include unnecessary legends, and
let the figure do the full story telling itself.

@author: Bjarne Schümann
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

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
    MAX_TOTAL_CH0,
    F_START_HZ,
    UM_PER_MHZ,
)
from sorting_steps_illustration import (
    load_illustration_shot,
    _crop_extent,
    _site_disc,
    _site_ring,
    GREEN,
    AMBER,
    GRAY,
    BLUE,
    BOX_EDGE,
    STROKE,
)

STEP_TIME_S = 80e-6
WPPS = 30
TRAJECTORY = "sta"
LEAD_FRAC = 0.3


def freq_to_pos(f_mhz):
    return (f_mhz * 1e6 - F_START_HZ) / 1e6 * UM_PER_MHZ


def pos_to_freq(p_um):
    return (F_START_HZ + p_um / UM_PER_MHZ * 1e6) / 1e6


def plot_move_execution(cal, img, mask, sites, moves):
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
    t_lead = -LEAD_FRAC * t_ms[-1]
    t_pre = np.concatenate(([t_lead], t_ms))
    pad = 0.05 * (t_ms[-1] - t_lead)

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.6, 0.9], hspace=0.3)
    ax_img = fig.add_subplot(gs[0])
    ax_freq = fig.add_subplot(gs[1])
    ax_amp = fig.add_subplot(gs[2], sharex=ax_freq)

    r0, r1, c0, c1, x0, x1 = _crop_extent(cal)
    crop = img[r0:r1, c0:c1].T
    ax_img.imshow(
        crop,
        cmap="viridis",
        aspect="auto",
        origin="upper",
        vmin=np.percentile(crop, 45),
        vmax=np.percentile(crop, 99.5),
        extent=(x0, x1, 0, 1),
    )

    for i in range(len(mask)):
        if mask[i]:
            _site_disc(ax_img, i, GREEN, y=0.5, alternative=True)
        else:
            _site_ring(ax_img, i, GRAY, ls=":", lw=1.2, y=0.5)

    for sc, dc in moves:
        rad = 0.3 if dc > sc else -0.3
        arrow = FancyArrowPatch(
            (sc, 0.5),
            (dc + 0.05 if dc < sc else dc - 0.05, 0.5),
            connectionstyle="arc3,rad=%.2f" % rad,
            arrowstyle="-|>",
            mutation_scale=15,
            shrinkA=13,
            shrinkB=6,
            color=AMBER,
            lw=0.5,
            zorder=4,
        )
        arrow.set_path_effects(STROKE)
        ax_img.add_patch(arrow)
    ax_img.set_xlim(-0.5, n - 0.5)
    ax_img.set_ylim(0, 1)
    ax_img.set_yticks([])
    ax_img.set_xticks(range(n))
    ax_img.set_xlabel("trap index")

    for i in range(n):
        ax_freq.axhline(cal.freqs_hz[i] / 1e6, color="#e6e6e6", lw=0.8, zorder=0)
        occ = bool(mask[i])
        color = AMBER if (occ and i in moving_src) else GREEN if occ else GRAY
        ax_freq.text(
            t_ms[-1] + pad,
            cal.freqs_hz[i] / 1e6,
            str(i),
            color=color,
            fontsize=7,
            va="center",
        )
    ax_freq.axhspan(
        cal.freqs_hz[lo] / 1e6,
        cal.freqs_hz[hi] / 1e6,
        color=BOX_EDGE,
        alpha=0.10,
        zorder=0,
    )
    for k in range(K):
        color = AMBER if occ_cols[k] in moving_src else GREEN
        freqs_line = np.concatenate(([freqs_kt[k, 0]], freqs_kt[k]))
        ax_freq.plot(t_pre, freqs_line, color=color, lw=1.8, zorder=3)
        if occ_cols[k] in moving_src:
            ax_freq.plot(t_ms[-1], freqs_kt[k, -1], "o", color=BLUE, ms=5, zorder=4)
    ax_freq.set_xlim(t_lead, t_ms[-1] + 4 * pad)
    ax_freq.set_ylabel("frequency [MHz]")
    ax_pos = ax_freq.secondary_yaxis("right", functions=(freq_to_pos, pos_to_freq))
    ax_pos.set_ylabel("position [$\\mu$m]")
    plt.setp(ax_freq.get_xticklabels(), visible=False)

    share_before = MAX_TOTAL_CH0 / n
    share_after = MAX_TOTAL_CH0 / K
    power_before = share_before**2
    power_after = share_after**2
    x_step = [t_lead, 0, 0, t_ms[-1]]
    for i in range(n):
        if not mask[i]:
            ax_amp.plot(
                x_step,
                [power_before, power_before, 0.0, 0.0],
                color=GRAY,
                lw=1.2,
                ls="--",
                zorder=1,
            )
    for i in range(n):
        if mask[i] and i not in moving_src:
            ax_amp.plot(
                x_step,
                [power_before, power_before, power_after, power_after],
                color=GREEN,
                lw=3.2,
                zorder=2,
            )
    for i in range(n):
        if mask[i] and i in moving_src:
            ax_amp.plot(
                x_step,
                [power_before, power_before, power_after, power_after],
                color=AMBER,
                lw=1.6,
                zorder=3,
            )
    ax_amp.set_ylim(-0.03 * power_after, power_after * 1.2)
    ax_amp.set_ylabel("power per trap [a.u.]")
    ax_amp.set_xlabel("time [ms]")

    fig.align_ylabels([ax_freq, ax_amp])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, "sorting_move_exe.pdf"), facecolor="white")
    fig.savefig(
        os.path.join(out_dir, "sorting_move_exe.png"), dpi=300, facecolor="white"
    )
    print("saved sorting_move_exe.png / .pdf")

    plt.show()
    return fig


if __name__ == "__main__":
    cal, img, mask, sites, moves, final_img, final_mask = load_illustration_shot()
    plot_move_execution(cal, img, mask, sites, moves)
