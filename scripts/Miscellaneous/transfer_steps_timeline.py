"""
Timeline variant of transfer_steps.py, in the spirit of
sorting_move_execution_timeline_fig.py: the real camera crops are rotated so
the transport axis runs horizontally and sit in a single row on top, and
underneath the AWG waveform that produced them is shown -- tone positions
(row 1) and tone amplitudes (row 2) -- broken into the five segments of the
Full Transfer Protocol (scripts/Experiments/Full_Transfer_Protocol.py):

    (1) load      -- only the 1064 source tone is on, atom is loaded.
    (2) turn on   -- 1064 destination tone and the 933 auxiliary tone ramp up,
                     the auxiliary sitting on top of the source trap.
    (3) move      -- STA transport of the auxiliary tone, source -> destination.
    (4) turn off  -- auxiliary tone ramps down, handing the atom over.
    (5) hold      -- final configuration, held while the survival image is taken.

The segments live on wildly different timescales (s vs us), so each is its own
axis with break marks in between and its duration written underneath.

Positions are physical (source at 0, destination N*4.6 um away); the lab script
commands each AOD in its own frequency calibration, so the tone frequencies are
written next to the traces instead of on a shared axis.

@author: Bjarne Schuemann
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0

DATA_DIR = r"c:\dev\GitHub\AWGController\Data\tweezerImages1407"
RUN_A = os.path.join(DATA_DIR, "tweezerLoad1x3-FTP-Transfer1_106_20260714-160133.npy")
RUN_B = os.path.join(DATA_DIR, "tweezerLoad1x3-FTP-Transfer1_100_20260714-160059.npy")
LOAD_FRAME, SURV_FRAME = 1, 2
BINNING = 2

# camera geometry, same crop constants as transfer_steps.py
SOURCE_ROW, DEST_ROW, TRAP_COL = 49, 116, 139
COL_HALF = 10
FULL_R0, FULL_R1 = 30, 135

# protocol geometry (Full_Transfer_Protocol.py)
N_SITES = 3
SITE_SPACING_UM = 4.6
SEP_UM = N_SITES * SITE_SPACING_UM
F_1064_START_MHZ = 79.79
F_933_START_MHZ = 91.0
MHZ_PER_SITE = 0.6

# protocol timing / resolutions (Full_Transfer_Protocol.py)
DURATIONS_S = [2.0, 5e-6, 80e-6, 5e-6, 1.5]
RESOLUTIONS = [2, 20, 50, 20, 2]
DURATION_LABELS = ["2 s", "5 $\\mu$s", "80 $\\mu$s", "5 $\\mu$s", "1.5 s"]
STEP_TITLES = [
    "(1) load",
    "(2) turn on",
    "(3) move",
    "(4) turn off",
    "(5) hold",
]

GREEN = "#2f9e58"
AMBER = "#e0972c"
GRAY = "#9a9a9a"
BLUE = "#3477eb"
CMAP = "viridis"

STROKE = [withStroke(linewidth=2.3, foreground="white")]
NUM_BADGE = dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor="k", lw=0.8)

COL_WIDTHS = np.array([0.17, 0.13, 0.30, 0.13, 0.27])
GAP_BREAK = 0.030
GAPS = [GAP_BREAK] * 4

# image panels: (run, frame, segment the badge sits in, where in that segment).
# Only the loading and the survival image are ever taken, so there is no real
# in-flight frame: the atom shows up at the source for steps 1-2 and at the
# destination for step 5.
IMAGE_STEPS = [
    (RUN_A, LOAD_FRAME, 0, 0.50),
    (RUN_A, LOAD_FRAME, 1, 0.65),
    (RUN_A, SURV_FRAME, 4, 0.45),
]


def counts_to_photons(data, binning=BINNING):
    return (np.asarray(data, dtype=np.int32) - 200 * binning**2) * 0.1


def load_frame(path, frame):
    d = np.load(path, allow_pickle=True)[()]
    return counts_to_photons(np.asarray(d["Images"])[frame])


def crop_rotated(img, col=TRAP_COL):
    """Full transport window, transposed so the transport axis is horizontal."""
    return img[FULL_R0:FULL_R1, col - COL_HALF : col + COL_HALF].T


def STA(p0, p1, arr):
    return (p1 - p0) * (10 * arr**3 - 15 * arr**4 + 6 * arr**5) + p0


def build_protocol():
    """Per-segment (t, pos (3, T), amp (3, T)) exactly as Full_Transfer_Protocol.py
    builds them, with positions expressed physically (source at 0)."""
    pos_ch0, pos_ch1 = 0.0, SEP_UM
    amp_nodes = [  # (ch0, ch1, ch2) at the end of each step, step 0 = start
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    aux_nodes = [0.0, 0.0, SEP_UM, SEP_UM, SEP_UM]

    segs = []
    for s, (dur, res) in enumerate(zip(DURATIONS_S, RESOLUTIONS)):
        u = np.linspace(0, 1, res)
        t = u * dur
        prev = amp_nodes[s - 1] if s > 0 else amp_nodes[0]
        cur = amp_nodes[s]
        amp = np.stack(
            [
                np.ones(res) * cur[k] if prev[k] == cur[k] else STA(prev[k], cur[k], u)
                for k in range(3)
            ]
        )
        aux_prev = aux_nodes[s - 1] if s > 0 else aux_nodes[0]
        aux = (
            np.ones(res) * aux_nodes[s]
            if aux_prev == aux_nodes[s]
            else STA(aux_prev, aux_nodes[s], u)
        )
        pos = np.stack([np.ones(res) * pos_ch0, np.ones(res) * pos_ch1, aux])
        segs.append((t, pos, amp))
    return segs


def _even_cols(left, right, n, gap):
    w = (right - left - gap * (n - 1)) / n
    return [left + i * (w + gap) for i in range(n)], w


def _col_geometry(left, right):
    scale = (right - left - sum(GAPS)) / COL_WIDTHS.sum()
    widths = COL_WIDTHS * scale
    lefts = [left]
    for i in range(len(GAPS)):
        lefts.append(lefts[-1] + widths[i] + GAPS[i])
    return lefts, widths


def _row_geometry(top, bottom, ratios, row_gap):
    scale = (top - bottom - row_gap * (len(ratios) - 1)) / sum(ratios)
    bottoms, heights = [], []
    cur = top
    for r in ratios:
        h = r * scale
        bottoms.append(cur - h)
        heights.append(h)
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
    Nc, Nr = crop.shape
    ax = fig.add_axes(rect)
    ax.imshow(
        crop,
        cmap=CMAP,
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


def _atom_ring(ax, x, y, color, ls="-"):
    art = ax.scatter(
        [x],
        [y],
        s=150,
        facecolors="none",
        edgecolors=color,
        linestyles=ls,
        linewidths=1.4,
        zorder=6,
    )
    art.set_path_effects(STROKE)


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


def plot_transfer_timeline():
    segs = build_protocol()
    crops = [crop_rotated(load_frame(run, frame)) for run, frame, _, _ in IMAGE_STEPS]
    vmin, vmax = np.percentile(crops[0], 45), np.percentile(crops[0], 99.5)

    src_x_px = SOURCE_ROW - FULL_R0
    dst_x_px = DEST_ROW - FULL_R0

    fig = plt.figure(figsize=(9.6, 6.0))
    lefts, widths = _col_geometry(0.085, 0.93)

    img_lefts, img_w = _even_cols(0.075, 0.945, len(IMAGE_STEPS), 0.045)
    img_row_top = 0.945
    img_row_h = 0.085
    row_bottoms, row_heights = _row_geometry(
        img_row_top - img_row_h - 0.045, 0.135, (1.35, 0.75), 0.075
    )

    img_axes = [
        _image_axis(
            fig,
            (img_lefts[j], img_row_top - img_row_h, img_w, img_row_h),
            crop,
            vmin,
            vmax,
        )
        for j, crop in enumerate(crops)
    ]
    site_y_px = crops[0].shape[0] / 2.0

    # (ring holding the atom, rings that are on but empty) per image panel
    atom_rings = [
        ((src_x_px, GREEN), []),
        ((src_x_px, AMBER), [dst_x_px]),
        ((dst_x_px, BLUE), [src_x_px]),
    ]
    for j, ax in enumerate(img_axes):
        (x_atom, col), empties = atom_rings[j]
        _atom_ring(ax, x_atom, site_y_px, col)
        for x in empties:
            _atom_ring(ax, x, site_y_px, GRAY, ls=":")
        ax.text(
            0.5,
            1.16,
            f"({IMAGE_STEPS[j][2] + 1})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    pos_axes = [
        fig.add_axes((lefts[j], row_bottoms[0], widths[j], row_heights[0]))
        for j in range(5)
    ]
    amp_axes = [
        fig.add_axes((lefts[j], row_bottoms[1], widths[j], row_heights[1]))
        for j in range(5)
    ]

    pad_p = 0.18 * SEP_UM
    colors = (GREEN, BLUE, AMBER)
    # ch1 dashed so it stays readable where the auxiliary tone lies on top of it
    styles = ("-", (0, (5, 2)), "-")
    for j, (ax_p, ax_a) in enumerate(zip(pos_axes, amp_axes)):
        t, pos, amp = segs[j]
        t_hi = t[-1] if t[-1] > 0 else 1.0
        if j == 4:
            t_hi *= 1.16
        for k in range(3):
            ax_p.axhline(pos[k, 0], color="#e6e6e6", lw=0.8, zorder=0)
        for k in range(3):
            on = amp[k] > 1e-9
            ax_p.plot(
                t[on] if on.any() else t,
                pos[k][on] if on.any() else np.full_like(t, np.nan),
                color=colors[k],
                lw=1.8,
                ls=styles[k],
                zorder=3,
            )
            if not on.all():
                ax_p.plot(t, pos[k], color=GRAY, lw=1.0, ls=":", zorder=2)
            ax_a.plot(
                t,
                amp[k],
                color=colors[k],
                lw=2.6 if k < 2 else 1.8,
                ls=styles[k],
                zorder=3 + k,
            )
        ax_p.set_xlim(0, t_hi)
        ax_p.set_ylim(-pad_p, SEP_UM + pad_p)
        ax_a.set_xlim(0, t[-1] if t[-1] > 0 else 1.0)
        ax_a.set_ylim(-0.08, 1.25)
        ax_p.set_xticks([])
        ax_a.set_xticks([])
        if j > 0:
            ax_p.set_yticks([])
            ax_a.set_yticks([])
        ax_a.text(
            0.5,
            -0.16,
            f"{STEP_TITLES[j]}\n{DURATION_LABELS[j]}",
            transform=ax_a.transAxes,
            fontsize=8,
            ha="center",
            va="top",
        )

    # tone labels inside the last segment, near its right edge, tucked
    # above/below their line (offset by LABEL_DY um) so they clear both the
    # trace and the segment's vertical marker line
    LABEL_DY = -2.5  # 0.5
    tone_labels = [
        (0.0, -LABEL_DY, GREEN, f"Ch0, 1064 nm\n{F_1064_START_MHZ:.2f} MHz", "top"),
        (
            SEP_UM,
            LABEL_DY,
            BLUE,
            f"Ch1, 1064 nm\n{F_1064_START_MHZ + N_SITES * MHZ_PER_SITE:.2f} MHz",
            "bottom",
        ),
    ]
    t_last = segs[4][0][-1]
    for y0, dy, col, txt, va in tone_labels:
        pos_axes[4].text(
            t_last * 0.97, y0 + dy, txt, color=col, fontsize=7, va=va, ha="right"
        )
    pos_axes[2].text(
        segs[2][0][-1] * 0.04,
        0.93 * SEP_UM,
        f"Ch2, 933 nm\n{F_933_START_MHZ:.2f} $\\rightarrow$ "
        f"{F_933_START_MHZ + N_SITES * MHZ_PER_SITE:.2f} MHz",
        color=AMBER,
        fontsize=7,
        va="top",
        ha="left",
    )

    pos_axes[0].set_ylabel("position [$\\mu$m]")
    amp_axes[0].set_ylabel("tone amplitude [a.u.]")
    amp_axes[0].set_xlabel("time")
    amp_axes[0].xaxis.set_label_coords(1.0 + GAP_BREAK / (2 * widths[0]) + 2.6, -0.52)

    for j in range(4):
        _break_marks(pos_axes[j], pos_axes[j + 1])
        _break_marks(amp_axes[j], amp_axes[j + 1])

    for _, _, seg, frac in IMAGE_STEPS:
        t = segs[seg][0]
        t_mark = frac * t[-1] if t[-1] > 0 else frac
        pos_axes[seg].axvline(t_mark, color="k", ls="-", lw=1.1, zorder=5)
        amp_axes[seg].axvline(t_mark, color="k", ls="-", lw=1.1, zorder=5)
        _badge(pos_axes[seg], t_mark, SEP_UM + pad_p * 0.55, str(seg + 1))

    fig.align_ylabels([pos_axes[0], amp_axes[0]])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(
        os.path.join(out_dir, "transfer_steps_timeline.svg"), facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(out_dir, "transfer_steps_timeline.png"), dpi=300, facecolor="white"
    )
    print("saved transfer_steps_timeline.png / .svg")

    plt.show()
    return fig


if __name__ == "__main__":
    plot_transfer_timeline()
