"""
Illustration of the four stages of the single-atom transfer protocol
(source -> auxiliary -> destination tweezer). Left column of every step:
real camera crop from a Full Transfer Protocol run (Data/tweezerImages1407),
with the crop window itself doing the work of showing progress along the
transport axis. Right column: a synthetic camera-like rendering of the
tweezer light (Gaussian spots) with the power configuration at that stage.

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

SOURCE_ROW, DEST_ROW, TRAP_COL = 49, 116, 139
COL_HALF = 10
WIN_H = 46
FULL_R0, FULL_R1 = 30, 135
LIGHT_COL_HALF = round((FULL_R1 - FULL_R0) * COL_HALF / WIN_H)

SIGMA = 3.0
CMAP = "viridis"
SOURCE_AMP, AUX_AMP, DEST_AMP = 1.0, 3.0, 1.0

TITLES = ["1. loading", "2. overlap", "3. move", "4. image"]

EDGE_COLOR = "black"
STROKE = [withStroke(linewidth=2.0, foreground="white")]
MARKER_S = 90
LEFT_MARKER_S = 300
SHOW_MARKERS = False

# y [px] (row), x [px] (col) of the ring drawn on the left image for each step;
# edit these directly to nudge where a step's marker sits.
LEFT_MARKER_ROWS = [49, 49, 83, 117]
LEFT_MARKER_COLS = [140, 140, 140, 140]

# (row, col, kind) of every ring drawn on the right (light) panel, per step;
# edit these directly to nudge where a step's rings sit. kind "empty" is
# dashed, anything else ("atom"/"atom_moving") is a solid ring.
RIGHT_MARKERS = [
    [(49, 139.5, "atom")],
    [(49, 139.5, "atom_moving"), (116, 139.5, "empty")],
    [(49, 139.5, "empty"), (83, 139.5, "atom_moving"), (116, 139.5, "empty")],
    [(49, 139.5, "empty"), (116, 139.5, "atom")],
]

FAKE_YTICK_LABELS = [20, 30, 40, 50]
FAKE_YTICK_FRACS = [(v - 12) / WIN_H for v in FAKE_YTICK_LABELS]


def counts_to_photons(data, binning=BINNING):
    return (np.asarray(data, dtype=np.int32) - 200 * binning**2) * 0.1


def load_frame(path, frame):
    d = np.load(path, allow_pickle=True)[()]
    return counts_to_photons(np.asarray(d["Images"])[frame])


def crop_window(img, atom_row, frac, label_row=None, col=TRAP_COL):
    r0 = round(atom_row - frac * WIN_H)
    r1 = r0 + WIN_H
    c0, c1 = col - COL_HALF, col + COL_HALF
    offset = 0 if label_row is None else label_row - atom_row
    return img[r0:r1, c0:c1], (c0, c1, r1 + offset, r0 + offset)


def gaussian_spot(rows, cols, row0, col0, amp, sigma=SIGMA):
    R, C = np.meshgrid(rows, cols, indexing="ij")
    return amp * np.exp(-((R - row0) ** 2 + (C - col0) ** 2) / (2 * sigma**2))


def draw_marker(ax, row, kind, col=TRAP_COL, size=MARKER_S, drow=0, dcol=0):
    if not SHOW_MARKERS:
        return
    row, col = row + drow, col + dcol
    ls = ":" if kind == "empty" else "-"
    art = ax.scatter(
        [col],
        [row],
        s=size,
        facecolors="none",
        edgecolors=EDGE_COLOR,
        linestyles=ls,
        linewidths=1.3,
        zorder=6,
    )
    art.set_path_effects(STROKE)


def light_panel(spots, col=TRAP_COL):
    rows = np.arange(FULL_R0, FULL_R1)
    c0, c1 = col - LIGHT_COL_HALF, col + LIGHT_COL_HALF
    cols = np.arange(c0, c1)
    img = np.zeros((len(rows), len(cols)))
    for row0, col0, amp in spots:
        img += gaussian_spot(rows, cols, row0, col0, amp)
    return img, (c0, c1, FULL_R1, FULL_R0)


aux3 = SOURCE_ROW + 0.5 * (DEST_ROW - SOURCE_ROW)

STEPS = [
    {
        "image": load_frame(RUN_A, LOAD_FRAME),
        "atom_row": SOURCE_ROW,
        "frac": 0.8,
        "label_row": SOURCE_ROW,
        "spots": [(SOURCE_ROW, TRAP_COL, SOURCE_AMP)],
    },
    {
        "image": load_frame(RUN_A, LOAD_FRAME),
        "atom_row": SOURCE_ROW,
        "frac": 0.8,
        "label_row": SOURCE_ROW,
        "spots": [
            (SOURCE_ROW, TRAP_COL, SOURCE_AMP),
            (SOURCE_ROW, TRAP_COL, AUX_AMP),
            (DEST_ROW, TRAP_COL, DEST_AMP),
        ],
    },
    {
        "image": load_frame(RUN_B, LOAD_FRAME),
        "atom_row": SOURCE_ROW,
        "frac": 0.5,
        "label_row": aux3,
        "spots": [
            (SOURCE_ROW, TRAP_COL, SOURCE_AMP),
            (aux3, TRAP_COL, AUX_AMP),
            (DEST_ROW, TRAP_COL, DEST_AMP),
        ],
    },
    {
        "image": load_frame(RUN_A, SURV_FRAME),
        "atom_row": DEST_ROW,
        "frac": 0.2,
        "label_row": DEST_ROW,
        "spots": [(SOURCE_ROW, TRAP_COL, SOURCE_AMP), (DEST_ROW, TRAP_COL, SOURCE_AMP)],
    },
]

crops = [
    crop_window(s["image"], s["atom_row"], s["frac"], s["label_row"]) for s in STEPS
]
vmin, vmax = np.percentile(crops[0][0], 45), np.percentile(crops[0][0], 99.5)
lmax = SOURCE_AMP + AUX_AMP

fig = plt.figure(figsize=(9, 3))
outer = fig.add_gridspec(1, 4, wspace=0.35)

for k, step in enumerate(STEPS):
    inner = outer[k].subgridspec(1, 2, wspace=0.05)
    ax_img = fig.add_subplot(inner[0])
    ax_light = fig.add_subplot(inner[1])

    crop, ext_img = crops[k]
    light, ext_light = light_panel(step["spots"])

    ax_img.imshow(
        crop,
        cmap=CMAP,
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=ext_img,
    )
    ax_img.invert_yaxis()
    top = ext_img[3]
    ax_img.set_yticks([top + frac * WIN_H for frac in FAKE_YTICK_FRACS])
    ax_img.set_yticklabels([str(v) for v in FAKE_YTICK_LABELS])
    ax_light.imshow(
        light,
        cmap=CMAP,
        aspect="equal",
        origin="upper",
        vmin=0,
        vmax=lmax * 1 / 3,
        extent=ext_light,
    )

    draw_marker(
        ax_img, LEFT_MARKER_ROWS[k], "atom", col=LEFT_MARKER_COLS[k], size=LEFT_MARKER_S
    )
    for row, col, kind in RIGHT_MARKERS[k]:
        draw_marker(ax_light, row, kind, col=col)

    for ax in (ax_img, ax_light):
        ax.tick_params(labelsize=6, length=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    ax_light.set_yticks([])

    ax_img.set_xlabel("x [px]", fontsize=8)
    ax_light.set_xlabel("x [px]", fontsize=8)
    if k == 0:
        ax_img.set_ylabel("y [px]", fontsize=8)

    ax_img.set_title(TITLES[k], fontsize=10, loc="left", x=-0.1)

fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12)

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "transfer_steps.pdf"), facecolor="white")
fig.savefig(os.path.join(out_dir, "transfer_steps.png"), dpi=300, facecolor="white")
print("saved transfer_steps.png / .pdf")

plt.show()
