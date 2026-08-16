import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0


RING_TINT = True
C_BLUE = "#1b4f9c"  # blue-shifted
C_GREY = "#3c3c3c"  # unshifted
C_RED = "#c0392b"  # red-shifted
C_SRC = "#d800f0"  # source
C_RING = "#e8543f"  # wavefronts
C_ATOM = "#2f6fb5"
C_TXT = "#2b2b2b"

CMAP_SHIFT = LinearSegmentedColormap.from_list("shift", [C_BLUE, "#9aa3ad", C_RED])


def draw_atom(ax, x, y, r=0.42, zorder=12, mask=True):

    if mask:
        ax.add_patch(
            Circle(
                (x, y),
                r * 1.75,
                facecolor="white",
                alpha=0.80,
                edgecolor="none",
                zorder=zorder - 2,
            )
        )

    for f, a in [(1.50, 0.10), (1.26, 0.16)]:
        ax.add_patch(
            Circle(
                (x, y),
                r * f,
                facecolor=C_ATOM,
                alpha=a,
                edgecolor="none",
                zorder=zorder - 1,
            )
        )

    ax.add_patch(
        Circle((x, y), r, facecolor=C_ATOM, edgecolor="#173963", lw=1.3, zorder=zorder)
    )


def draw_source(ax, x=0.0, y=0.0, r=0.42, zorder=14):

    for f, a in [(3.4, 0.05), (2.6, 0.08), (1.95, 0.13), (1.45, 0.22)]:
        ax.add_patch(
            Circle(
                (x, y),
                r * f,
                facecolor=C_SRC,
                alpha=a,
                edgecolor="none",
                zorder=zorder - 1,
            )
        )

    ax.add_patch(
        Circle(
            (x, y),
            r,
            facecolor="#d800f0",
            edgecolor="#400048",
            lw=1.3,
            zorder=zorder,  # ffd34d, d18f00
        )
    )


def draw_photon(
    ax, angle_deg, r0, r1, n_cycles, amp, color, lw=1.9, zorder=8, head=True
):

    th = np.deg2rad(angle_deg)
    s = np.linspace(0.0, 1.0, 700)
    r = r0 + (r1 - r0) * s
    taper = np.sin(np.pi * s) ** 0.55  # fade the wiggle in and out
    perp = amp * taper * np.sin(2 * np.pi * n_cycles * s)
    x = r * np.cos(th) - perp * np.sin(th)
    y = r * np.sin(th) + perp * np.cos(th)
    ax.plot(x, y, color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    if head:
        ax.annotate(
            "",
            xy=((r1 + 0.42) * np.cos(th), (r1 + 0.42) * np.sin(th)),
            xytext=(r1 * np.cos(th), r1 * np.sin(th)),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color, mutation_scale=13),
            zorder=zorder,
        )


def draw_rings(ax, radii, tint=False):
    for i, rad in enumerate(radii):
        alpha = 0.78 * (1.0 - 0.55 * i / max(len(radii) - 1, 1))

        if not tint:
            ax.add_patch(
                Circle(
                    (0, 0),
                    rad,
                    fill=False,
                    edgecolor=C_RING,
                    lw=1.8,
                    alpha=alpha,
                    zorder=2,
                )
            )

        else:
            th = np.linspace(0, 2 * np.pi, 400)
            pts = np.column_stack([rad * np.cos(th), rad * np.sin(th)])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            # 0 on the left (towards the approaching atom) -> 1 on the right
            frac = 0.5 * (1.0 + np.cos(th[:-1]))
            lc = LineCollection(
                segs, colors=CMAP_SHIFT(frac), lw=1.8, alpha=alpha, zorder=2
            )
            ax.add_collection(lc)


def draw_wave(ax, x0, x1, y, n_cycles, amp, color, lw=1.9, zorder=5):

    t = np.linspace(0, 1, 800)
    x = x0 + (x1 - x0) * t
    ax.plot(
        x,
        y + amp * np.sin(2 * np.pi * n_cycles * t),
        color=color,
        lw=lw,
        zorder=zorder,
        solid_capstyle="round",
    )


BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)  # noqa: C408


fig = plt.figure(figsize=(12.8, 6.3))
gs = GridSpec(
    1,
    2,
    width_ratios=[1.95, 1.0],
    wspace=0.04,
    left=0.01,
    right=0.99,
    top=0.95,
    bottom=0.04,
)


axA = fig.add_subplot(gs[0])
axA.set_xlim(-8.6, 8.6)
axA.set_ylim(-5.5, 5.9)
axA.set_aspect("equal")
axA.axis("off")

draw_rings(axA, np.arange(1.30, 5.05, 0.52), tint=RING_TINT)
draw_source(axA)


# axA.annotate(
#     r"light source, $\omega_L$",
#     xy=(0.34, -0.34),
#     xytext=(-2.75, -4.60),
#     fontsize=13,
#     color=C_TXT,
#     ha="center",
#     va="center",
#     arrowprops=dict(arrowstyle="-", lw=1.0, color="#8a8a8a", shrinkA=6, shrinkB=8),
#     bbox=BBOX,
#     zorder=13,
# )


atoms = [
    (180.0, C_BLUE, 12.0, +1.0, "moving towards\nthe source"),
    (90.0, C_GREY, 8.0, 0.0, "at rest"),
    (0.0, C_RED, 5.0, +1.0, "moving away\nfrom the source"),
]
R_ATOM = 4.35
for ang, col, ncyc, vdx, cap in atoms:
    th = np.deg2rad(ang)
    ax_, ay_ = R_ATOM * np.cos(th), R_ATOM * np.sin(th)

    draw_photon(axA, ang, 1.05, R_ATOM - 0.95, ncyc, 0.30, col)
    draw_atom(axA, ax_, ay_)

    if vdx == 0.0:
        axA.text(
            ax_ + 0.95,
            ay_ + 0.05,
            r"$v = 0$",
            fontsize=13.5,
            color=col,
            ha="left",
            va="center",
            zorder=14,
            bbox=BBOX,
        )
    else:
        y_arr = ay_ + 1.02
        axA.annotate(
            "",
            xy=(ax_ + 0.72 * vdx, y_arr),
            xytext=(ax_ - 0.72 * vdx, y_arr),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col, mutation_scale=16),  # noqa: C408
            zorder=14,
        )
        axA.text(
            ax_,
            y_arr + 0.30,
            r"$\mathbf{v}$",
            fontsize=14.5,
            color=col,
            ha="center",
            va="bottom",
            zorder=14,
        )

    if ang == 90.0:
        axA.text(
            ax_,
            ay_ + 1.02,
            cap,
            fontsize=12.5,
            color=col,
            ha="center",
            va="bottom",
            zorder=14,
            bbox=BBOX,
        )

    else:
        axA.text(
            ax_,
            ay_ - 1.00,
            cap,
            fontsize=12.5,
            color=col,
            ha="center",
            va="top",
            zorder=14,
            linespacing=1.35,
            bbox=BBOX,
        )

axA.text(
    0.005,
    1.0,
    "(a)",
    transform=axA.transAxes,
    fontsize=19,
    fontweight="bold",
    ha="left",
    va="top",
)


axB = fig.add_subplot(gs[1])
axB.set_xlim(0.0, 10.0)
axB.set_ylim(-5.5, 5.9)
axB.axis("off")

rows = [
    (2.75, C_BLUE, 12.0, "blue-shifted", r"$\omega' = \omega_L + kv$"),
    (0.00, C_GREY, 8.0, "unshifted", r"$\omega' = \omega_L$"),
    (-2.75, C_RED, 5.0, "red-shifted", r"$\omega' = \omega_L - kv$"),
]

wx0, wx1 = 0.85, 8.10
for y, col, ncyc, name, eq in rows:
    axB.text(
        wx0,
        y + 1.05,
        name,
        fontsize=13,
        color=col,
        style="italic",
        ha="left",
        va="bottom",
    )
    axB.text(wx1, y + 1.05, eq, fontsize=15, color=col, ha="right", va="bottom")
    draw_wave(axB, wx0, wx1, y, ncyc, 0.58, col)

axB.text(
    0.0,
    1.0,
    "(b)",
    transform=axB.transAxes,
    fontsize=19,
    fontweight="bold",
    ha="left",
    va="top",
)
# axB.text(
#     wx0,
#     5.00,
#     "field in the atom's frame",
#     fontsize=12.5,
#     color="#6a6a6a",
#     ha="left",
#     va="center",
# )

fig.savefig("doppler_shift.pdf", facecolor="white")
fig.savefig("doppler_shift.png", dpi=300, facecolor="white")
print("saved")
