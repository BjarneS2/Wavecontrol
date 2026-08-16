import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14

W0 = 1.0  # beam waist (focal spot radius)
ZR = 2.5 * W0  # Rayleigh range (sets elongation along z)
Z_EXTENT = 2 * ZR  # how far along z to draw panel (a)
R_EXTENT = 1.5 * W0  # radial extent shown in panel (b)

# Red color scheme (red-detuned trap -> red beam)
RED_CMAP = LinearSegmentedColormap.from_list(
    "red_glow", ["#ffffff", "#ffcccc", "#ff6666", "#cc0000", "#4d0000"]
)
DARK_RED = "#8b0000"
ACCENT_RED = "#e31a1c"


def beam_waist(z, w0=W0, zr=ZR):
    """Gaussian beam radius w(z)."""
    return w0 * np.sqrt(1.0 + (z / zr) ** 2)


def intensity(r, z, w0=W0, zr=ZR):
    """Normalized Gaussian-beam intensity I(r, z), peak value 1 at r=z=0."""
    w = beam_waist(z, w0, zr)
    return (w0 / w) ** 2 * np.exp(-2.0 * r**2 / w**2)


# ----------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(15, 5))
ax_a = fig.add_subplot(1, 3, 1, projection="3d")
ax_b = fig.add_subplot(1, 3, 2)
ax_c = fig.add_subplot(1, 3, 3)

# ========================================================================
# Panel (a): nested iso-intensity surfaces -> "cigar" shaped focal volume
# ========================================================================
theta = np.linspace(0, 2 * np.pi, 60)

# Iso-intensity levels (fraction of peak intensity). Lower levels extend
# further along z (dimmer / larger halo), higher levels are short and fat
# (bright core near the focus) -- exactly like nested contours in the
# original figure.
levels = [0.06, 0.15, 0.32, 0.55, 0.8]
alphas = [0.0, 0.16, 0.24, 0.35, 0.55]
colors = [RED_CMAP(0.25), RED_CMAP(0.4), RED_CMAP(0.6), RED_CMAP(0.8), RED_CMAP(0.97)]

for level, alpha, color in zip(levels, alphas, colors):
    # For iso-intensity level f: r(z)^2 = -0.5 * w(z)^2 * ln(f * (w(z)/w0)^2)
    # valid only where the argument of the log is positive.
    z_max = ZR * np.sqrt(max(1.0 / level - 1.0, 0.0))
    z_max = min(z_max, Z_EXTENT)
    z_line = np.linspace(-z_max, z_max, 80)
    w_line = beam_waist(z_line)
    arg = level * (w_line / W0) ** 2
    arg = np.clip(arg, 1e-12, 1.0)
    r_line = w_line * np.sqrt(-0.5 * np.log(arg))

    # Build surface of revolution around the z-axis
    R, TH = np.meshgrid(r_line, theta, indexing="ij")
    Z = np.meshgrid(z_line, theta, indexing="ij")[0]
    X = R * np.cos(TH)
    Y = R * np.sin(TH)

    ax_a.plot_surface(
        X,
        Y,
        Z,
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
        rstride=2,
        cstride=2,
    )

ax_a.set_xlabel("x [µm]")
ax_a.set_ylabel("y [µm]")
ax_a.set_zlabel("z [µm]")
xy_lim = 1.0
z_lim = Z_EXTENT  # match the axial range actually drawn, instead of an unrelated fixed value
ax_a.set_xlim(-xy_lim, xy_lim)
ax_a.set_ylim(-xy_lim, xy_lim)
ax_a.set_zlim(-z_lim, z_lim)
ax_a.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_a.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_a.set_zticks(np.linspace(-z_lim, z_lim, 9))
ax_a.tick_params(axis="both", labelsize=8)
ax_a.set_box_aspect(
    (1, 1, 2.2)
)  # fixed visual proportions; axis limits (not aspect) carry the true scale
ax_a.view_init(elev=12, azim=-60)
ax_a.set_title("(a)", loc="left", fontweight="bold")

# Clean white background (remove default grey 3D panes)
for pane in (ax_a.xaxis.pane, ax_a.yaxis.pane, ax_a.zaxis.pane):
    pane.set_facecolor("white")
    pane.set_alpha(1.0)
ax_a.grid(True, alpha=0.25)

# ========================================================================
# Panel (b): intensity map in the focal plane + gradient-force arrows
# ========================================================================
n_grid = 300
xy = np.linspace(-R_EXTENT, R_EXTENT, n_grid)
X, Y = np.meshgrid(xy, xy)
Rmap = np.sqrt(X**2 + Y**2)
I_focal = intensity(Rmap, 0.0)

ax_b.imshow(
    I_focal,
    extent=(-R_EXTENT, R_EXTENT, -R_EXTENT, R_EXTENT),
    origin="lower",
    cmap=RED_CMAP,
    vmin=0,
    vmax=1,
)

# Gradient-force arrows: F(r) is proportional to -dU/dr ~ +dI/dr, pointing
# toward the intensity maximum (center) for a red-detuned trap. Its
# magnitude peaks at r ~ w0/2 and vanishes both at the center and far away.
n_arrows = 10
grid1d = np.linspace(-R_EXTENT, R_EXTENT, n_arrows)
Xa, Ya = np.meshgrid(grid1d, grid1d)
Ra = np.sqrt(Xa**2 + Ya**2)
Ra_safe = np.where(Ra == 0, 1e-9, Ra)

# Radial gradient-force magnitude (unnormalized), pointing inward (-r_hat)
force_mag = Ra * np.exp(-2.0 * Ra**2 / W0**2)
Fx = -force_mag * (Xa / Ra_safe)
Fy = -force_mag * (Ya / Ra_safe)

# Skip the very center (force ~ 0 there, arrow undefined) and drop
# vanishingly small arrows far from the trap so the plot stays clean.
mask = (Ra > 0.2 * W0) & (force_mag > 0.03 * force_mag.max())
ax_b.quiver(
    Xa[mask],
    Ya[mask],
    Fx[mask],
    Fy[mask],
    color="black",
    alpha=0.85,
    scale=1.1,
    scale_units="xy",
    width=0.007,
    headwidth=4.5,
    headlength=5.5,
)

ax_b.set_xlabel("x [µm]")
ax_b.set_ylabel("y [µm]")
ax_b.set_xlim(-R_EXTENT, R_EXTENT)
ax_b.set_ylim(-R_EXTENT, R_EXTENT)
ax_b.set_aspect("equal")
ax_b.set_title("(b)", loc="left", fontweight="bold")

# ========================================================================
# Panel (c): trapping potential along a radial cut (solid) vs. the
# harmonic approximation near the trap minimum (dashed)
# ========================================================================
rho = np.linspace(-1.4, 1.4, 400)  # rho / w0

U_true = -np.exp(-2.0 * rho**2)  # true Gaussian potential well
U_harm = -1.0 + 2.0 * rho**2  # harmonic (parabolic) approx.

# Only show the harmonic curve where it stays within the visible range,
# mimicking the original figure (dashed parabola shooting up near the edges)
y_top = 0.5
U_harm_plot = np.where(U_harm <= y_top, U_harm, np.nan)

ax_c.plot(rho, U_true, color=DARK_RED, linewidth=2.5, label="Actual potential")
ax_c.plot(
    rho,
    U_harm_plot,
    color=ACCENT_RED,
    linewidth=2,
    linestyle="--",
    label="Harmonic approximation",
)

ax_c.set_xlim(-1.4, 1.4)
ax_c.set_ylim(-1.2, y_top)
ax_c.set_xlabel(r"$\rho / w_0$", fontsize=14)  # , x=0.55)

# Style like the original: no y-ticks, just an upward arrow labeled "Potential"
ax_c.set_yticks([])
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.spines["left"].set_position(("data", 0))
ax_c.spines["bottom"].set_position(("data", 0))
ax_c.annotate(
    "Potential [a.u.]",
    xy=(0, y_top),
    xytext=(0.05, y_top),
    fontsize=14,
    va="top",
    ha="left",
)
ax_c.plot(0, y_top, marker="^", color="black", markersize=6, clip_on=False)
ax_c.set_title("(c)", loc="left", fontweight="bold")
ax_c.legend(
    loc="lower right",
    bbox_to_anchor=(0.975, -0.15),
    frameon=False,
    fontsize=14,
)

# fig.suptitle(
#    "Red-Detuned Optical Tweezer for Neutral-Atom Trapping",
#    fontsize=13,
#    fontweight="bold",
#    y=1.02,
# )
fig.tight_layout()

fig.savefig("optical_tweezer.png", dpi=200, bbox_inches="tight")
print("Saved optical_tweezer.png")
plt.show()
