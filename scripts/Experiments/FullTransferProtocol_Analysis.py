import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm

# Backbone (mask, photon conversion, masked ROI sums, bimodal threshold) follows
# tweezerAnalysis_3Frames_V3_Load_8x8.py. cv2 is unavailable here, so get_location
# is reproduced with scipy. The traps are invisible in the mean image (dim, ~50%
# loading) but stand out in the shot-to-shot std, which is used as the reference.

DATA_DIR = "C:/dev/GitHub/AWGController/data/tweezerImages1407"
CALIB_PREFIX = "tweezerLoad1x3-FTP-1064Only"  # both 1064 traps -> locate initial+final
RUN_PREFIXES = [
    "tweezerLoad1x3-FTP-Transfer1",
    "tweezerLoad1x3-FTP-Transfer2",
    "tweezerLoad1x3-FTP-Transfer3",
]
SIZE = 10  # ROI / mask window (px)
BINNING = 2  # counts_to_photons offset (200 * BINNING**2 per pixel)
LOC_THRESHOLD = 100
ATOM_THRESHOLD = 45  # photons; set to None to use the bimodal-fit crossing

# Frame layout per shot: 0 = background, 1 = loading (image 2), 2 = survival (image 3)
LOAD, SURV = 1, 2


def load_stack(prefix):
    files = sorted(glob.glob(f"{DATA_DIR}/{prefix}_*.npy"))
    return np.array(
        [np.load(f, allow_pickle=True)[()]["Images"] for f in files]
    ).astype(np.int32)


def counts_to_photons(data, size=BINNING):
    return (data - 200 * size**2) * 0.1


def twoD_Gaussian(xs, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xs
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
    )
    return g.ravel()


def get_location(img, threshold=LOC_THRESHOLD, contourArea=2):
    im = img - np.min(img)
    im = im / np.max(im) * 255
    im = ndimage.gaussian_filter(im, 0.8)
    labels, n = ndimage.label(im > threshold)
    x, y = [], []
    for k in range(1, n + 1):
        if (labels == k).sum() > contourArea:
            cy, cx = ndimage.center_of_mass(labels == k)
            if 5 < cy < im.shape[1] - 5:
                x.append(int(cx))
                y.append(int(cy))
    return np.array([x, y]).T


def get_mask(img, location, size=SIZE):
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    X, Y = np.meshgrid(x, y)
    mask = np.zeros(img.shape, dtype=bool)
    popts = []
    for spot in location:
        ly = int(spot[0] - size / 2)
        lx = int(spot[1] - size / 2)
        sub = img[lx : lx + size, ly : ly + size]
        p0 = (sub.max() - np.median(img), spot[1], spot[0], 3, 3, 0, np.median(img))
        xf, yf = np.meshgrid(x[lx : lx + size], y[ly : ly + size])
        popt, _ = curve_fit(
            twoD_Gaussian, (xf, yf), sub.ravel(order="F"), p0=p0, maxfev=5000
        )
        mask |= (
            twoD_Gaussian((X, Y), *popt).reshape(img.shape, order="F")
            > popt[0] * 0.3 + popt[-1]
        )
        popts.append(popt)
    return np.logical_not(mask), popts


def get_result(img_array, mask, location, size=SIZE):
    counts = np.zeros([len(location), len(img_array)])
    for j, img in enumerate(np.copy(img_array)):
        img[mask] = 0
        for i, b in enumerate(location):
            bx = int(b[0] - size / 2)
            by = int(b[1] - size / 2)
            counts[i, j] = np.sum(img[by : by + size + 1, bx : bx + size + 1])
    return counts


def fit_function(k, A, lamb1, lamb2, sc1, sc2):
    return (1 - A) * norm.pdf(k, lamb1, sc1) + A * norm.pdf(k, lamb2, sc2)


def fit_histogram(hist):
    bins = np.arange(hist.min(), hist.max() + 2, 2) - 0.5
    entries, edges = np.histogram(hist, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak1, peak2 = hist.max() / 6, hist.max() * 0.7
    p0 = [0.5, peak1, peak2, np.sqrt(abs(peak1)) + 1, np.sqrt(abs(peak2)) + 1]
    popt, _ = curve_fit(fit_function, centers, entries, p0=p0, maxfev=10000)
    return popt


def threshold_from_fit(popt):
    A, l1, l2, s1, s2 = popt
    lo, hi = sorted([l1, l2])
    grid = np.linspace(lo, hi, 2000)
    diff = (1 - A) * norm.pdf(grid, l1, s1) - A * norm.pdf(grid, l2, s2)
    idx = np.where(np.diff(np.sign(diff)))[0]
    return grid[idx[0]] if len(idx) else 0.5 * (l1 + l2)


def binom_err(p, n):
    return np.sqrt(p * (1 - p) / n) if n > 0 else 0.0


# --- Reference: locate the two traps and build the mask from the calibration run
calib = load_stack(CALIB_PREFIX)
ref = calib[:, LOAD].std(0) + calib[:, SURV].std(0)
location = get_location(ref)
mask, popts = get_mask(ref, location)

runs = {p: load_stack(p) for p in RUN_PREFIXES}
load_counts = {
    n: get_result(counts_to_photons(s[:, LOAD]), mask, location)
    for n, s in runs.items()
}
surv_counts = {
    n: get_result(counts_to_photons(s[:, SURV]), mask, location)
    for n, s in runs.items()
}

# The atom loads at the initial trap -> larger loading-frame signal there
site_means = [
    np.mean([load_counts[n][i].mean() for n in RUN_PREFIXES])
    for i in range(len(location))
]
initial = int(np.argmax(site_means))
final = int(np.argmin(site_means))

# Threshold from the bimodal fit of the pooled initial-site loading counts
pooled_load = np.concatenate([load_counts[n][initial] for n in RUN_PREFIXES])
fit_params = fit_histogram(pooled_load)
thr = ATOM_THRESHOLD if ATOM_THRESHOLD is not None else threshold_from_fit(fit_params)

results = {}
for name in RUN_PREFIXES:
    ntot = len(runs[name])
    ld, sv = load_counts[name], surv_counts[name]

    loaded = ld[initial] > thr
    at_final = sv[final] > thr
    at_init = sv[initial] > thr

    nl = int(loaded.sum())
    survived = int((loaded & at_final).sum())
    unsuccessful = int((loaded & at_init & ~at_final).sum())
    lost = nl - survived - unsuccessful

    p_load = nl / ntot
    p_surv, p_uns, p_lost = (
        (survived / nl, unsuccessful / nl, lost / nl) if nl else (0, 0, 0)
    )
    results[name] = dict(
        ntot=ntot,
        nl=nl,
        survived=survived,
        unsuccessful=unsuccessful,
        lost=lost,
        p_load=p_load,
        e_load=binom_err(p_load, ntot),
        p_surv=p_surv,
        e_surv=binom_err(p_surv, nl),
        p_uns=p_uns,
        e_uns=binom_err(p_uns, nl),
        p_lost=p_lost,
        e_lost=binom_err(p_lost, nl),
        loaded=loaded,
        at_final=at_final,
        at_init=at_init,
        load_init=ld[initial],
        surv_final=sv[final],
    )

labels = [n.split("-")[-1] for n in RUN_PREFIXES]

# --- Figure 1: detected traps + mask on the full array ------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(ref, cmap="magma")
ax1.set_title("Detected traps (std of 1064 calibration)")
for i, spot in enumerate(location):
    x, y = spot
    tag, col = ("initial", "cyan") if i == initial else ("final", "lime")
    ax1.add_patch(
        patches.Rectangle(
            (x - SIZE / 2, y - SIZE / 2), SIZE, SIZE, ec=col, fc="none", lw=2
        )
    )
    ax1.text(x + SIZE, y, tag, color=col, va="center")

ax2.imshow(calib[:, LOAD].mean(0), cmap="gray")
ax2.imshow(
    np.ma.masked_where(mask, np.ones_like(mask, float)), cmap="autumn", alpha=0.9
)
ax2.set_title("Mask (kept pixels) on full array")
fig.tight_layout()

# --- Figure 2: count histograms with fit + threshold (loading & survival) -----
fig, axes = plt.subplots(
    2, len(RUN_PREFIXES), figsize=(4 * len(RUN_PREFIXES), 6), sharex=True
)
xs = np.linspace(pooled_load.min(), pooled_load.max(), 400)
for j, name in enumerate(RUN_PREFIXES):
    ax = axes[0, j]
    ax.hist(
        load_counts[name][initial], bins=25, density=True, color="steelblue", alpha=0.8
    )
    ax.plot(xs, fit_function(xs, *fit_params), "k")
    ax.axvline(thr, ls="--", c="r")
    ax.set_title(f"{labels[j]}\nloading @ initial")

    ax = axes[1, j]
    ax.hist(
        surv_counts[name][final], bins=25, density=True, color="seagreen", alpha=0.8
    )
    ax.axvline(thr, ls="--", c="r")
    ax.set_title("survival @ final")
    ax.set_xlabel("photon counts")
axes[0, 0].set_ylabel("density")
axes[1, 0].set_ylabel("density")
fig.tight_layout()

# --- Figure 3: classification scatter (loading vs survival) -------------------
# x = initial trap in the loading frame (was an atom loaded?)
# y = final trap in the survival frame (did it arrive?)   -> survived = top-right
fig, axes = plt.subplots(
    1, len(RUN_PREFIXES), figsize=(4 * len(RUN_PREFIXES), 4.2), sharex=True, sharey=True
)
for ax, name in zip(axes, RUN_PREFIXES):
    r = results[name]
    loaded, at_final, at_init = r["loaded"], r["at_final"], r["at_init"]
    xl, yf = r["load_init"], r["surv_final"]
    for mask_c, c, lbl in [
        (~loaded, "k", f"not loaded: {int((~loaded).sum())}"),
        (loaded & at_final, "g", f"survived: {r['survived']}"),
        (loaded & at_init & ~at_final, "orange", f"unsuccessful: {r['unsuccessful']}"),
        (loaded & ~at_final & ~at_init, "r", f"lost: {r['lost']}"),
    ]:
        ax.scatter(xl[mask_c], yf[mask_c], s=12, c=c, alpha=0.7, label=lbl)
    ax.axvline(thr, ls="--", c="grey", lw=0.8)
    ax.axhline(thr, ls="--", c="grey", lw=0.8)
    ax.set_title(name.split("-")[-1])
    ax.set_xlabel("initial counts (loading)")
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("final counts (survival)")
fig.tight_layout()

# --- Figure 4: probabilities across runs --------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
xpos = np.arange(len(RUN_PREFIXES))
ax1.bar(
    xpos,
    [results[n]["p_load"] for n in RUN_PREFIXES],
    yerr=[results[n]["e_load"] for n in RUN_PREFIXES],
    capsize=4,
    color="steelblue",
)
ax1.set_xticks(xpos, labels)
ax1.set_ylabel("loading probability")
ax1.set_ylim(0, 1)
ax1.set_title("Loading")

width = 0.25
for i, (key, ekey, c, lbl) in enumerate(
    [
        ("p_surv", "e_surv", "g", "survived"),
        ("p_uns", "e_uns", "orange", "unsuccessful"),
        ("p_lost", "e_lost", "r", "lost"),
    ]
):
    ax2.bar(
        xpos + (i - 1) * width,
        [results[n][key] for n in RUN_PREFIXES],
        width,
        yerr=[results[n][ekey] for n in RUN_PREFIXES],
        capsize=3,
        color=c,
        label=lbl,
    )
ax2.set_xticks(xpos, labels)
ax2.set_ylabel("probability (given loaded)")
ax2.set_ylim(0, 1)
ax2.set_title("Outcome after transfer")
ax2.legend()
fig.tight_layout()

# --- Summary table ------------------------------------------------------------
print(
    f"\nInitial trap (x,y)={tuple(location[initial])}   Final trap (x,y)={tuple(location[final])}"
)
src = "manual" if ATOM_THRESHOLD is not None else "bimodal-fit crossing"
print(
    f"Atom threshold = {thr:.1f} photons ({src}; fit peaks: no-atom {fit_params[1]:.0f}, atom {fit_params[2]:.0f})\n"
)
hdr = f"{'Run':<12}{'N':>5}{'loaded':>7}{'loading':>17}{'survived':>17}{'unsuccessful':>17}{'lost':>17}"
print(hdr)
print("-" * len(hdr))
for n in RUN_PREFIXES:
    r = results[n]
    print(
        f"{n.split('-')[-1]:<12}{r['ntot']:>5}{r['nl']:>7}"
        f"{r['p_load']:>9.3f}+/-{r['e_load']:<5.3f}"
        f"{r['p_surv']:>9.3f}+/-{r['e_surv']:<5.3f}"
        f"{r['p_uns']:>9.3f}+/-{r['e_uns']:<5.3f}"
        f"{r['p_lost']:>9.3f}+/-{r['e_lost']:<5.3f}"
    )

plt.show()
