"""
sorter.py
Self-contained, camera-driven 1D tweezer sorter as a class (Sorter1D).

Instantiate once from a folder / glob of calibration images (learns the trap
layout, count mask and occupancy threshold). Then hand a single in-memory image
to sort(img): it detects occupancy, plans the rearrangement, fires the moves on
an internal trigger (no EXT0, no file round-trip), holds the sorted block for
revert_delay_s, and returns to the original 1xN array so the sequence can loop.

Ch0 carries the N-tone tweezer array that gets rearranged; Ch1 holds a single,
fixed, centered tone that stays on through setup, sort and revert. The sort goes
through move_2d so Ch1 is never dropped (a single-channel move resets all cores).

The analysis functions below are copied verbatim from image_analysis.py so this
file has no dependency on it; only the hardware layer comes from Controller.py.

@author: Bjarne Schümann
"""

import os
import sys
import time
import glob
import json
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import curve_fit
from scipy.stats import norm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
)  # src/: Controller

try:
    from Controller import AWGController, convert_position_to_freq
except Exception:  # AWG SDK absent -> calibration / planning still work
    print("Something fails with the import here.")
    AWGController = None

    def convert_position_to_freq(position_arr, f_start_hz=91.0e6, um_per_MHz=4.6 / 0.6):
        position_arr = np.asarray(position_arr, dtype=float)
        if position_arr.ndim == 1:
            position_arr = position_arr[np.newaxis, :]
        return f_start_hz + (position_arr / um_per_MHz) * 1e6


@lru_cache(maxsize=None)
def schroeder_generalized(M):  # IMD-suppressing phase prescription
    if M < 2:
        return np.zeros(max(M, 1))
    n = np.arange(M)
    return np.pi * n * (n - 1) / M


# Crest-factor-optimised phases from optimize_phases.py, keyed by tone count. Used ONLY for
# the static array (initial arm + every revert); sort moves stay on Schroeder because K
# varies per shot and the tones chirp, so a fixed optimum does not apply there.
#
# Lower peak -> larger amplitudes for the same headroom -> more power: at n=10 these give
# peak 3.697 (CF 1.653) against Schroeder's 4.299 (CF 1.922), i.e. sum(a^2) 0.732 vs 0.541,
# about 1.35x the RF power. See the derivation block in optimize_phases.py.
#
# The peak is invariant to f_start and to the spacing for an equally spaced set, so snapping
# to the DDS grid does NOT invalidate these. They ARE tied to the amplitude ratios, so re-run
# optimize_phases.py with WEIGHTS set if you hand-tune amplitudes_ch0. Any tone count not
# listed here silently falls back to Schroeder.
try:
    from optimized_phases import OPTIMIZED_STATIC_PHASES
except ImportError:  # table absent -> every tone count silently falls back to Schroeder
    print("[sorter] optimized_phases.py not found - using Schroeder phases everywhere.")
    OPTIMIZED_STATIC_PHASES = {}


def static_phases(n):
    """Optimised phases for an n-tone static array if we have them, else Schroeder."""
    n = int(n)
    ph = OPTIMIZED_STATIC_PHASES.get(n)
    if ph is None:
        return schroeder_generalized(n)
    ph = np.asarray(ph, dtype=float)
    if ph.shape != (n,):
        raise ValueError(
            f"OPTIMIZED_STATIC_PHASES[{n}] must have length {n} (got {ph.size})"
        )
    return ph


def peak_of_sum(w, freqs_hz, phases, n=8192):
    """Peak of sum_k w_k sin(2 pi f_k t + phi_k), in units of one tone at w=1.

    Analytic envelope |sum_k w_k exp(i(...))|: upper-bounds |signal|, depends only on
    difference frequencies so it is slowly varying. Same function as in hold_array.py /
    optimize_phases.py -- keep the three in step."""
    w = np.asarray(w, dtype=float)
    f = np.asarray(freqs_hz, dtype=float)
    p = np.asarray(phases, dtype=float)
    d = np.diff(np.unique(f))
    t = np.linspace(0.0, 1.0 / (d.min() if d.size else f[0]), n, endpoint=False)
    env = (w[:, None] * np.exp(1j * (2 * np.pi * np.outer(f, t) + p[:, None]))).sum(0)
    return float(np.abs(env).max())


def scale_static_amps(w, freqs_hz, phases, headroom=None):
    """Scale the STATIC array's tone weights (ratios kept) up to the largest amplitudes
    that do not clip the DAC.

    Power goes as sum(a_k^2), and the binding limit is the composite PEAK, not sum(a_k).
    With a known phase set the peak is ~1.35*sqrt(K)*a rather than K*a, so this delivers
    roughly half the single-tone power at any K, instead of the 1/K that a_k = 1/n gives.
    See the derivation block in optimize_phases.py.

    Only valid where the phases are known and the tones are stationary, i.e. the static
    array. Sort moves must NOT use this: the tones chirp, their relative phases drift into
    full alignment, and the only rigorous bound there is a_k <= headroom/K -- which is what
    _ch0_amps()'s max_total_ch0/n_full already provides."""
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise ValueError("static amplitude weights must be non-negative")
    hr = STATIC_HEADROOM if headroom is None else float(headroom)
    pk = peak_of_sum(w, freqs_hz, phases)
    return w * (hr / pk)


# --- configuration defaults -------------------------------------------------
# Array frequency convention: f_start_hz is the frequency at position 0.
#   center=False (default): f_start_hz IS the first (lowest-freq / leftmost) site;
#                           the array steps upward by spacing_um.
#   center=True           : the array is symmetric about f_start_hz.
SPACING_UM = 4.6  # site separation [um]
CENTER_ARRAY = False  # False -> f_start = first site; True -> symmetric about f_start
UM_PER_MHZ = 4.6 / 0.6

STEP_TIME_S = (
    60e-6  # transport speed: time for one single-site hop (4.6um/40us = 115 um/ms)
)
WAYPOINTS_PER_STEP = 50  # trajectory samples per single-site step
# Adiabatic amplitude ramp bracketing every sort move. The card ramps amplitude in HARDWARE
# via amp_slope, so a whole ramp is ONE segment, not one per waypoint: the cost is 2 extra
# segments for the entire move. AMP_RAMP_S = 0 disables it. AMP_RAMP_TOP is the per-tone
# amplitude at the two ends of the ramp; None -> the level the static arm runs at, so the
# depth the atoms feel is continuous from loading, through transport, and back again.
# Because _arm_and_stream parks on amps_kt[:, -1], this is also the depth the card HOLDS
# while the survival image is taken.
# AMP_RAMP_OUT chooses what happens at the END of the move, i.e. the depth the survival
# image is taken in:
#   True  -> ramp back to AMP_RAMP_TOP. The park holds the loading depth, so image 3 is
#            taken in exactly the traps the atoms loaded into (like-for-like survival).
#   False -> no closing ramp. The park holds the MOVE level, so image 3 is taken in the
#            deeper sorted traps. The last waypoint pair then becomes a plain hold of
#            AMP_RAMP_S at the move level (no extra DDS commands, _changed_mask skips it),
#            and revert() is what brings the depth back down afterwards.
# Ignored when AMP_RAMP_S = 0 (then both ends are hard steps anyway).
AMP_RAMP_S = 0.0
AMP_RAMP_TOP = None
AMP_RAMP_OUT = False
# Waypoints per amplitude ramp. 1 = the card does one constant amp_slope, i.e. a straight
# line (the old behaviour). N > 1 samples a min-jerk envelope at N anchors, so the ramp is
# built from N linear segments approximating the same profile the positions use - zero slope
# at both ends, no kink. Costs N-1 extra segments per ramp; keep AMP_RAMP_S / N above ~1 us
# so each ramp segment stays well clear of the command-execution floor.
AMP_RAMP_WAYPOINTS = 20

# How the absolute per-tone amplitude during a MOVE is chosen. _ch0_amps() only ever supplies
# the RATIOS between tones; this picks the scale. K = tones actually on, n = full array.
#   "match_static" : the static arm's per-tone level -> trap depth constant from loading,
#                    through transport, to the hold. The amplitude ramp becomes a no-op.
#   "sum_amp"      : max_total_ch0 / sum(w), the sum-amplitude bound over the K tones on.
#   "equal_power"  : same TOTAL RF power as the static arm (sum(a^2) preserved), so fewer
#                    tones means more power each - deeper than loading at small K.
#   "explicit"     : MOVE_AMP_VALUE, taken literally.
# _guard_amps (sum_k a_k <= max_total_ch0) is applied in EVERY mode. That sum is the rigorous
# worst-case peak of the summed waveform, so exceeding it clips the DAC -- and hard clipping on
# K carriers produces a forest of intermodulation products. Measured on an 11-site array with
# K=6: "match_static" wants a_k = 0.2715, sum = 1.63, envelope peak 1.38 = 138% of full scale.
# The modes may therefore only LOWER the amplitude; the clamp prints when it bites, which is
# how you learn that a given mode is unreachable at your K.
MOVE_AMP_MODE = "sum_amp"
MOVE_AMP_VALUE = None
TRAJECTORY = "sta"  # single-site profile: "sta" (min-jerk) or "linear"

CALIB_PREFIX = "tweezerLoad1x11_"  # calibration files live under <folder>/<prefix>*.npy
CONFIG_NAME = "sorter_config.json"  # sidecar config saved in the calibration folder
CALIB_CACHE = (
    "sorter_calibration.npz"  # cached learned calibration (for instant reload)
)

SERIAL = 24909
F_START_HZ = 91.0e6
F_MIN_HZ = 70.0e6
F_MAX_HZ = 130.0e6
X_FREQ_HZ = 91.0e6  # Ch1 single static tone (centered in the AOD band)
MAX_AMPV = 1.2  # CH0 full-scale [V]
MAX_AMPV_CH1 = 1.2  # CH1 full-scale [V]  (independent AOD)
MAX_TOTAL_CH0 = 1  # CH0 summed-fraction budget (the MOVE budget: a_k = this / n_full)
MAX_TOTAL_CH1 = 1  # CH1 summed-fraction budget
STATIC_HEADROOM = 1.0
CORE_MAPPING = "20/1"  # cores on CH0 / CH1: up to 20 tweezers + 1 x-tone

# --- image analysis ---------------------------------------------------------
N_SITES = 11  # expected sites for the cross-check; None -> infer from detection
LOAD_FRAME = 1  # 0 background, 1 loading (used for occupancy), 2 survival
BINNING = 2
BORDER_PX = 8  # K outermost pixels ignored when locating (edge dark counts)
ROI_SIZE = 10  # ROI box [px] for the Gaussian fit and the counting
DETECT_SMOOTH = 1.0  # gaussian pre-smoothing for detection [px]
DETECT_FRAC = 0.4  # detection threshold as a fraction of the brightest pixel
MIN_AREA = 3  # min connected pixels for a valid spot
GAUSS_FRAC = 0.3  # active mask pixels: gaussian > GAUSS_FRAC * A + offset
SPACING_TOL = 0.30  # max fractional deviation of inter-site spacing before warning
FIRST_SITE = "top"  # "top" | "bottom" | "left" | "right"
ATOM_THRESHOLD = None  # None -> auto double-Gaussian threshold; float -> override

# DIAGNOSTICS ONLY. None -> sort_live()/sort() use the camera occupancy (normal operation).
# A list, e.g. [0,1,0,1,0,0,0,0,0,0,0], overrides the image and forces that mask on every
# shot - the one-atom / two-atom control experiments. Leave it None for real runs.
FORCE_MASK = None


# ===========================================================================
# Assignment / trajectory (copied verbatim from image_analysis.py)
# ===========================================================================


def find_best_window(mask, start_window=None):
    """
    Find the best target window for the occupancy mask by:
    Assigning each atom an index for a window starting at "start" and the ith atom
    sitting at start+i, so it is already in the correct place iff occupied[i]-i == start
    Then just look for the windows in which most atoms are left untouched -> fewest moves.

    start_window: if None, pick the move-minimizing window as above. If an int, force the
    block to start at that index (positions before it are left empty) and fill from there.

    start: index of the first trap of best window.
    sites: (K,) target trap indices (start .. start+K-1).
    in_place: #atoms already in right place
    """
    mask = np.asarray(mask, dtype=bool)
    N = len(mask)
    occupied = np.flatnonzero(mask)  # current trap index of each atom, in order
    K = occupied.size
    if K == 0:
        return 0, np.empty(0, dtype=int), 0

    lo, hi = 0, N - K  # valid window starts keep the block in range
    if start_window is not None:
        start = int(np.clip(start_window, lo, hi))
        sites = start + np.arange(K)
        in_place = int(np.count_nonzero(occupied == sites))
        return start, sites, in_place

    disp = occupied - np.arange(K)
    disp = np.clip(disp, lo, hi)
    starts, counts = np.unique(disp, return_counts=True)
    best = int(np.argmax(counts))
    start = int(starts[best])
    in_place = int(counts[best])

    sites = start + np.arange(K)
    return start, sites, in_place


def plan_moves(mask, start_window=None):
    """
    Returns [(from_trap, to_trap),...] for atoms that need to move only.
    """
    _, sites, _ = find_best_window(mask, start_window=start_window)
    occupied = np.flatnonzero(np.asarray(mask, dtype=bool))
    moves = [(int(src), int(dst)) for src, dst in zip(occupied, sites) if src != dst]
    right = sorted((m for m in moves if m[1] > m[0]), key=lambda m: -m[0])
    left = sorted((m for m in moves if m[1] < m[0]), key=lambda m: m[0])
    return sites, left + right


def site_positions(n, spacing_um, center=True):
    x = np.arange(n) * float(spacing_um)
    if center:
        x = x - x.mean()
    return x


def snap_freq_to_grid(f_hz, f_step_hz):
    """Nearest settable DDS grid frequency [Hz] to f_hz."""
    return round(float(f_hz) / f_step_hz) * f_step_hz


def snap_spacing_to_grid(spacing_um, f_step_hz, um_per_MHz=UM_PER_MHZ):
    """Nearest site spacing [um] whose frequency step is an EXACT integer multiple of the
    card's DDS grid step (f_step_hz = SPC_DDS_AVAIL_FREQ_STEP). Off-grid spacings make each
    tone round to a slightly different grid point, so pairwise spacings differ and the array
    beats slowly in power; snapping the spacing to the grid removes that residual."""
    df_req = spacing_um / um_per_MHz * 1e6
    n_steps = max(1, round(df_req / f_step_hz))
    df_hz = n_steps * f_step_hz
    return df_hz / 1e6 * um_per_MHz


def grid_site_positions(n, spacing_um, f_step_hz, um_per_MHz=UM_PER_MHZ, center=True):
    """Drop-in alternative to site_positions() that snaps the spacing to the card grid so
    all tone frequencies land exactly equally spaced (no off-grid beating). Pair it with an
    on-grid start frequency (snap_freq_to_grid) so absolute frequencies are on-grid too."""
    return site_positions(
        n, snap_spacing_to_grid(spacing_um, f_step_hz, um_per_MHz), center
    )


def min_jerk(p0, p1, n):
    """compute min jerk trajectory from p0->p1 in n steps"""
    s = np.linspace(0.0, 1.0, n)
    return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def linear_ramp(p0, p1, n):
    """straight-line (constant-velocity) trajectory from p0->p1 in n steps"""
    return np.linspace(p0, p1, n)


def _profile_fn(trajectory):
    """map a trajectory name to its single-site profile function."""
    return linear_ramp if str(trajectory).lower() == "linear" else min_jerk


def build_sort_trajectory(
    mask,
    positions_um,
    moves,
    step_time_s=STEP_TIME_S,
    wpps=WAYPOINTS_PER_STEP,
    wpps_amp=AMP_RAMP_WAYPOINTS,
    trajectory=TRAJECTORY,
    ramp_s=0.0,
):
    """
    Turn the ordered moves into one (K, T) trajectory on a single axis.
    Each move slides one atom straight from its source site to its target site
    with a minimum-jerk profile. The trajectories are built sequentially, so the
    first move starts from the initial state, the second from the state after the
    first move, and so on. Empty sites are turned OFF at the beginning.
    I think I could tehcnically also do 2 moves at the same time, but here it is
    sequentially just to be save.

    Returns (t, P, occ_cols):
        t        : (T,) waypoint times [s]
        P        : (K, T) per-tone positions [um]; row k tracks one physical atom
        occ_cols : the initially occupied columns, in tone order (ascending column)
    """
    mask = np.asarray(mask).astype(bool)
    occ_cols = [c for c in range(len(mask)) if mask[c]]
    K = len(occ_cols)
    if K == 0:
        raise ValueError("empty mask: no atoms loaded")

    tone_at_col = {occ_cols[i]: i for i in range(K)}  # column -> tone

    profile = _profile_fn(trajectory)
    t = [0.0]
    P = [[positions_um[occ_cols[i]]] for i in range(K)]

    for sc, dc in moves:
        tone = tone_at_col.pop(sc)
        n_steps = abs(dc - sc)
        if n_steps == 0:
            continue
        n = max(2, n_steps * wpps)
        dur = n_steps * step_time_s
        seg_t = np.linspace(t[-1], t[-1] + dur, n)[1:]  # drop dup start
        seg_x = profile(positions_um[sc], positions_um[dc], n)[1:]
        for j in range(K):
            if j == tone:
                P[j].extend(seg_x.tolist())
            else:
                P[j].extend([P[j][-1]] * len(seg_t))
        t.extend(seg_t.tolist())
        tone_at_col[dc] = tone

    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    if ramp_s and ramp_s > 0.0:
        # AMP_RAMP_WAYPOINTS extra waypoints at each end with the positions HELD. The
        # amplitude envelope applied in _move_setpoints then turns those segments into the
        # depth ramp, produced by the card's own amp_slope. With N = 1 this is one segment
        # and one constant slope (a straight line); with N > 1 the same total ramp_s is
        # split into N equal segments that trace a min-jerk envelope.
        nr = max(1, int(wpps_amp))
        r = float(ramp_s)
        lead = np.linspace(0.0, r, nr, endpoint=False)  # 0 .. r-r/nr
        tail = t[-1] + r + np.linspace(r / nr, r, nr)  # ends at t[-1]+2r
        t = np.concatenate((lead, t + r, tail))
        P = np.concatenate(
            (np.repeat(P[:, :1], nr, axis=1), P, np.repeat(P[:, -1:], nr, axis=1)),
            axis=1,
        )
    return t, P, occ_cols


# ===========================================================================
# Image loading + calibration (copied verbatim from image_analysis.py)
# ===========================================================================


def _counts_to_photons(data, binning=BINNING):
    return (np.asarray(data, dtype=np.int32) - 200 * binning**2) * 0.1


def _twoD_gaussian(coords, amp, x0, y0, sx, sy, theta, offset):
    x, y = coords
    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = -np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    g = offset + amp * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )
    return g.ravel()


def _double_gaussian(k, A, m1, s1, m2, s2):
    return (1 - A) * norm.pdf(k, m1, s1) + A * norm.pdf(k, m2, s2)


def _resolve_glob(pattern):
    """Return a glob that actually matches something.

    CALIB_PREFIX has to equal the GUI's 'Series pathname' basename character for
    character, and that mismatch (1x8 vs 1x9, '_' vs '-') has already cost us runs.
    If the prefixed pattern matches nothing but the folder does hold .npy files, fall
    back to <folder>/*.npy and say loudly what was actually used."""
    if glob.glob(pattern):
        return pattern
    folder = os.path.dirname(pattern) or "."
    alt = os.path.join(folder, "*.npy")
    files = glob.glob(alt)
    if not files:
        raise FileNotFoundError(
            "no images match %s, and no .npy files in %s at all" % (pattern, folder)
        )
    seen = sorted({os.path.basename(f).rsplit("_", 2)[0] for f in files})
    print(
        "[sorter] no match for %s\n"
        "         falling back to %s (%d files; prefixes present: %s)"
        % (pattern, alt, len(files), ", ".join(seen[:4]))
    )
    return alt


def load_frames(glob_pattern, frame=LOAD_FRAME, binning=BINNING):
    """(M, H, W) photon-converted stack of one frame index over all matching .npy files."""
    files = sorted(glob.glob(_resolve_glob(glob_pattern)))
    if not files:
        raise FileNotFoundError("no images match: %s" % glob_pattern)
    imgs = [
        np.asarray(np.load(f, allow_pickle=True)[()]["Images"])[frame] for f in files
    ]
    return _counts_to_photons(np.asarray(imgs), binning)


def load_shot(glob_pattern, index=0, frame=LOAD_FRAME, binning=BINNING):
    """Single photon-converted frame from one file (one experimental shot)."""
    files = sorted(glob.glob(_resolve_glob(glob_pattern)))
    if not files:
        raise FileNotFoundError("no images match: %s" % glob_pattern)
    d = np.load(files[index], allow_pickle=True)[()]
    return _counts_to_photons(np.asarray(d["Images"])[frame], binning)


def array_axis(pts, first_site=FIRST_SITE):
    """Unit vector along the trap array, with a PINNED sign convention.

    np.linalg.svd has no sign convention: the singular vector it returns can flip on a
    sub-pixel change in one centroid, which silently reverses the whole site ordering
    between calibration runs. Pinning the sign here makes the ordering reproducible.

    pts are (x, y) = (column, row). Image y grows DOWNWARD, so "top" means the site
    with the smallest row index sorts first.
    """
    p = np.asarray(pts, dtype=float)
    c = p - p.mean(0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    comp, sign = {
        "top": (1, 1.0),
        "bottom": (1, -1.0),
        "left": (0, 1.0),
        "right": (0, -1.0),
    }[str(first_site).lower()]
    if abs(u[comp]) < 0.2:
        print(
            "  WARNING: array axis is nearly perpendicular to the %r direction "
            "(|u|=%.3f) - FIRST_SITE is ambiguous, pick the other axis"
            % (first_site, abs(u[comp]))
        )
    return u if sign * u[comp] >= 0 else -u


def _check_sites(pts, n_expected, spacing_tol):
    n = len(pts)
    print(
        "  found %d sites%s"
        % (n, "" if n_expected is None else " (expected %d)" % n_expected)
    )
    if n_expected is not None and n != n_expected:
        print("  WARNING: site count %d != expected %d" % (n, n_expected))
    if n >= 3:
        d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        med = np.median(d)
        bad = np.where(np.abs(d - med) > spacing_tol * med)[0]
        print("  inter-site spacing [px]: %s  (median %.1f)" % (np.round(d, 1), med))
        if len(bad):
            print(
                "  WARNING: uneven spacing at gaps %s - possible outlier / missed site"
                % bad.tolist()
            )


def locate_sites(
    mean_img,
    n_expected=N_SITES,
    border_px=BORDER_PX,
    row_band=None,
    smooth=DETECT_SMOOTH,
    frac=DETECT_FRAC,
    min_area=MIN_AREA,
    spacing_tol=SPACING_TOL,
    first_site=FIRST_SITE,
):
    """Spot centroids (x, y) in trap-index order. Ignores the border_px outermost
    pixels (edge dark counts); cross-checks the count and the inter-site spacing."""
    H, W = mean_img.shape
    work = gaussian_filter(mean_img - np.median(mean_img), smooth)
    valid = np.zeros((H, W), bool)
    r0, r1 = (0, H) if row_band is None else row_band
    valid[max(r0, border_px) : min(r1, H - border_px), border_px : W - border_px] = True
    binimg = (work > work[valid].max() * frac) & valid
    lbl, n = label(binimg)  # type: ignore
    pts = []
    for i in range(1, n + 1):
        m = lbl == i
        if m.sum() < min_area:
            continue
        ys, xs = np.nonzero(m)
        w = work[m]
        pts.append((np.average(xs, weights=w), np.average(ys, weights=w)))
    if not pts:
        raise RuntimeError("no spots detected - lower DETECT_FRAC or check the image")

    pts = np.array(pts)
    u = array_axis(pts, first_site)
    pts = pts[np.argsort((pts - pts.mean(0)) @ u)]
    print(
        "  array axis (x,y) = (%+.3f, %+.3f); site 0 is the %s-most spot"
        % (u[0], u[1], first_site)
    )
    _check_sites(pts, n_expected, spacing_tol)
    return pts


def fit_site_masks(mean_img, locations, roi_size=ROI_SIZE, gauss_frac=GAUSS_FRAC):
    """Per-site 2D-Gaussian fit -> union 'active' pixel mask (gaussian > frac*A + offset)."""
    H, W = mean_img.shape
    YY, XX = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    active = np.zeros((H, W), bool)
    popts = []
    for xc, yc in locations:
        c0 = int(round(xc - roi_size / 2))
        r0 = int(round(yc - roi_size / 2))
        c0 = max(0, min(c0, W - roi_size))
        r0 = max(0, min(r0, H - roi_size))
        sub = mean_img[r0 : r0 + roi_size, c0 : c0 + roi_size]
        rr, cc = np.meshgrid(
            np.arange(r0, r0 + roi_size), np.arange(c0, c0 + roi_size), indexing="ij"
        )
        p0 = (sub.max() - np.median(sub), xc, yc, 2.0, 2.0, 0.0, np.median(sub))
        popt, _ = curve_fit(_twoD_gaussian, (cc, rr), sub.ravel(), p0=p0, maxfev=10000)
        popts.append(popt)
        g = _twoD_gaussian((XX, YY), *popt).reshape(H, W)
        active |= g > gauss_frac * popt[0] + popt[-1]
    return active, popts


def site_counts(img, locations, active, roi_size=ROI_SIZE):
    """Masked ROI pixel-sum at each site (pixels outside `active` are zeroed)."""
    H, W = img.shape
    masked = np.where(active, img, 0.0)
    out = np.empty(len(locations))
    for i, (xc, yc) in enumerate(locations):
        c0 = int(round(xc - roi_size / 2))
        r0 = int(round(yc - roi_size / 2))
        c0 = max(0, min(c0, W - roi_size))
        r0 = max(0, min(r0, H - roi_size))
        out[i] = masked[r0 : r0 + roi_size + 1, c0 : c0 + roi_size + 1].sum()
    return out


def auto_threshold(counts_flat):
    """Threshold at the valley between the two Gaussians fitted to the count histogram."""
    flat = np.asarray(counts_flat, dtype=float)
    lo, hi = flat.min(), flat.max()
    bins = np.linspace(lo, hi, max(12, int(np.sqrt(flat.size))))
    ent, edg = np.histogram(flat, bins=bins, density=True)
    ctr = 0.5 * (edg[1:] + edg[:-1])
    p0 = [
        0.5,
        lo + 0.2 * (hi - lo),
        0.1 * (hi - lo),
        lo + 0.75 * (hi - lo),
        0.1 * (hi - lo),
    ]
    popt, _ = curve_fit(_double_gaussian, ctr, ent, p0=p0, maxfev=10000)
    m1, m2 = sorted((popt[1], popt[3]))
    ks = np.linspace(m1, m2, 1000)
    return float(ks[np.argmin(_double_gaussian(ks, *popt))]), popt


class Calibration:
    """Trap layout learned from the calibration images: site locations in
    trap-index order, the Gaussian count mask, the occupancy threshold, and the
    site positions/frequencies. occupancy(img) turns one shot into a binary mask."""

    def __init__(
        self,
        mean_img,
        locations,
        active,
        threshold,
        popts,
        spacing_um=SPACING_UM,
        center=CENTER_ARRAY,
        f_start_hz=F_START_HZ,
        um_per_MHz=UM_PER_MHZ,
    ):
        self.mean_img = mean_img
        self.locations = np.asarray(locations, dtype=float)
        self.active = active
        self.threshold = float(threshold)
        self.popts = popts
        self.n = len(self.locations)
        self.positions_um = site_positions(self.n, spacing_um, center)
        self.freqs_hz = convert_position_to_freq(
            self.positions_um, f_start_hz=f_start_hz, um_per_MHz=um_per_MHz
        )[0]

    def counts(self, img):
        return site_counts(img, self.locations, self.active)

    def occupancy(self, img):
        return (self.counts(img) > self.threshold).astype(int)

    def save(self, path):
        """Cache the learned calibration so it can be reloaded without re-detecting.

        Everything is written as a plain, fixed-shape array of a definite dtype, so the
        file reloads with allow_pickle=False on any numpy. The old version let `popts`
        become an object array, which is what forced the manual np.savez fixups."""
        popts = np.asarray([np.asarray(p, dtype=float).ravel() for p in self.popts])
        np.savez(
            path,
            version=np.array(2),
            n=np.array(int(self.n)),
            mean_img=np.asarray(self.mean_img, dtype=float),
            locations=np.asarray(self.locations, dtype=float),
            active=np.asarray(self.active, dtype=bool),
            threshold=np.array(float(self.threshold)),
            popts=popts.astype(float),
            positions_um=np.asarray(self.positions_um, dtype=float),
            freqs_hz=np.asarray(self.freqs_hz, dtype=float),
        )

    def plot_sites(self):
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.imshow(self.mean_img, cmap="viridis")
        ax.imshow(np.where(self.active, 1.0, np.nan), cmap="autumn", alpha=0.35)
        for i, (x, y) in enumerate(self.locations):
            ax.plot(x, y, "r.")
            ax.text(
                x + 2,
                y - 2,
                "%d\n%.3f" % (i, self.freqs_hz[i] / 1e6),
                color="r",
                fontsize=7,
                va="bottom",
            )
        ax.set_title("calibration: %d sites (index / MHz)" % self.n)
        fig.tight_layout()
        plt.show()
        return fig

    def plot_mask(self, crop=None, ax=None):
        from matplotlib.patches import Circle

        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(9.0, 3.4))
        ax.set_facecolor("black")
        ax.imshow(np.where(self.active, 1.0, np.nan), cmap="autumn", vmin=0, vmax=1)
        for i, (x, y) in enumerate(self.locations):
            ax.add_patch(
                Circle((x, y), ROI_SIZE / 2, fill=False, edgecolor="orange", lw=1.8)
            )
            ax.text(
                x,
                y - ROI_SIZE / 2 - 1.5,
                str(i),
                color="orange",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        if crop is not None:
            ax.set_xlim(*crop[0])
            ax.set_ylim(crop[1][1], crop[1][0])
        ax.set_title("ROI mask: %d sites (threshold %.1f)" % (self.n, self.threshold))
        if own:
            fig.tight_layout()
            plt.show()
        return ax


def calibrate(
    glob_pattern,
    n_expected=N_SITES,
    first_site=FIRST_SITE,
    frame=LOAD_FRAME,
    binning=BINNING,
    row_band=None,
    atom_threshold=ATOM_THRESHOLD,
    spacing_um=SPACING_UM,
    center=CENTER_ARRAY,
    f_start_hz=F_START_HZ,
    um_per_MHz=UM_PER_MHZ,
    show=False,
):
    """Build a Calibration from a set of calibration images: locate the sites,
    fit the count mask, and set the occupancy threshold."""
    stack = load_frames(glob_pattern, frame=frame, binning=binning)
    mean_img = stack.mean(0)
    print("calibrating from %d images" % len(stack))
    locations = locate_sites(
        mean_img, n_expected=n_expected, row_band=row_band, first_site=first_site
    )
    active, popts = fit_site_masks(mean_img, locations)
    counts = np.array([site_counts(img, locations, active) for img in stack])
    if atom_threshold is None:
        threshold, fit = auto_threshold(counts.flatten())
        print(
            "  auto threshold = %.1f  (empty %.1f / filled %.1f)"
            % (threshold, min(fit[1], fit[3]), max(fit[1], fit[3]))
        )
    else:
        threshold, fit = float(atom_threshold), None
        print("  fixed threshold = %.1f" % threshold)
    cal = Calibration(
        mean_img,
        locations,
        active,
        threshold,
        popts,
        spacing_um=spacing_um,
        center=center,
        f_start_hz=f_start_hz,
        um_per_MHz=um_per_MHz,
    )
    print("  per-site loading fraction: %s" % np.round((counts > threshold).mean(0), 2))
    print("  site frequencies [MHz]: %s" % np.round(cal.freqs_hz / 1e6, 3))
    if show:
        cal.plot_sites()
    return cal


def load_calibration(
    path,
    spacing_um=SPACING_UM,
    center=CENTER_ARRAY,
    f_start_hz=F_START_HZ,
    um_per_MHz=UM_PER_MHZ,
):
    """Rebuild a Calibration from a cache written by Calibration.save().

    Loads without pickle. A cache written by the old code may hold object arrays; that is
    detected, coerced once, and the file is rewritten in the v2 plain-array format -- so
    the conversion happens automatically, once, instead of by hand in a scratch script."""
    try:
        d = np.load(path, allow_pickle=False)
        f = {k: d[k] for k in d.files}
        rewrite = "version" not in f
    except ValueError:
        d = np.load(path, allow_pickle=True)
        f = {k: d[k] for k in d.files}
        rewrite = True
        print(
            "[sorter] %s holds pickled objects (old format) - converting once." % path
        )

    locations = np.asarray(f["locations"], dtype=float)
    popts = [np.asarray(p, dtype=float).ravel() for p in f["popts"]]
    cal = Calibration(
        np.asarray(f["mean_img"], dtype=float),
        locations,
        np.asarray(f["active"], dtype=bool),
        float(f["threshold"]),
        popts,
        spacing_um=spacing_um,
        center=center,
        f_start_hz=f_start_hz,
        um_per_MHz=um_per_MHz,
    )
    if rewrite:
        cal.save(path)
        print("[sorter] rewrote %s in the plain-array format (v2)." % path)
    return cal


# ===========================================================================
# Config (single reproducible parameter source, saved beside the images)
# ===========================================================================

PARAM_KEYS = [
    "spacing_um",
    "center",
    "f_start_hz",
    "x_freq_hz",
    "um_per_MHz",
    "n_expected",
    "binning",
    "atom_threshold",
    "row_band",
    "serial",
    "f_min_hz",
    "f_max_hz",
    "max_channel_amp_v",
    "max_channel_amp_v_ch1",
    "max_total_ch0",
    "max_total_ch1",
    "core_mapping",
    "step_time_s",
    "wpps",
    "trajectory",
    "revert_delay_s",
    "start_window",
    "image_is_photons",
    "transpose_image",
    "calib_prefix",
]
# Note: amplitudes_ch0 / amplitude_ch1 are deliberately NOT persisted here. They are a
# runtime trap-power tuning knob (set per launch / in hold_array.py); keeping them out of
# the saved config means the constructor kwarg always wins over a reloaded folder config.


def save_config(folder, cfg):
    """Write the run parameters to <folder>/sorter_config.json (reproducibility)."""
    out = {k: cfg[k] for k in PARAM_KEYS if k in cfg}
    if cfg.get("calib_glob") is not None:
        out["calib_glob"] = cfg["calib_glob"]
    with open(os.path.join(folder, CONFIG_NAME), "w") as f:
        json.dump(out, f, indent=2)


def load_config(folder):
    """Read <folder>/sorter_config.json if present, else return {}."""
    path = os.path.join(folder, CONFIG_NAME)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if isinstance(cfg.get("row_band"), list):  # json has no tuples
        cfg["row_band"] = tuple(cfg["row_band"])
    return cfg


# ===========================================================================
# Sorter1D
# ===========================================================================


def _as_glob(calib_source):
    """Accept a folder path or a glob string; a folder becomes '<folder>/*.npy'.
    To pick specific files, pass a glob that matches them."""
    if isinstance(calib_source, (list, tuple)):
        raise ValueError(
            "pass a folder path or a glob string (e.g. r'dir\\*.npy'), not a list"
        )
    if os.path.isdir(calib_source):
        return os.path.join(calib_source, "*.npy")
    return calib_source


def _pick_folder(title="Select calibration folder"):
    """Windows Explorer folder picker (tkinter, stdlib)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        raise RuntimeError("no calibration folder selected")
    return folder


def _collect_until_done(folder, calib_prefix):
    """Block on a small window showing the live count of <prefix>*.npy in `folder`
    (the GUI keeps saving there) until the user clicks 'Done'. Returns the count."""
    import tkinter as tk

    pat = os.path.join(folder, calib_prefix + "*.npy")
    root = tk.Tk()
    root.title("Calibration data-taking")
    lbl = tk.Label(root, font=("Segoe UI", 12), padx=24, pady=14)
    lbl.pack()
    tk.Button(
        root, text="Done - enough data", command=root.destroy, padx=14, pady=6
    ).pack(pady=(0, 14))

    def tick():
        lbl.config(text="calibration shots: %d" % len(glob.glob(pat)))
        root.after(500, tick)

    tick()
    root.mainloop()
    return len(glob.glob(pat))


class Sorter1D:
    """Camera-driven 1D sorter. Three ways to build it:

      Sorter1D(None)          -> calibration run: pick a folder, arm the 1xN on the AWG,
                                 collect <calib_prefix>*.npy shots (GUI saves them) until
                                 'Done', calibrate ROIs, cache the calibration + config.
      Sorter1D(<folder>)      -> reload: read sorter_config.json and the cached
                                 sorter_calibration.npz (instant; the resume-after-break path),
                                 else calibrate from <folder>/<calib_prefix>*.npy.
      Sorter1D(<glob>)        -> calibrate directly from images matching the glob.

    Then feed images to sort() (blocking) or sort_live()/revert() (non-blocking, for the GUI
    hook). Ch0 = the N tweezers that move; Ch1 = one fixed centered tone held throughout.
    """

    def __init__(
        self,
        calib_source,
        *,
        revert_delay_s=2.0,
        x_freq_hz=X_FREQ_HZ,
        record_path=None,
        start_window=None,
        serial=SERIAL,
        f_start_hz=F_START_HZ,
        f_min_hz=F_MIN_HZ,
        f_max_hz=F_MAX_HZ,
        max_channel_amp_v=MAX_AMPV,
        max_channel_amp_v_ch1=MAX_AMPV_CH1,
        max_total_ch0=MAX_TOTAL_CH0,
        max_total_ch1=MAX_TOTAL_CH1,
        um_per_MHz=UM_PER_MHZ,
        core_mapping=CORE_MAPPING,
        spacing_um=SPACING_UM,
        center=CENTER_ARRAY,
        step_time_s=STEP_TIME_S,
        wpps=WAYPOINTS_PER_STEP,
        trajectory=TRAJECTORY,
        n_expected=N_SITES,
        row_band=None,
        atom_threshold=ATOM_THRESHOLD,
        binning=BINNING,
        image_is_photons=False,
        transpose_image=False,
        calib_prefix=CALIB_PREFIX,
        pre_move_guard_s=0.0,  # was 0.01: a 10 ms time.sleep() before every move_2d
        amp_ramp_s=AMP_RAMP_S,
        amp_ramp_top=AMP_RAMP_TOP,
        amp_ramp_out=AMP_RAMP_OUT,
        amp_ramp_waypoints=AMP_RAMP_WAYPOINTS,
        move_amp_mode=MOVE_AMP_MODE,
        move_amp_value=MOVE_AMP_VALUE,
        show_calibration=False,
        amplitudes_ch0=None,
        amplitude_ch1=None,
        snap_to_grid=True,
    ):
        # amplitudes_ch0: None -> equal share of max_total_ch0; else a length-n array of
        #   per-tone fractions (hand-tuned to equalize trap depths across the array).
        # amplitude_ch1: None -> max_total_ch1; else the single CH1 tone's fraction.
        # Stored JSON-safe (list/float) so save_config can serialize them.
        amplitudes_ch0 = (
            None if amplitudes_ch0 is None else [float(a) for a in amplitudes_ch0]
        )
        amplitude_ch1 = None if amplitude_ch1 is None else float(amplitude_ch1)
        params = dict(
            spacing_um=spacing_um,
            center=center,
            f_start_hz=f_start_hz,
            x_freq_hz=x_freq_hz,
            um_per_MHz=um_per_MHz,
            n_expected=n_expected,
            binning=binning,
            atom_threshold=atom_threshold,
            row_band=row_band,
            serial=serial,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            max_channel_amp_v=max_channel_amp_v,
            max_channel_amp_v_ch1=max_channel_amp_v_ch1,
            max_total_ch0=max_total_ch0,
            max_total_ch1=max_total_ch1,
            core_mapping=core_mapping,
            step_time_s=step_time_s,
            wpps=wpps,
            trajectory=trajectory,
            revert_delay_s=revert_delay_s,
            start_window=start_window,
            image_is_photons=image_is_photons,
            transpose_image=transpose_image,
            calib_prefix=calib_prefix,
            amplitudes_ch0=amplitudes_ch0,
            amplitude_ch1=amplitude_ch1,
        )
        self.record_path = record_path
        self.pre_move_guard_s = float(pre_move_guard_s)
        # amp_ramp_s: duration [s] of the adiabatic amplitude ramp at each end of a move.
        # amp_ramp_top: per-tone amplitude at the ramp ends; None -> the static-arm level.
        self.amp_ramp_s = float(amp_ramp_s or 0.0)
        self.amp_ramp_top = None if amp_ramp_top is None else float(amp_ramp_top)
        # amp_ramp_out: ramp back down at the end of the move (park at the loading depth)
        # or leave the sorted traps deep for the survival image. See AMP_RAMP_OUT above.
        self.amp_ramp_out = bool(amp_ramp_out)
        self.amp_ramp_waypoints = amp_ramp_waypoints
        # move_amp_mode / move_amp_value: see MOVE_AMP_MODE above.
        self.move_amp_mode = str(move_amp_mode)
        self.move_amp_value = None if move_amp_value is None else float(move_amp_value)
        # snap_to_grid: on connect(), round f_start / spacing / x-tone to the card's DDS
        # frequency grid so all array tones are exactly equally spaced (kills slow beating).
        self.snap_to_grid = bool(snap_to_grid)

        if not calib_source:  # ---- calibration run ----
            folder = _pick_folder()
            params["calib_glob"] = os.path.join(
                folder, params["calib_prefix"] + "*.npy"
            )
            save_config(folder, params)
            self._apply_params(params)
            self._build_ctrl(params)
            if self.ctrl is not None:  # arm the 1xN so atoms load into it
                self.ctrl.connect()
                # must snap BEFORE arming: an off-grid spacing makes each tone round to a
                # slightly different grid point, so the pairwise spacings differ and the
                # array beats slowly in power - during the very shots we calibrate from.
                if self.snap_to_grid:
                    self._snap_to_grid(n=self.n_expected)
                self._arm_array(self.n_expected)
            n = _collect_until_done(folder, params["calib_prefix"])
            print("collected %d calibration shots; calibrating..." % n)
            self.cal = calibrate(
                params["calib_glob"],
                n_expected=self.n_expected,
                binning=self.binning,
                row_band=self.row_band,
                atom_threshold=self.atom_threshold,
                spacing_um=self.spacing_um,
                center=self.center,
                f_start_hz=self.f_start_hz,
                um_per_MHz=self.um_per_MHz,
                show=show_calibration,
            )
            self.cal.save(os.path.join(folder, CALIB_CACHE))
            self._finish_geometry()
            print(
                "calibration ready. Next:  python run_sort_live.py  (CALIB_FOLDER=%r)"
                % folder
            )
            return

        folder = calib_source if os.path.isdir(calib_source) else None
        if folder is not None:
            params.update(load_config(folder))  # config in the folder wins
        self._apply_params(params)
        self._build_ctrl(params)

        cache = os.path.join(folder, CALIB_CACHE) if folder else None
        if cache and os.path.isfile(cache):  # instant reload (resume path)
            self.cal = load_calibration(
                cache,
                spacing_um=self.spacing_um,
                center=self.center,
                f_start_hz=self.f_start_hz,
                um_per_MHz=self.um_per_MHz,
            )
            print("loaded cached calibration: %s" % cache)
        else:  # calibrate from images
            glob_pat = params.get("calib_glob")
            if glob_pat is None:
                glob_pat = (
                    os.path.join(folder, self.calib_prefix + "*.npy")
                    if folder
                    else _as_glob(calib_source)
                )
            self.cal = calibrate(
                glob_pat,
                n_expected=self.n_expected,
                binning=self.binning,
                row_band=self.row_band,
                atom_threshold=self.atom_threshold,
                spacing_um=self.spacing_um,
                center=self.center,
                f_start_hz=self.f_start_hz,
                um_per_MHz=self.um_per_MHz,
                show=show_calibration,
            )
        self._finish_geometry()

    # --- construction helpers ----------------------------------------------
    def _apply_params(self, p):
        for k in (
            "spacing_um",
            "center",
            "f_start_hz",
            "x_freq_hz",
            "um_per_MHz",
            "n_expected",
            "binning",
            "atom_threshold",
            "row_band",
            "step_time_s",
            "wpps",
            "trajectory",
            # amp_ramp_s / amp_ramp_top / move_amp_mode / move_amp_value are deliberately
            # NOT read from the config, for the same reason as amplitudes_ch0: they are
            # runtime trap-depth knobs, not properties of the calibration. They come from
            # the constructor kwarg, defaulting to the module constants. Leaving them here
            # meant a sorter_config.json written by an older version silently overrode both.
            "revert_delay_s",
            "start_window",
            "image_is_photons",
            "transpose_image",
            "calib_prefix",
            "max_total_ch0",
            "max_total_ch1",
        ):
            if k in p:  # tolerate a kwarg/config set that predates a newer key
                setattr(self, k, p[k])
        self.amplitudes_ch0 = p.get("amplitudes_ch0")
        self.amplitude_ch1 = p.get("amplitude_ch1")
        self._awg_params = {
            k: p[k]
            for k in (
                "serial",
                "f_min_hz",
                "f_max_hz",
                "max_channel_amp_v",
                "max_channel_amp_v_ch1",
                "core_mapping",
            )
        }
        self.x_pos_um = (
            (self.x_freq_hz - self.f_start_hz) / 1e6 * self.um_per_MHz
        )  # Ch1 tone

    def _build_ctrl(self, p):
        if AWGController is None:
            self.ctrl = None
        else:
            self.ctrl = AWGController(
                serial_number=p["serial"],
                f_start_hz=p["f_start_hz"],
                f_min_hz=p["f_min_hz"],
                f_max_hz=p["f_max_hz"],
                max_channel_amp_v=p["max_channel_amp_v"],
                max_channel_amp_v_ch1=p["max_channel_amp_v_ch1"],
                max_total_amplitude=max(p["max_total_ch0"], p["max_total_ch1"]),
                um_per_MHz=p["um_per_MHz"],
                core_mapping=p["core_mapping"],
            )

    def _finish_geometry(self):
        # precomputed invariants (the useful "cache") - reused every revert
        self.n = self.cal.n
        self.positions_um = site_positions(self.n, self.spacing_um, self.center)
        self.all_on = np.ones(self.n, dtype=bool)
        self.phases_static = static_phases(self.n)
        kind = "optimised" if self.n in OPTIMIZED_STATIC_PHASES else "Schroeder"
        f0 = convert_position_to_freq(
            self.positions_um, f_start_hz=self.f_start_hz, um_per_MHz=self.um_per_MHz
        )[0]
        w = self._ch0_amps(n=self.n)
        a_s = scale_static_amps(w, f0, self.phases_static)
        k_half = max(self.n // 2, 1)
        ramp_txt = (
            "amp ramp %.0f us to a_k %.4f (%s)"
            % (
                self.amp_ramp_s * 1e6,
                a_s[0] if self.amp_ramp_top is None else self.amp_ramp_top,
                "in+out, park at that level"
                if self.amp_ramp_out
                else "in only, park at the MOVE level",
            )
            if self.amp_ramp_s
            else "no amp ramp"
        )
        print(
            f"[Sorter1D] n={self.n}  static phases: {kind}  "
            f"peak {peak_of_sum(a_s, f0, self.phases_static):.3f} of full scale\n"
            f"           static a_k {a_s[0]:.4f} sum(a^2) {np.sum(a_s**2):.4f}"
            f"  |  move a_k [{self.move_amp_mode}]: "
            f"{self._move_a_k(self.n):.4f} at K={self.n} .. "
            f"{self._move_a_k(k_half):.4f} at K={k_half}"
            f"  |  {ramp_txt}"
        )
        self._check_cached_geometry()

    def _check_cached_geometry(self):
        """Shout if the loaded calibration cannot be trusted.

        load_calibration() replays whatever order was frozen into the npz at write time --
        locate_sites() never runs on that path. So an edit to the detection code is NOT
        picked up by run_sort_live.py; only re-running Calibration.py is. These two guards
        make a stale or mis-ordered cache announce itself instead of silently addressing
        the wrong frequencies."""
        s = (self.cal.locations - self.cal.locations.mean(0)) @ array_axis(
            self.cal.locations
        )
        if np.any(np.diff(s) <= 0):
            print(
                "[Sorter1D] WARNING: cached site order is not monotonic along the array "
                "axis -> the npz predates the ordering fix. Re-run Calibration.py."
            )
        if self.n_expected is not None and self.n != self.n_expected:
            print(
                "[Sorter1D] WARNING: calibration has %d sites but n_expected=%d."
                % (self.n, self.n_expected)
            )

    def _snap_to_grid(self, n=None):
        """Round f_start, the site spacing and the CH1 tone onto the card's DDS frequency
        grid so every array tone lands exactly equally spaced (no off-grid beating). The
        physical shift is < one grid step; positions/geometry are recomputed to match.

        Site frequencies become f_start + k*n_steps*f_step, all exact grid points, which is
        also what puts every sort-move START and END frequency on the grid: moves run
        between site positions. `n` is passed during the calibration run, where self.n does
        not exist yet."""
        f_step = self.ctrl.dds_freq_step_hz()
        self._f_step_hz = f_step  # cached: _snap_positions runs per move
        self.f_start_hz = snap_freq_to_grid(self.f_start_hz, f_step)
        self.spacing_um = snap_spacing_to_grid(self.spacing_um, f_step, self.um_per_MHz)
        self.x_freq_hz = snap_freq_to_grid(self.x_freq_hz, f_step)
        self.ctrl.f_start_hz = self.f_start_hz  # controller uses this at program time
        self.x_pos_um = (self.x_freq_hz - self.f_start_hz) / 1e6 * self.um_per_MHz
        n = getattr(self, "n", None) if n is None else n
        if n is not None:
            self.positions_um = site_positions(n, self.spacing_um, self.center)
        print(
            "[Sorter1D] snapped to DDS grid (step %.3f Hz): f_start=%.6f MHz, "
            "spacing=%.6f um (%d steps), x-tone=%.6f MHz"
            % (
                f_step,
                self.f_start_hz / 1e6,
                self.spacing_um,
                round(self.spacing_um / self.um_per_MHz * 1e6 / f_step),
                self.x_freq_hz / 1e6,
            )
        )

    def _snap_positions(self, P):
        """Round every trajectory waypoint onto the DDS frequency grid, so the commanded
        chirp runs between frequencies the card can actually produce and its own rounding
        adds nothing further. Endpoints are site positions, already on-grid via
        _snap_to_grid; this covers the intermediate sweep samples."""
        f_step = getattr(self, "_f_step_hz", None) or self.ctrl.dds_freq_step_hz()
        f = convert_position_to_freq(
            P, f_start_hz=self.f_start_hz, um_per_MHz=self.um_per_MHz
        )
        f = np.round(f / f_step) * f_step
        return (f - self.f_start_hz) / 1e6 * self.um_per_MHz

    # --- lifecycle ---------------------------------------------------------
    def connect(self):
        if self.ctrl is None:
            raise RuntimeError("AWG SDK (Controller) not available")
        self.ctrl.connect()
        if self.snap_to_grid:
            self._snap_to_grid()
        self.arm_initial()
        return self

    def close(self):
        if self.ctrl is not None:
            self.ctrl.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _guard_amps(amps, budget, label):
        """Hard per-channel guard: scale the tone amplitudes down (keeping their ratios)
        so the summed fraction never exceeds `budget`. Each output channel drives its own
        AOD, so this is applied independently per channel."""
        amps = np.asarray(amps, dtype=float)
        s = float(np.max(np.sum(amps, axis=0)))
        if budget > 0 and s > budget:
            print(
                "[Sorter1D] %s summed amplitude %.3f > budget %.3f -> scaling tones to fit"
                % (label, s, budget)
            )
            amps = amps * (budget / s)
        return amps

    def _ch0_amps(self, cols=None, n=None):
        """Per-tone CH0 amplitude fractions. amplitudes_ch0 is None -> equal share
        max_total_ch0/n_full; else the hand-tuned length-n_full array, indexed by `cols` for
        a subset of occupied sites. `n` is the full array size (defaults to self.n; passed
        explicitly during the calibration-run arm, before self.n exists).

        Read as ABSOLUTE fractions by the move (max_total_ch0/n_full is the rigorous
        no-clip bound there), and as RATIOS by the static arm, which rescales them via
        scale_static_amps(). Either way the ratios between tones - the relative trap
        depths you hand-tune - are what carry over to both."""
        n_full = int(n) if n is not None else self.n
        if self.amplitudes_ch0 is None:
            share = self.max_total_ch0 / n_full
            return np.full(n_full if cols is None else len(cols), share)
        amps = np.asarray(self.amplitudes_ch0, dtype=float)
        if amps.shape != (n_full,):
            raise ValueError(
                "amplitudes_ch0 must have length n=%d (got %d)" % (n_full, amps.size)
            )
        return amps if cols is None else amps[np.asarray(cols, dtype=int)]

    def _ch1_amp(self):
        """The single CH1 tone's amplitude fraction (amplitude_ch1 or max_total_ch1)."""
        return self.max_total_ch1 if self.amplitude_ch1 is None else self.amplitude_ch1

    def _arm_array(self, n):
        """Program a static 1xn array on Ch0 plus the single static tone on Ch1.

        CH0 is peak-scaled by scale_static_amps(): _ch0_amps() supplies only the RATIOS
        between tones, the absolute level comes from STATIC_HEADROOM. Deliberately NOT run
        through _guard_amps(max_total_ch0) -- the scaled sum is ~2.6 for n=10, and that
        guard would drag it straight back to 1.0 and undo the gain. The peak, which is what
        actually clips, is held at STATIC_HEADROOM by construction."""
        positions = site_positions(n, self.spacing_um, self.center)
        phases0 = static_phases(n)
        freqs0 = convert_position_to_freq(
            positions, f_start_hz=self.f_start_hz, um_per_MHz=self.um_per_MHz
        )[0]
        amps0 = scale_static_amps(self._ch0_amps(n=n), freqs0, phases0)
        amps1 = self._guard_amps(
            np.array([self._ch1_amp()]), self.max_total_ch1, "CH1 arm"
        )
        self.ctrl.program_static_2d(
            positions,
            np.ones(n, dtype=bool),
            [self.x_pos_um],
            [True],
            amplitudes_ch0=amps0,
            amplitudes_ch1=amps1,
            phases_ch0=phases0,
            phases_ch1=[0.0],
        )
        self._static_amps0 = amps0

    def arm_initial(self):
        """Program the original 1xN array on Ch0 plus the single static tone on Ch1."""
        self._arm_array(self.n)

    def set_transport(self, step_time_s=None, trajectory=None):
        """Change transport speed / profile between runs (STA-vs-linear / speed scans)."""
        if step_time_s is not None:
            self.step_time_s = float(step_time_s)
        if trajectory is not None:
            self.trajectory = str(trajectory).lower()

    # --- occupancy ---------------------------------------------------------
    def _occupancy(self, image):
        img = np.asarray(image)
        if self.transpose_image:
            img = img.T
        if not self.image_is_photons:
            img = _counts_to_photons(img, self.binning)
        return self.cal.occupancy(img)

    # --- visualization (mirrors image_analysis.plot_shot) -----------------
    def plot(self, image, start_window=None, crop=None):
        """Side-by-side: the shot with per-site occupancy (idx:0/1), and the
        target block with curved arrows from each source to its target site."""
        from matplotlib.patches import Circle, FancyArrowPatch

        img = np.asarray(image)
        if self.transpose_image:
            img = img.T
        if not self.image_is_photons:
            img = _counts_to_photons(img, self.binning)
        if start_window is None:
            start_window = self.start_window

        mask = self.cal.occupancy(img)
        sites, moves = plan_moves(mask, start_window=start_window)
        locs = self.cal.locations

        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 3.8))
        a0.imshow(
            img,
            cmap="viridis",
            vmin=np.percentile(img, 50),
            vmax=np.percentile(img, 99.7),
        )
        for i, (x, y) in enumerate(locs):
            col = "#4cdf70" if mask[i] else "#ff6b6b"
            a0.add_patch(
                Circle((x, y), ROI_SIZE / 3, fill=False, edgecolor=col, lw=1.8)
            )
            a0.text(
                x,
                y - ROI_SIZE / 2 - 1.5,
                "%d:%d" % (i, mask[i]),
                color=col,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        a0.set_title("image + classification  (idx:occ,  %d atoms)" % int(mask.sum()))

        a1.set_facecolor("black")
        a1.imshow(np.where(self.cal.active, 1.0, np.nan), cmap="bone", vmin=0, vmax=1.6)
        target = set(int(s) for s in sites)
        for i, (x, y) in enumerate(locs):
            if i in target:
                a1.add_patch(
                    Circle(
                        (x, y),
                        ROI_SIZE / 3 + 1.5,
                        fill=False,
                        edgecolor="#ffce85",
                        lw=2.2,
                    )
                )
            a1.plot(x, y, "o", color="#4cdf70" if mask[i] else "#666666", ms=6)
            a1.text(
                x,
                y + ROI_SIZE / 2 + 2,
                str(i),
                color="w",
                ha="center",
                va="top",
                fontsize=8,
            )
        for sc, dc in moves:
            a1.add_patch(
                FancyArrowPatch(
                    locs[sc],
                    locs[dc],
                    connectionstyle="arc3,rad=-0.5",
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#ff5555",
                    lw=1.8,
                )
            )
        a1.set_title("target block (orange) + %d moves" % len(moves))

        for a in (a0, a1):
            if crop is not None:
                a.set_xlim(*crop[0])
                a.set_ylim(crop[1][1], crop[1][0])
        fig.tight_layout()
        plt.show()
        return sites, moves

    # --- planning (no hardware) -------------------------------------------
    def plan_only(self, image):
        """Return (mask, sites, moves, t, P) without touching hardware."""
        mask = self._occupancy(image)
        sites, moves = plan_moves(mask, start_window=self.start_window)
        if int(mask.sum()) == 0:
            return mask, sites, moves, None, None
        t, P, _ = build_sort_trajectory(
            mask,
            self.positions_um,
            moves,
            self.step_time_s,
            self.wpps,
            self.amp_ramp_waypoints,
            self.trajectory,
        )
        return mask, sites, moves, t, P

    def dry_run(self, image, start_window=None):
        mask = self._occupancy(image)
        if start_window is None:
            start_window = self.start_window
        sites, moves = plan_moves(mask, start_window=start_window)
        target = set(int(s) for s in sites)
        print(
            "loaded : "
            + "".join("#" if x else "." for x in mask)
            + "   (%d atoms)" % int(mask.sum())
        )
        print(
            "target : " + "".join("#" if i in target else "." for i in range(len(mask)))
        )
        if len(sites):
            print(
                "         sites=%d..%d  block=%d  moves=%d"
                % (sites[0], sites[-1], len(sites), len(moves))
            )
        for k, (sc, dc) in enumerate(moves, 1):
            print(
                "  move %2d:  col %d -> col %d   (%.3f -> %.3f MHz)"
                % (k, sc, dc, self.cal.freqs_hz[sc] / 1e6, self.cal.freqs_hz[dc] / 1e6)
            )
        if not moves:
            print("  nothing to move.")
        return mask, sites, moves

    def _static_amps_full(self):
        """The per-tone amplitudes the static arm actually programs, for all n tones."""
        f0 = convert_position_to_freq(
            self.positions_um, f_start_hz=self.f_start_hz, um_per_MHz=self.um_per_MHz
        )[0]
        return scale_static_amps(self._ch0_amps(n=self.n), f0, static_phases(self.n))

    def _static_level(self):
        """Per-tone amplitude the static arm runs at - the default top of the ramp."""
        return float(np.max(self._static_amps_full()))

    def _move_a_k(self, K):
        """The per-tone move amplitude that will actually be programmed for K equal tones,
        i.e. after the sum_k a_k <= max_total_ch0 clamp. Used for the startup report so it
        cannot advertise a level the clamp will not allow."""
        K = max(int(K), 1)
        return min(self._move_level(np.ones(K)), self.max_total_ch0 / K)

    def _move_level(self, w):
        """Absolute per-tone amplitude during a move, for weights w normalised to max 1.

        w carries the ratios between the K tones that are on; this returns the scale applied
        to them. See MOVE_AMP_MODE for what each mode means."""
        w = np.asarray(w, dtype=float)
        mode = str(self.move_amp_mode).lower()
        if mode == "explicit":
            if self.move_amp_value is None:
                raise ValueError("move_amp_mode='explicit' requires move_amp_value")
            return float(self.move_amp_value)
        if mode == "sum_amp":
            return self.max_total_ch0 / float(np.sum(w))
        a_s = self._static_amps_full()
        if mode == "match_static":
            return float(np.max(a_s))
        if mode == "equal_power":
            return float(np.sqrt(np.sum(a_s**2) / np.sum(w**2)))
        raise ValueError(
            "unknown move_amp_mode %r (match_static | sum_amp | equal_power | explicit)"
            % self.move_amp_mode
        )

    def _move_setpoints(self, t, P, occ_cols, ramp_s=0.0):
        """freqs / phases / amplitudes for the Ch0 sort move (mirrors run_in_the_lab).
        occ_cols: the initially occupied columns in tone order (row k of P tracks the atom
        that started in column occ_cols[k]) -> each moving tone keeps its own site's amplitude."""
        K, T = P.shape
        freqs_kt = convert_position_to_freq(
            P, f_start_hz=self.f_start_hz, um_per_MHz=self.um_per_MHz
        )
        dt = np.diff(t)
        # pre-compensate the chirp phase accumulated over the move so tones land on schroeder phases
        accumulated = (
            np.pi * ((freqs_kt[:, :-1] + freqs_kt[:, 1:]) * dt).sum(axis=1)
        ) % (2 * np.pi)
        phases_move = (schroeder_generalized(K) - accumulated) % (2 * np.pi)
        # per-tone CH0 amplitude (hand-tuned array or equal share), held across the move,
        # scaled down to the budget if ever exceeded (hard guard, ratios kept)

        w = self._ch0_amps(cols=occ_cols)
        w = w / float(np.max(w))  # ratios, normalised to 1
        a_move = self._move_level(w)  # scale chosen by move_amp_mode
        amps_kt = np.tile((w * a_move)[:, np.newaxis], (1, T))
        # Always clamp: sum_k a_k is the worst-case peak during a chirp, so exceeding
        # max_total_ch0 clips and generates intermodulation products.
        amps_kt = self._guard_amps(amps_kt, self.max_total_ch0, "CH0 move")
        if ramp_s and ramp_s > 0.0 and T >= 3:
            # build_sort_trajectory() added one waypoint at each end with the positions
            # held, so raising just those two columns turns segment 0 into a ramp DOWN from
            # the top level to the move level, and the final segment into a ramp back UP.
            # _program_segment_core() reads amps_kt[:, i] as the anchor and
            # amps_kt[:, i+1] - amps_kt[:, i] as the slope, so the card does both in
            # hardware. Applied AFTER _guard_amps so the two ramp columns do not drag the
            # whole move down; the top level is set by amp_ramp_top.
            a_flat = float(np.max(amps_kt[:, T // 2]))
            top = (
                self._static_level() if self.amp_ramp_top is None else self.amp_ramp_top
            )
            if a_flat > 0.0:
                nr = max(1, int(self.amp_ramp_waypoints))
                # Anchors of a min-jerk envelope from top down to the move level. nr + 1
                # points, the last of which IS the flat level (already in the array), so only
                # the first nr are written. Scaling by env/a_flat keeps the per-tone weights.
                env = min_jerk(float(top), a_flat, nr + 1)[:nr] / a_flat
                amps_kt[:, :nr] *= env[None, :]
                if self.amp_ramp_out:
                    # mirror image: min-jerk is symmetric, so the reversed lead envelope is
                    # exactly the closing ramp, ending on top in the last column.
                    amps_kt[:, -nr:] *= env[::-1][None, :]
        return amps_kt, phases_move

    def _write_record(self, mask, sites, moves):
        rec = {
            "time": time.time(),
            "trajectory": self.trajectory,
            "step_time_s": self.step_time_s,
            "mask": [int(x) for x in mask],
            "sites": [int(s) for s in sites],
            "moves": [[int(a), int(b)] for a, b in moves],
        }
        with open(self.record_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _start_move(self, mask, moves):
        """Build + fire the self-triggered sort move (non-blocking; card holds at the end
        until stop()). Returns the move duration t[-1] [s]."""
        t, P, occ_cols = build_sort_trajectory(
            mask,
            self.positions_um,
            moves,
            self.step_time_s,
            self.wpps,
            self.amp_ramp_waypoints,
            self.trajectory,
            ramp_s=self.amp_ramp_s,
        )
        if self.snap_to_grid:
            # snap before _move_setpoints: the phase pre-compensation integrates these
            # frequencies, so it must see the ones the card will actually emit
            P = self._snap_positions(P)
        amps_kt, phases_move = self._move_setpoints(
            t, P, occ_cols, ramp_s=self.amp_ramp_s
        )
        pos_ch1 = np.full((1, P.shape[1]), self.x_pos_um)
        amps_ch1 = self._guard_amps(
            np.array([self._ch1_amp()]), self.max_total_ch1, "CH1 move"
        )
        if self.pre_move_guard_s:  # guard against motion before stream flush
            time.sleep(self.pre_move_guard_s)
        self.ctrl.move_2d(
            t,
            P,
            pos_ch1,
            amplitudes_ch0=amps_kt,
            amplitudes_ch1=amps_ch1,
            phases_ch0=phases_move,
            phases_ch1=[0.0],
            force_trigger=True,
        )  # fires now, holds when done
        return float(t[-1])

    # --- the core per-image entry points ----------------------------------
    def sort(self, image):
        """Sort one incoming image (BLOCKING): fire the moves on an internal trigger, hold the
        sorted block for revert_delay_s, then re-arm the original 1xN array.
        Returns a record dict {mask, sites, moves}."""
        if self.ctrl is None:
            raise RuntimeError(
                "AWG SDK (Controller) not available - use plan_only() offline"
            )

        mask = self._occupancy(image)
        sites, moves = plan_moves(mask, start_window=self.start_window)
        record = {"mask": mask, "sites": sites, "moves": moves}
        print(
            "loaded : "
            + "".join("#" if x else "." for x in mask)
            + "   (%d atoms)" % int(mask.sum())
        )

        if int(mask.sum()) == 0 or not moves:
            return record  # already the original 1xN array

        t0 = time.perf_counter()
        dur = self._start_move(mask, moves)

        if self.record_path:  # idle-time work while the card runs/holds
            self._write_record(mask, sites, moves)

        wait = dur + self.revert_delay_s - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)

        self.revert()  # back to the original 1xN array
        return record

    def sort_live(self, image):
        """Sort one image (NON-BLOCKING): compute occupancy, fire the self-triggered move,
        and return immediately so the survival image (image 3) can be taken during the hold.
        Call revert() afterwards (e.g. wired to the GUI's image-3 signal)."""
        # S = time.perf_counter()
        if self.ctrl is None:
            raise RuntimeError(
                "AWG SDK (Controller) not available - use plan_only() offline"
            )
        mask = (
            self._occupancy(image)
            if FORCE_MASK is None
            else np.asarray(FORCE_MASK, dtype=int)
        )
        sites, moves = plan_moves(mask, start_window=self.start_window)
        record = {"mask": mask, "sites": sites, "moves": moves}
        print(
            "loaded : "
            + "".join("#" if x else "." for x in mask)
            + "   (%d atoms)" % int(mask.sum())
        )
        # print(record)
        if int(mask.sum()) == 0 or not moves:
            return record  # already the original 1xN array
        self._start_move(mask, moves)
        if self.record_path:
            self._write_record(mask, sites, moves)

        # elapsed_time = time.perf_counter() - S
        # print(f"Code took {elapsed_time:.6f} seconds to run.")
        return record

    def revert(self):
        """Stop the held sort move and re-arm the original 1xN array (call after image 3).

        park=False: the old stop() zeroed every core and slept 10 ms before the re-arm,
        i.e. >10 ms of darkness. _program_static_cores() stops the card itself, so the
        sorted block simply stays lit until the re-armed array replaces it."""
        if self.ctrl is None:
            return
        self.ctrl.stop(park=False)
        self.arm_initial()
