"""
image_analysis.py
Lab pipeline for the 1D tweezer sorter. Learn the trap layout from a set of
calibration images (locate the spots, fit a per-site Gaussian count mask as in
tweezerAnalysis), threshold each shot into a binary occupancy mask, and feed it
to the sliding-window sorter (find_best_window / plan_moves) to plan the moves
that pack the loaded atoms into a filled block.

Two entry points:
  analyze_offline()  - load old runs, compute the mask, find the sort trajectory
                       (no hardware / AWG SDK needed).
  run_in_the_lab()   - camera-driven loop: image, sort, then hold the sorted
                       tones for the readout image (cf. testv1 / testv2).

The sorter / trajectory / plotting helpers below are shared with Sorting1DArrays.py.

@author: Bjarne Schümann
29.06.2026
"""

import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import curve_fit
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent: Controller / Fading_Shepard

try:
    from Controller import AWGController, convert_position_to_freq
    from Fading_Shepard import schroeder_generalized
except Exception:                                 # AWG SDK absent -> offline analysis still works
    AWGController = None
    def convert_position_to_freq(position_arr, f_start_hz=91.0e6, um_per_MHz=4.6 / 0.6):
        position_arr = np.asarray(position_arr, dtype=float)
        if position_arr.ndim == 1:
            position_arr = position_arr[np.newaxis, :]
        return f_start_hz + (position_arr / um_per_MHz) * 1e6
    def schroeder_generalized(M):                 # IMD phase prescription; falls back to zeros offline
        if M < 2:
            return np.zeros(max(M, 1))
        n = np.arange(M)
        return np.pi * n * (n - 1) / M

try:                                              # pure-numpy signal reconstruction (no AWG SDK needed)
    from Fading_Shepard import render_segments, crest_factor
except Exception:
    render_segments = None
    crest_factor = None



N            = 30        # number of trap sites in the 1D array (<= 20 on one channel)
SPACING_UM   = 4.6       # site separation [um]
CENTER_ARRAY = True      # center the sites about position 0 (= f_start)

STEP_TIME_S        = 20e-6   # speed: 3um/20us = 150 um/ms
WAYPOINTS_PER_STEP = 15      # min-jerk samples per single-site step
HOLD_S             = 2.5     # [s] hold at the sorted block for imaging [s]
LOAD_LEAD_S        = 0.5     # [s] plot-only lead: loading point drawn at -LOAD_LEAD_S

SERIAL        = 24909
F_START_HZ    = 91.0e6
MAX_AMPV      = 0.65
MAX_TOTAL     = 0.8
CHANNEL       = 0
FORCE_TRIGGER = True

# --- image analysis ---------------------------------------------------------
CALIB_GLOB       = r"c:\dev\GitHub\AWGController\Data\tweezerImages\tweezerLoad1x8-SpeedScan_*.npy"  # real 1x8 line
CALIB_DEBUG_GLOB = r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts\ExperimentalDataAnalysis\ExperimentalData\20260610\tweezerLoad2x2-NoMotion_99_*.npy"
DEBUG_ROW_BAND   = (50, 85)   # debug only: keep one row of the 2x2/2x3 test data -> a 1D line
N_SITES          = 8          # expected sites for the cross-check; None -> infer from detection
LOAD_FRAME       = 1          # 0 background, 1 loading (used for occupancy), 2 survival
BINNING          = 2
BORDER_PX        = 8          # K outermost pixels ignored when locating (edge dark counts)
ROI_SIZE         = 10         # ROI box [px] for the Gaussian fit and the counting
DETECT_SMOOTH    = 1.0        # gaussian pre-smoothing for detection [px]
DETECT_FRAC      = 0.4        # detection threshold as a fraction of the brightest pixel
MIN_AREA         = 3          # min connected pixels for a valid spot
GAUSS_FRAC       = 0.3        # active mask pixels: gaussian > GAUSS_FRAC * A + offset
SPACING_TOL      = 0.30       # max fractional deviation of inter-site spacing before warning
LEFT_IS_LOW_FREQ = True       # leftmost spot in image == lowest frequency / trap index 0
ATOM_THRESHOLD   = None       # None -> auto double-Gaussian threshold; float -> override
ACQUIRE_SOURCE   = None        # default for acquire_image(): file path, folder path, glob, or array


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
    occupied = np.flatnonzero(mask) # current trap index of each atom, in order
    K = occupied.size
    if K == 0:
        return 0, np.empty(0, dtype=int), 0

    lo, hi = 0, N - K # valid window starts keep the block in range
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

def min_jerk(p0, p1, n):
    """compute min jerk trajectory from p0->p1 in n steps"""
    s = np.linspace(0.0, 1.0, n)
    return p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)

def build_sort_trajectory(mask, positions_um, moves,
                          step_time_s=STEP_TIME_S, wpps=WAYPOINTS_PER_STEP):
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

    tone_at_col = {occ_cols[i]: i for i in range(K)}      # column -> tone

    t = [0.0]
    P = [[positions_um[occ_cols[i]]] for i in range(K)]

    for sc, dc in moves:
        tone = tone_at_col.pop(sc)
        n_steps = abs(dc - sc)
        if n_steps == 0:
            continue
        n = max(2, n_steps * wpps)
        dur = n_steps * step_time_s
        seg_t = np.linspace(t[-1], t[-1] + dur, n)[1:]            # drop dup start
        seg_x = min_jerk(positions_um[sc], positions_um[dc], n)[1:]
        for j in range(K):
            if j == tone:
                P[j].extend(seg_x.tolist())
            else:
                P[j].extend([P[j][-1]] * len(seg_t))
        t.extend(seg_t.tolist())
        tone_at_col[dc] = tone

    return np.asarray(t), np.asarray(P), occ_cols

def plot_trajectories(t, P, positions_um, sites, n_moves):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for c in range(len(positions_um)):
        ax.axhline(positions_um[c], color="#e6e6e6", lw=0.7, zorder=0)
    for c in sites:
        ax.axhline(positions_um[c], color="#ffce85", lw=7, alpha=0.55, zorder=0)
    for k in range(P.shape[0]):
        ax.plot(t * 1e3, P[k], lw=1.7, zorder=3)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("position [$\\mu$m]")
    ax.set_title("1D sort: %d atoms, %d moves, step=%.2f ms/site, hold=%.1f s"
                 % (P.shape[0], n_moves, STEP_TIME_S * 1e3, HOLD_S))
    fig.tight_layout()
    plt.show()
    return fig

def build_freq_amp(mask, t, P, positions_um, ctrl, lead_s=LOAD_LEAD_S):
    """
    THIS IS JUST FOR PLOTTING SAKES!
    Per-core frequency/amplitude over the WHOLE channel (all N sites), with two
    loading waypoints prepended at -lead_s so the initialization phase is shown.

    Power is split equally over all N sites and held constant: every core sits at
    max_total/N during loading and the occupied cores stay there throughout the move
    (no power boost when the empties switch off — that wouldn't be repeatable in the
    experiment). Empty cores carry max_total/N only while loading and drop to 0 at the
    move start. A core's frequency is blanked (NaN) wherever its amplitude is 0, so an
    off tone leaves no line.

    Returns (t_ext, F, A): t_ext (T+2,) [s]; F, A each (N, T+2) [Hz], [frac].
    """
    mask = np.asarray(mask).astype(bool)
    positions_um = np.asarray(positions_um, dtype=float)
    P = np.asarray(P, dtype=float)
    n_sites = len(positions_um)
    occ_cols = [c for c in range(n_sites) if mask[c]]
    row_of_col = {c: k for k, c in enumerate(occ_cols)}    # site column -> tone row in P
    T = P.shape[1]
    tone_amp = ctrl.max_total_amplitude / n_sites          # equal, constant share

    P_all = np.empty((n_sites, T + 2))
    A_all = np.empty((n_sites, T + 2))
    for c in range(n_sites):
        if c in row_of_col:                                # occupied -> moves, always on
            traj = P[row_of_col[c]]
            P_all[c] = np.concatenate([traj[:1], traj[:1], traj])
            A_all[c] = tone_amp
        else:                                              # empty -> parked, on only at load
            P_all[c] = positions_um[c]
            A_all[c, :2] = tone_amp
            A_all[c, 2:] = 0.0

    F_all = convert_position_to_freq(P_all, f_start_hz=ctrl.f_start_hz,
                                     um_per_MHz=ctrl.um_per_MHz)
    F_all = np.where(A_all > 0, F_all, np.nan)             # frequency off when amplitude is 0

    t_ext = np.insert(np.asarray(t, dtype=float), 0, [t[0] - lead_s, t[0] - 6.4e-9])
    return t_ext, F_all, A_all

def plot_freq_amp(mask, t, P, positions_um, sites, ctrl, lead_s=LOAD_LEAD_S):
    """Frequency + amplitude of every core in the channel vs time, including the
    loading/initialization phase drawn at -lead_s. Occupied cores are solid, empty
    (load-only) cores dashed; their lines vanish when they switch off at the move."""
    positions_um = np.asarray(positions_um, dtype=float)
    n_sites = len(positions_um)
    t_ext, F, A = build_freq_amp(mask, t, P, positions_um, ctrl, lead_s=lead_s)
    occ = np.asarray(mask).astype(bool)
    tone_amp = ctrl.max_total_amplitude / n_sites

    fig, (ax_f, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6.4))

    site_f = convert_position_to_freq(positions_um, f_start_hz=ctrl.f_start_hz,
                                      um_per_MHz=ctrl.um_per_MHz)[0]
    for c in range(n_sites):
        ax_f.axhline(site_f[c] / 1e6, color="#e6e6e6", lw=0.7, zorder=0)
    for c in sites:
        ax_f.axhline(site_f[c] / 1e6, color="#ffce85", lw=7, alpha=0.55, zorder=0)

    for c in range(n_sites):
        kw = dict(lw=1.7) if occ[c] else dict(lw=1.3, ls="--", alpha=0.7)
        line, = ax_f.plot(t_ext * 1e3, F[c] / 1e6, zorder=3, **kw)
        ax_a.plot(t_ext * 1e3, A[c], color=line.get_color(), zorder=3, **kw)
        ax_f.plot(t_ext[:2] * 1e3, F[c, :2] / 1e6, "o", ms=5,
                  color=line.get_color(), zorder=4)
        ax_a.plot(t_ext[:2] * 1e3, A[c, :2], "o", ms=5,
                  color=line.get_color(), zorder=4)

    for ax in (ax_f, ax_a):
        ax.axvline(0.0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    ax_f.set_ylabel("frequency [MHz]")
    ax_a.set_ylabel("amplitude [frac]")
    ax_a.set_xlabel("time [ms]   (loading at %.0f ms)" % (-lead_s * 1e3))
    ax_f.set_title("1D sort: %d/%d cores on, constant %.3f/tone (max_total=%.2f)"
                   % (int(occ.sum()), n_sites, tone_amp, ctrl.max_total_amplitude))
    fig.tight_layout()
    plt.show()
    return fig

def replay_states(mask, moves):
    """Occupied-column lists before and after each move: states[0] is the initial
    load, states[k] the occupancy after the k-th move."""
    occ = [c for c in range(len(mask)) if mask[c]]
    states = [list(occ)]
    cur = list(occ)
    for sc, dc in moves:
        cur = sorted(dc if c == sc else c for c in cur)
        states.append(list(cur))
    return states

def plot_slider(mask, sites, moves):
    """Interactive 1D analog of HCA's plot_slider: a row of sites with a slider to
    step through the moves. Atoms inside the target block are green, outside red;
    the current move is drawn as an arrow from source to destination. Needs a GUI
    backend."""
    from matplotlib.patches import Circle
    from matplotlib.widgets import Slider

    mask = np.asarray(mask).astype(bool)
    n_sites = len(mask)
    states = replay_states(mask, moves)
    target = set(int(s) for s in sites)
    n = len(moves)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * n_sites), 2.8))
    plt.subplots_adjust(bottom=0.30)

    def render(step):
        ax.clear()
        if len(sites):
            ax.axvspan(sites[0] - 0.5, sites[-1] + 0.5, color="#ffce85",
                       alpha=0.45, zorder=0)
        for c in range(n_sites):
            ax.add_patch(Circle((c, 0), 0.16, facecolor="none",
                                edgecolor="#bbbbbb", lw=1.1, zorder=1))
        for c in states[step]:
            color = "#4c9f70" if c in target else "#d96459"
            ax.add_patch(Circle((c, 0), 0.33, facecolor=color, edgecolor="none", zorder=2))
        if step > 0:
            sc, dc = moves[step - 1]
            ax.annotate("", xy=(dc, 0), xytext=(sc, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.2,
                                        shrinkA=8, shrinkB=8), zorder=3)
            ax.add_patch(Circle((sc, 0), 0.30, facecolor="none",
                                edgecolor="#666666", lw=1.8, zorder=4))
        title = "initial" if step == 0 else \
            "after move %d:  col %d -> col %d" % (step, moves[step - 1][0], moves[step - 1][1])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1, n_sites)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_yticks([])
        ax.set_xticks(range(0, n_sites, max(1, n_sites // 15)))
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        fig.canvas.draw_idle()

    sax = plt.axes((0.16, 0.10, 0.68, 0.05))
    slider = Slider(sax, "step", 0, max(n, 1), valinit=0, valstep=1)
    slider.on_changed(lambda v: render(int(v)))
    render(0)
    plt.show()
    return fig, slider

def report_bandwidth(ctrl, t, P, n_sites=N):
    """Offline timing/FIFO sanity check via plan()."""
    res = ctrl.plan(t, P, channel=CHANNEL, hold=HOLD_S,
                    amplitudes=ctrl.max_total_amplitude / n_sites, skip_timing_check=True)
    K, T = res.freqs_kt.shape
    n_cmds = ctrl._estimate_n_commands(K, T)
    cap = ctrl._queue_max_actual or ctrl.SINGLE_MODE_QUEUE_MAX
    dt = np.diff(res.time_arr)
    print("  K=%d tones, T=%d waypoints, min dt=%.2f us"
          % (K, T, dt.min() * 1e6))
    print("  ~%d DDS commands vs FIFO %d  ->  %s"
          % (n_cmds, cap, "STREAMING" if n_cmds > cap else "single-shot"))
    print("  freq band used: [%.3f, %.3f] MHz"
          % (res.freqs_kt.min() / 1e6, res.freqs_kt.max() / 1e6))

def emulate(seed=0, fill_prob=0.6, mask=None, start_window=None):
    """
    Either pass a mask yourself or generate a random one given the 
    hardcoded variables above.
    """
    if mask is None:
        rng = np.random.default_rng(seed)
        mask = (rng.random(N) < fill_prob).astype(int)
        if mask.sum() == 0:
            mask[int(rng.integers(N))] = 1
    else:
        mask = np.asarray(mask).astype(int) # accept a list/array of 0/1
    n = len(mask)  # array size follows the given mask
    positions = site_positions(n, SPACING_UM, CENTER_ARRAY)

    print("loaded : " + "".join("#" if x else "." for x in mask)
          + "   (%d atoms)" % int(mask.sum()))
    sites, moves = plan_moves(mask, start_window=start_window)
    print("target : sites=%d..%d  block size=%d  moves=%d"
          % (sites[0], sites[-1], len(sites), len(moves)))
    for k, (sc, dc) in enumerate(moves, 1):
        print("  move %2d:  col %d -> col %d" % (k, sc, dc))

    plot_slider(mask, sites, moves)

    if not moves:
        print("already sorted - nothing to move.")
        return

    t, P, _ = build_sort_trajectory(mask, positions, moves)

    assert AWGController is not None, "AWG SDK (Controller) not available"
    ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                         max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                         realtime_priority=False)
    print("playback: %.3f ms sort + %.1f s hold" % (t[-1] * 1e3, HOLD_S))
    report_bandwidth(ctrl, t, P, n_sites=n)

    # append the hold for visualization
    t_full = np.append(t, t[-1] + HOLD_S)
    P_full = np.concatenate([P, P[:, -1:]], axis=1)
    plot_trajectories(t_full, P_full, positions, sites, len(moves))
    plot_freq_amp(mask, t_full, P_full, positions, sites, ctrl)

def monte_carlo(trials=10000, fill_prob=0.5, seed0=71):
    counts, atoms = [], []
    for sd in range(seed0, seed0 + trials):
        rng = np.random.default_rng(sd)
        mask = (rng.random(N) < fill_prob).astype(int)
        if mask.sum() < 1:
            continue
        counts.append(len(plan_moves(mask)[1]))
        atoms.append(int(mask.sum()))
    counts, atoms = np.array(counts), np.array(atoms)
    print("MC over %d loads (p=%.2f, N=%d):  <atoms>=%.2f  <moves>=%.2f  max moves=%d  max atoms=%d"
          % (len(counts), fill_prob, N, atoms.mean(), counts.mean(), counts.max(), atoms.max()))
    # print moves per atoms and max atoms
    print("%.4f moves per atom on average"
          % (counts.mean()/atoms.mean()))

def _counts_to_photons(data, binning=BINNING):
    return (np.asarray(data, dtype=np.int32) - 200 * binning ** 2) * 0.1

def _twoD_gaussian(coords, amp, x0, y0, sx, sy, theta, offset):
    x, y = coords
    a = np.cos(theta) ** 2 / (2 * sx ** 2) + np.sin(theta) ** 2 / (2 * sy ** 2)
    b = -np.sin(2 * theta) / (4 * sx ** 2) + np.sin(2 * theta) / (4 * sy ** 2)
    c = np.sin(theta) ** 2 / (2 * sx ** 2) + np.cos(theta) ** 2 / (2 * sy ** 2)
    g = offset + amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
    return g.ravel()

def _double_gaussian(k, A, m1, s1, m2, s2):
    return (1 - A) * norm.pdf(k, m1, s1) + A * norm.pdf(k, m2, s2)

def load_frames(glob_pattern, frame=LOAD_FRAME, binning=BINNING):
    """(M, H, W) photon-converted stack of one frame index over all matching .npy files."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError("no images match: %s" % glob_pattern)
    imgs = [np.asarray(np.load(f, allow_pickle=True)[()]["Images"])[frame] for f in files]
    return _counts_to_photons(np.asarray(imgs), binning)

def load_shot(glob_pattern, index=0, frame=LOAD_FRAME, binning=BINNING):
    """Single photon-converted frame from one file (one experimental shot)."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError("no images match: %s" % glob_pattern)
    d = np.load(files[index], allow_pickle=True)[()]
    return _counts_to_photons(np.asarray(d["Images"])[frame], binning)

def _check_sites(pts, n_expected, spacing_tol):
    n = len(pts)
    print("  found %d sites%s" % (n, "" if n_expected is None else " (expected %d)" % n_expected))
    if n_expected is not None and n != n_expected:
        print("  WARNING: site count %d != expected %d" % (n, n_expected))
    if n >= 3:
        d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        med = np.median(d)
        bad = np.where(np.abs(d - med) > spacing_tol * med)[0]
        print("  inter-site spacing [px]: %s  (median %.1f)" % (np.round(d, 1), med))
        if len(bad):
            print("  WARNING: uneven spacing at gaps %s - possible outlier / missed site" % bad.tolist())

def locate_sites(mean_img, n_expected=N_SITES, border_px=BORDER_PX, row_band=None,
                 smooth=DETECT_SMOOTH, frac=DETECT_FRAC, min_area=MIN_AREA,
                 spacing_tol=SPACING_TOL, left_is_low_freq=LEFT_IS_LOW_FREQ):
    """Spot centroids (x, y) in trap-index order. Ignores the border_px outermost
    pixels (edge dark counts); cross-checks the count and the inter-site spacing."""
    H, W = mean_img.shape
    work = gaussian_filter(mean_img - np.median(mean_img), smooth)
    valid = np.zeros((H, W), bool)
    r0, r1 = (0, H) if row_band is None else row_band
    valid[max(r0, border_px):min(r1, H - border_px), border_px:W - border_px] = True
    binimg = (work > work[valid].max() * frac) & valid
    lbl, n = label(binimg) # type: ignore
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
    pts = np.array(sorted(pts))                      # ascending image-x
    if not left_is_low_freq:
        pts = pts[::-1]
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
        sub = mean_img[r0:r0 + roi_size, c0:c0 + roi_size]
        rr, cc = np.meshgrid(np.arange(r0, r0 + roi_size), np.arange(c0, c0 + roi_size), indexing="ij")
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
        out[i] = masked[r0:r0 + roi_size + 1, c0:c0 + roi_size + 1].sum()
    return out

def auto_threshold(counts_flat):
    """Threshold at the valley between the two Gaussians fitted to the count histogram."""
    flat = np.asarray(counts_flat, dtype=float)
    lo, hi = flat.min(), flat.max()
    bins = np.linspace(lo, hi, max(12, int(np.sqrt(flat.size))))
    ent, edg = np.histogram(flat, bins=bins, density=True)
    ctr = 0.5 * (edg[1:] + edg[:-1])
    p0 = [0.5, lo + 0.2 * (hi - lo), 0.1 * (hi - lo), lo + 0.75 * (hi - lo), 0.1 * (hi - lo)]
    popt, _ = curve_fit(_double_gaussian, ctr, ent, p0=p0, maxfev=10000)
    m1, m2 = sorted((popt[1], popt[3]))
    ks = np.linspace(m1, m2, 1000)
    return float(ks[np.argmin(_double_gaussian(ks, *popt))]), popt

class Calibration:
    """Trap layout learned from the calibration images: site locations in
    trap-index order, the Gaussian count mask, the occupancy threshold, and the
    site positions/frequencies. occupancy(img) turns one shot into a binary mask."""

    def __init__(self, mean_img, locations, active, threshold, popts,
                 spacing_um=SPACING_UM, center=CENTER_ARRAY,
                 f_start_hz=F_START_HZ, um_per_MHz=4.6 / 0.6):
        self.mean_img = mean_img
        self.locations = np.asarray(locations, dtype=float)
        self.active = active
        self.threshold = float(threshold)
        self.popts = popts
        self.n = len(self.locations)
        self.positions_um = site_positions(self.n, spacing_um, center)
        self.freqs_hz = convert_position_to_freq(self.positions_um, f_start_hz=f_start_hz,
                                                 um_per_MHz=um_per_MHz)[0]

    def counts(self, img):
        return site_counts(img, self.locations, self.active)

    def occupancy(self, img):
        return (self.counts(img) > self.threshold).astype(int)

    def plot_sites(self):
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.imshow(self.mean_img, cmap="viridis")
        ax.imshow(np.where(self.active, 1.0, np.nan), cmap="autumn", alpha=0.35)
        for i, (x, y) in enumerate(self.locations):
            ax.plot(x, y, "r.")
            ax.text(x + 2, y - 2, "%d\n%.3f" % (i, self.freqs_hz[i] / 1e6),
                    color="r", fontsize=7, va="bottom")
        ax.set_title("calibration: %d sites (index / MHz)" % self.n)
        fig.tight_layout()
        plt.show()
        return fig

def _plot_count_histogram(counts_flat, threshold, fit=None):
    flat = np.asarray(counts_flat, dtype=float)
    bins = np.linspace(flat.min(), flat.max(), max(12, int(np.sqrt(flat.size))))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(flat, bins=bins, density=True, alpha=0.6)
    if fit is not None:
        ks = np.linspace(flat.min(), flat.max(), 500)
        ax.plot(ks, _double_gaussian(ks, *fit), "k", lw=1.5)
    ax.axvline(threshold, color="r", ls="--", label="threshold %.1f" % threshold)
    ax.set_xlabel("masked ROI counts")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig

def plot_mask(cal, crop=None, ax=None):
    """ROI/mask view: active pixels in orange on a black field, ROI circles + index labels."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9.0, 3.4))
    from matplotlib.patches import Circle
    ax.set_facecolor("black")
    ax.imshow(np.where(cal.active, 1.0, np.nan), cmap="autumn", vmin=0, vmax=1)
    for i, (x, y) in enumerate(cal.locations):
        ax.add_patch(Circle((x, y), ROI_SIZE / 2, fill=False, edgecolor="orange", lw=1.8))
        ax.text(x, y - ROI_SIZE / 2 - 1.5, str(i), color="orange",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    if crop is not None:
        ax.set_xlim(*crop[0])
        ax.set_ylim(crop[1][1], crop[1][0])
    ax.set_title("ROI mask: %d sites (threshold %.1f)" % (cal.n, cal.threshold))
    if own:
        fig.tight_layout() # type: ignore
        plt.show()
    return ax

def plot_shot(cal, img, start_window=None, crop=None, axes=None):
    """Side-by-side: shot image with per-site occupancy (idx:0/1), and the target
    block with curved arrows from each occupied source to its target site."""
    from matplotlib.patches import Circle, FancyArrowPatch
    mask = cal.occupancy(img)
    sites, moves = plan_moves(mask, start_window=start_window)
    locs = cal.locations
    own = axes is None
    if own:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 3.8))
    else:
        a0, a1 = axes

    a0.imshow(img, cmap="viridis",
              vmin=np.percentile(img, 50), vmax=np.percentile(img, 99.7))
    for i, (x, y) in enumerate(locs):
        col = "#4cdf70" if mask[i] else "#ff6b6b"
        a0.add_patch(Circle((x, y), ROI_SIZE / 3, fill=False, edgecolor=col, lw=1.8))
        a0.text(x, y - ROI_SIZE / 2 - 1.5, "%d:%d" % (i, mask[i]), color=col,
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    a0.set_title("image + classification  (idx:occ,  %d atoms)" % int(mask.sum()))

    a1.set_facecolor("black")
    a1.imshow(np.where(cal.active, 1.0, np.nan), cmap="bone", vmin=0, vmax=1.6)
    target = set(int(s) for s in sites)
    for i, (x, y) in enumerate(locs):
        if i in target:
            a1.add_patch(Circle((x, y), ROI_SIZE / 3 + 1.5, fill=False, edgecolor="#ffce85", lw=2.2))
        a1.plot(x, y, "o", color="#4cdf70" if mask[i] else "#666666", ms=6)
        a1.text(x, y + ROI_SIZE / 2 + 2, str(i), color="w", ha="center", va="top", fontsize=8)
    for sc, dc in moves:
        a1.add_patch(FancyArrowPatch(locs[sc], locs[dc], connectionstyle="arc3,rad=-0.5",
                                     arrowstyle="-|>", mutation_scale=12, color="#ff5555", lw=1.8))
    a1.set_title("target block (orange) + %d moves" % len(moves))

    for a in (a0, a1):
        if crop is not None:
            a.set_xlim(*crop[0])
            a.set_ylim(crop[1][1], crop[1][0])
    if own:
        fig.tight_layout() # type: ignore
        plt.show()
    return sites, moves

def calibrate(glob_pattern=None, n_expected=N_SITES, frame=LOAD_FRAME, binning=BINNING,
              row_band=None, atom_threshold=ATOM_THRESHOLD, show=True):
    """Build a Calibration from a set of calibration images: locate the sites,
    fit the count mask, and set the occupancy threshold."""
    glob_pattern = glob_pattern or CALIB_GLOB
    stack = load_frames(glob_pattern, frame=frame, binning=binning)
    mean_img = stack.mean(0)
    print("calibrating from %d images" % len(stack))
    locations = locate_sites(mean_img, n_expected=n_expected, row_band=row_band)
    active, popts = fit_site_masks(mean_img, locations)
    counts = np.array([site_counts(img, locations, active) for img in stack])
    if atom_threshold is None:
        threshold, fit = auto_threshold(counts.flatten())
        print("  auto threshold = %.1f  (empty %.1f / filled %.1f)"
              % (threshold, min(fit[1], fit[3]), max(fit[1], fit[3])))
    else:
        threshold, fit = float(atom_threshold), None
        print("  fixed threshold = %.1f" % threshold)
    cal = Calibration(mean_img, locations, active, threshold, popts)
    print("  per-site loading fraction: %s" % np.round((counts > threshold).mean(0), 2))
    print("  site frequencies [MHz]: %s" % np.round(cal.freqs_hz / 1e6, 3))
    if show:
        cal.plot_sites()
        _plot_count_histogram(counts.flatten(), threshold, fit)
    return cal

def analyze_offline(calib_glob=None, shot_glob=None, n_expected=N_SITES,
                    row_band=None, start_window=None, shot_index="auto", show=True):
    """Offline test: calibrate from old runs, pick a shot, compute its occupancy
    mask and find the sort trajectory - no hardware / AWG SDK required."""
    calib_glob = calib_glob or CALIB_DEBUG_GLOB
    cal = calibrate(calib_glob, n_expected=n_expected, row_band=row_band, show=show)

    shots = load_frames(shot_glob or calib_glob)
    occ_all = np.array([cal.occupancy(img) for img in shots])
    n_moves = np.array([len(plan_moves(o)[1]) for o in occ_all])
    if shot_index == "auto":                          # most illustrative shot
        idx = int(np.argmax(n_moves)) if n_moves.max() > 0 else int(np.argmax(occ_all.sum(1)))
    else:
        idx = int(shot_index)
    mask = occ_all[idx]
    print("shot %d : " % idx + "".join("#" if x else "." for x in mask)
          + "   (%d atoms)" % int(mask.sum()))

    sites, moves = plan_moves(mask, start_window=start_window)
    if len(sites):
        print("target  : sites=%d..%d  block=%d  moves=%d"
              % (sites[0], sites[-1], len(sites), len(moves)))
    for k, (sc, dc) in enumerate(moves, 1):
        print("  move %2d:  col %d -> col %d" % (k, sc, dc))

    if show and len(sites):
        plot_slider(mask, sites, moves)
    if not moves:
        print("nothing to move.")
        return cal, mask, sites, moves

    t, P, _ = build_sort_trajectory(mask, cal.positions_um, moves)
    if show:
        t_full = np.append(t, t[-1] + HOLD_S)
        P_full = np.concatenate([P, P[:, -1:]], axis=1)
        plot_trajectories(t_full, P_full, cal.positions_um, sites, len(moves))
        if AWGController is not None:
            ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                                 max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                                 realtime_priority=False)
            report_bandwidth(ctrl, t, P, n_sites=cal.n)
            plot_freq_amp(mask, t_full, P_full, cal.positions_um, sites, ctrl)
    return cal, mask, sites, moves

def acquire_image(source=None, frame=LOAD_FRAME, binning=BINNING, convert=True):
    """Return one occupancy frame (H, W), photon-converted to match the calibration.

    source may be:
      - ndarray (H, W)                 -> used directly
      - ndarray (2, H, W) / (3, H, W)  -> the loading frame (index `frame`) is taken
      - a file path (.npy)             -> loaded (dict with 'Images', or a raw array)
      - a folder path or glob pattern  -> the most recently modified .npy is loaded
      - None                           -> falls back to the ACQUIRE_SOURCE constant
    Set convert=False if the data is already photon-converted (float counts)."""
    import os
    if source is None:
        source = ACQUIRE_SOURCE
    if source is None:
        raise ValueError("acquire_image: no source - pass a file/folder path or an array, "
                         "or set ACQUIRE_SOURCE")

    if isinstance(source, str):
        if os.path.isdir(source):
            files = glob.glob(os.path.join(source, "*.npy"))
        elif os.path.isfile(source):
            files = [source]
        else:
            files = glob.glob(source)                       # treat as a glob pattern
        if not files:
            raise FileNotFoundError("acquire_image: no .npy found for %s" % source)
        loaded = np.load(max(files, key=os.path.getmtime), allow_pickle=True)   # latest taken
        arr = np.asarray(loaded[()]["Images"] if loaded.dtype == object else loaded)
    else:
        arr = np.asarray(source)

    if arr.ndim == 3:
        arr = arr[min(int(frame), arr.shape[0] - 1)]        # (F, H, W) -> loading frame
    elif arr.ndim != 2:
        raise ValueError("acquire_image: expected a 2D or 3D image, got shape %s" % (arr.shape,))

    return _counts_to_photons(arr, binning) if convert else arr.astype(float)

def run_in_the_lab(cal, start_window=None):
    """Camera-driven sort loop: image -> occupancy mask -> sort move that holds at the
    target for the readout image, then re-arm all tones for the next load."""
    if AWGController is None:
        raise RuntimeError("AWG SDK (Controller) not available")
    n = cal.n
    positions = cal.positions_um
    all_on = np.ones(n, dtype=bool)
    phases_static = schroeder_generalized(n)                           # IMD-suppressing tone phases

    def crest(label, freqs, amps, phases, tarr):                       # 1 us crest-factor probe
        if render_segments is None:
            return
        _, waves = render_segments(freqs, amps, phases, tarr, 1.25e9,
                                   (float(tarr[0]), float(tarr[0]) + 1e-6))
        print("  crest factor (%s): %.3f" % (label, crest_factor(waves.sum(axis=0))))

    two = np.array([0.0, 1e-6])

    ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                         max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                         realtime_priority=True)
    ctrl.connect()
    try:
        ctrl.program_static(positions, all_on, channel=CHANNEL, phases=phases_static)  # load all n tones
        crest("loading", np.tile(cal.freqs_hz[:, None], (1, 2)),
              np.full((n, 2), ctrl.max_total_amplitude / n), phases_static, two)
        while True:
            img = acquire_image()
            if img is None:
                break
            mask = cal.occupancy(img)
            print("loaded : " + "".join("#" if x else "." for x in mask)
                  + "   (%d atoms)" % int(mask.sum()))
            if mask.sum() == 0:
                print("  no atoms - re-arming.")
                continue

            sites, moves = plan_moves(mask, start_window=start_window)
            print("  target sites %d..%d, %d moves" % (sites[0], sites[-1], len(moves)))
            if moves:
                t, P, _ = build_sort_trajectory(mask, positions, moves)
                K = P.shape[0]
                freqs_kt = convert_position_to_freq(P, f_start_hz=ctrl.f_start_hz, um_per_MHz=ctrl.um_per_MHz)
                dt = np.diff(t)
                # pre-compensate the chirp phase accumulated over the move so tones land on schroeder phases
                accumulated = (np.pi * ((freqs_kt[:, :-1] + freqs_kt[:, 1:]) * dt).sum(axis=1)) % (2 * np.pi)
                phases_move = (schroeder_generalized(K) - accumulated) % (2 * np.pi)
                # renormalize per-tone amplitudes down to the total-power budget (never up)
                amps_kt = np.full((K, P.shape[1]), ctrl.max_total_amplitude / n)
                s = amps_kt.sum(axis=0).max()
                if s > ctrl.max_total_amplitude:
                    amps_kt = amps_kt * (ctrl.max_total_amplitude / s)
                crest("move start", freqs_kt, amps_kt, phases_move, t)
                ctrl.move(t, P, channel=CHANNEL, hold=HOLD_S, amplitudes=amps_kt,
                          phases=phases_move, force_trigger=FORCE_TRIGGER)
                if FORCE_TRIGGER:
                    time.sleep(t[-1] + HOLD_S + 0.1)
                else:
                    input("  armed, waiting for EXT0. press enter once imaged...\n")
                crest("hold", np.tile(freqs_kt[:, -1:], (1, 2)), np.tile(amps_kt[:, -1:], (1, 2)),
                      schroeder_generalized(K), two)
            else:
                print("  already sorted.")

            ctrl.stop()
            ctrl.program_static(positions, all_on, channel=CHANNEL, phases=phases_static)    # re-arm to all n
            crest("loading", np.tile(cal.freqs_hz[:, None], (1, 2)),
                  np.full((n, 2), ctrl.max_total_amplitude / n), phases_static, two)
    finally:
        ctrl.disconnect()

def demo(glob_pattern=None, n_expected=N_SITES, n_timing=10, n_examples=5, seed=123):
    """Full offline reproduction on a real dataset (no hardware): calibrate + show
    the ROI mask, time the lab-style loop over random shots, show example shots
    (classification | target + curved moves), then the sort trajectory + amplitudes.
    Figures pop up one at a time - close each window to advance."""
    glob_pattern = glob_pattern or CALIB_GLOB
    files = sorted(glob.glob(glob_pattern))
    print("dataset: %d files\n" % len(files))

    cal = calibrate(glob_pattern, n_expected=n_expected, show=False)
    x, y = cal.locations[:, 0], cal.locations[:, 1]
    crop = ((x.min() - ROI_SIZE, x.max() + ROI_SIZE), (y.min() - ROI_SIZE, y.max() + ROI_SIZE))
    plot_mask(cal, crop=crop)                                  # 1. the mask

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(files), size=min(n_timing, len(files)), replace=False)
    print("\nlab-style timing over %d random shots [ms]:" % len(idxs))
    print("  %5s | %7s %7s %8s | atoms" % ("shot", "load", "mask", "traj"))
    tot = []
    for i in idxs:
        t0 = time.perf_counter()
        img = load_shot(glob_pattern, index=int(i))            # read one frame (camera proxy)
        t1 = time.perf_counter()
        mask = cal.occupancy(img)                              # apply mask -> occupancy
        t2 = time.perf_counter()
        sites, moves = plan_moves(mask)                        # plan + trajectory
        if moves:
            build_sort_trajectory(mask, cal.positions_um, moves)
        t3 = time.perf_counter()
        tot.append((t3 - t0) * 1e3)
        print("  %5d | %7.2f %7.3f %8.3f | %d (%d moves)"
              % (int(i), (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, int(mask.sum()), len(moves)))
    tot = np.array(tot)
    print("  ---- mean total %.2f ms  (min %.2f / max %.2f) ----" % (tot.mean(), tot.min(), tot.max()))

    occ = [cal.occupancy(load_shot(glob_pattern, index=int(i))) for i in idxs]
    pick = [int(idxs[o]) for o in np.argsort([-len(plan_moves(o_)[1]) for o_ in occ])[:n_examples]]
    fig, axs = plt.subplots(len(pick), 2, figsize=(13, 3.1 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):                               # 3. example shots
        plot_shot(cal, load_shot(glob_pattern, index=i), crop=crop, axes=(axs[r, 0], axs[r, 1]))
        axs[r, 0].set_ylabel("shot %d" % i, fontsize=11)
    fig.tight_layout()
    plt.show()

    rep = pick[0]
    mask = cal.occupancy(load_shot(glob_pattern, index=rep))
    sites, moves = plan_moves(mask)
    if not moves:
        print("representative shot has no moves; done.")
        return cal
    t, P, _ = build_sort_trajectory(mask, cal.positions_um, moves)
    plot_trajectories(t, P, cal.positions_um, sites, len(moves))   # 4. trajectory (sort phase)
    if AWGController is not None:
        ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                             max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                             realtime_priority=False)
    else:
        from types import SimpleNamespace
        ctrl = SimpleNamespace(max_total_amplitude=MAX_TOTAL, f_start_hz=F_START_HZ, um_per_MHz=4.6 / 0.6)
    plot_freq_amp(mask, t, P, cal.positions_um, sites, ctrl, lead_s=t[-1] * 0.25)   # 4. amplitudes

    # 5. real RF signal for this trial: individual tones + summed output, with IMD-suppressing phases
    if render_segments is not None:
        K = P.shape[0]
        freqs_kt = convert_position_to_freq(P, f_start_hz=ctrl.f_start_hz, um_per_MHz=ctrl.um_per_MHz)
        amps_kt  = np.full((K, P.shape[1]), ctrl.max_total_amplitude / K)
        ph  = schroeder_generalized(K)
        win = (float(t[0]), float(t[0]) + 1e-6)
        tw, waves = render_segments(freqs_kt, amps_kt, ph, t, 1.25e9, win)
        summed = waves.sum(axis=0)
        cf  = crest_factor(summed)
        _, waves0 = render_segments(freqs_kt, amps_kt, np.zeros(K), t, 1.25e9, win)
        cf0 = crest_factor(waves0.sum(axis=0))
        print("shot %d: %d tones, crest factor Schroeder=%.3f vs zero-phase=%.3f (%.1f%% lower)"
              % (rep, K, cf, cf0, 100 * (1 - cf / cf0)))
        fig, (bx1, bx2) = plt.subplots(2, 1, sharex=True)
        for k in range(K):
            bx1.plot(tw * 1e6, waves[k], lw=0.7)
        bx2.plot(tw * 1e6, summed, color="k", lw=0.8)
        bx1.set_ylabel("individual tones"); bx2.set_ylabel("summed output"); bx2.set_xlabel("t [us]")
        bx1.set_title("real DDS signal, shot %d (%d tones) | crest factor %.2f (zero-phase %.2f)"
                      % (rep, K, cf, cf0))
        plt.show()
    return cal

if __name__ == "__main__":    
    demo(CALIB_GLOB)
    # live lab loop: cal = calibrate(CALIB_GLOB); run_in_the_lab(cal)
