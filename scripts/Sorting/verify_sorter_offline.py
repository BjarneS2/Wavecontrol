"""
verify_sorter_offline.py
Offline acceptance test for the 1D sorter's detection + planning + trajectory stack,
driven by the example data in Data/tweezerImages2906/ (1x8 loads, 3 frames each).

No AWG / spcm SDK and no camera are needed: sorter.py already guards a missing Controller
(AWGController -> None), so calibrate(), plan_moves() and build_sort_trajectory() all run
on the CPU. This script calibrates the 8-site layout from the plain load shots, then for a
broad sample of shots and every target policy (best / left / right / index) it asserts:
  - atom count conserved (occupied columns == trajectory rows K),
  - target block contiguous and length K,
  - every move src != dst and in range,
  - trajectory endpoints land exactly on the target sites,
  - all tone frequencies stay inside the AOD band,
  - every segment dt >= the DDS timer floor.
Finally it prints a per-policy summary. Optionally saves a few classification/plan figures.

Run:  python Sorting/verify_sorter_offline.py
      python Sorting/verify_sorter_offline.py --plot 5      # also save 5 plan figures

@author: Bjarne Schümann
"""

import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sorter import (  # noqa: E402
    calibrate, plan_moves, build_sort_trajectory, site_positions,
    convert_position_to_freq, _counts_to_photons, Sorter1D,
    SPACING_UM, CENTER_ARRAY, F_START_HZ, F_MIN_HZ, F_MAX_HZ, UM_PER_MHZ,
    N_SITES, BINNING, STEP_TIME_S, WAYPOINTS_PER_STEP, TRAJECTORY,
)

# DDS hardware timer floor (Controller.DDS_TIMER_MIN_NS); hardcoded so this script needs no
# spcm SDK. A segment shorter than this cannot be clocked by the card.
DDS_TIMER_MIN_S = 83.2e-9

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "Data", "tweezerImages2906")
CALIB_GLOB = os.path.join(DATA_DIR, "tweezerLoad1x8_*.npy")   # 40 plain loads -> calibration
TEST_GLOB  = os.path.join(DATA_DIR, "tweezerLoad1x8*.npy")    # all 1x8 loads -> test sample
LOAD_FRAME = 1                                                # 0 bg, 1 loading, 2 survival
N_TEST     = 300                                              # evenly-spaced test subset


def target_start(policy, mask):
    """Map a named policy to the start_window arg of plan_moves (per shot)."""
    N = len(mask)
    K = int(np.asarray(mask).sum())
    if policy == "best":
        return None            # move-minimizing window
    if policy == "left":
        return 0               # flush-left block
    if policy == "right":
        return N - K           # flush-right block
    if policy == "index":
        return min(2, max(0, N - K))   # arbitrary fixed start, clipped in-range
    raise ValueError(policy)


def check_shot(mask, positions_um, policy):
    """Run one shot through plan+trajectory for one policy; return per-shot metrics or
    raise AssertionError on the first violated invariant."""
    N = len(mask)
    occ = np.flatnonzero(mask)
    K = occ.size
    sw = target_start(policy, mask)
    sites, moves = plan_moves(mask, start_window=sw)

    # target block: contiguous, length K, in range
    assert len(sites) == K, "sites length %d != K %d" % (len(sites), K)
    if K:
        assert np.all(np.diff(sites) == 1), "target block not contiguous: %s" % sites
        assert sites[0] >= 0 and sites[-1] < N, "target block out of range: %s" % sites

    # moves: real (src != dst), in range, each source is an occupied site
    for src, dst in moves:
        assert 0 <= src < N and 0 <= dst < N, "move out of range %s" % ((src, dst),)
        assert src != dst, "no-op move %s" % ((src, dst),)
        assert mask[src], "move from empty site %d" % src

    if K == 0 or not moves:
        return dict(K=K, n_moves=len(moves), dur_s=0.0, min_dt_s=np.inf, T=0)

    t, P, occ_cols = build_sort_trajectory(
        mask, positions_um, moves, STEP_TIME_S, WAYPOINTS_PER_STEP, TRAJECTORY)

    # atom count conserved and tones start on the occupied sites
    assert P.shape[0] == K, "trajectory rows %d != K %d" % (P.shape[0], K)
    assert list(occ_cols) == list(occ), "occ_cols %s != occupied %s" % (occ_cols, occ.tolist())
    assert np.allclose(P[:, 0], positions_um[occ]), "tones do not start on their sites"

    # every atom lands exactly on a target site (as a set)
    assert np.allclose(np.sort(P[:, -1]), np.sort(positions_um[sites])), \
        "trajectory endpoints do not match target sites"

    # frequencies stay inside the AOD band
    freqs = convert_position_to_freq(P, f_start_hz=F_START_HZ, um_per_MHz=UM_PER_MHZ)
    assert freqs.min() >= F_MIN_HZ and freqs.max() <= F_MAX_HZ, \
        "freqs leave band [%.1f, %.1f] MHz" % (freqs.min() / 1e6, freqs.max() / 1e6)

    # segment timing floor
    dt = np.diff(t)
    min_dt = float(dt.min())
    assert min_dt >= DDS_TIMER_MIN_S - 1e-12, \
        "segment dt %.1f ns below DDS timer floor %.1f ns" % (min_dt * 1e9, DDS_TIMER_MIN_S * 1e9)

    return dict(K=K, n_moves=len(moves), dur_s=float(t[-1]), min_dt_s=min_dt, T=P.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", type=int, default=0,
                    help="save this many plan figures (needs a working Sorter1D.plot)")
    args = ap.parse_args()

    calib_files = sorted(glob.glob(CALIB_GLOB))
    test_files = sorted(glob.glob(TEST_GLOB))
    if not calib_files:
        raise SystemExit("no calibration files at %s" % CALIB_GLOB)
    if len(test_files) > N_TEST:                     # evenly spaced subset for speed
        test_files = test_files[:: max(1, len(test_files) // N_TEST)]

    print("calibrating from %d shots (%s)..." % (len(calib_files), os.path.basename(CALIB_GLOB)))
    cal = calibrate(CALIB_GLOB, n_expected=N_SITES, binning=BINNING,
                    spacing_um=SPACING_UM, center=CENTER_ARRAY,
                    f_start_hz=F_START_HZ, um_per_MHz=UM_PER_MHZ)
    N = cal.n
    positions_um = site_positions(N, SPACING_UM, CENTER_ARRAY)
    print("calibrated N=%d sites; testing %d shots x 4 policies\n" % (N, len(test_files)))

    policies = ["best", "left", "right", "index"]
    agg = {p: [] for p in policies}
    n_fail = 0
    for f in test_files:
        d = np.load(f, allow_pickle=True)[()]
        img = _counts_to_photons(np.asarray(d["Images"])[LOAD_FRAME], BINNING)
        mask = cal.occupancy(img)
        for p in policies:
            try:
                agg[p].append(check_shot(mask, positions_um, p))
            except AssertionError as e:
                n_fail += 1
                print("FAIL [%s] %s: %s" % (p, os.path.basename(f), e))

    print("\n%-7s %7s %10s %11s %13s %9s" %
          ("policy", "shots", "mean_K", "mean_moves", "mean_dur_us", "min_dt_ns"))
    for p in policies:
        rows = agg[p]
        if not rows:
            continue
        Ks = np.array([r["K"] for r in rows])
        mv = np.array([r["n_moves"] for r in rows])
        dur = np.array([r["dur_s"] for r in rows]) * 1e6
        mdt = np.array([r["min_dt_s"] for r in rows])
        mdt = mdt[np.isfinite(mdt)]
        print("%-7s %7d %10.2f %11.2f %13.1f %9.1f" %
              (p, len(rows), Ks.mean(), mv.mean(), dur.mean(),
               (mdt.min() * 1e9) if mdt.size else float("nan")))

    print("\n%s: %d assertion failure(s) across %d shots x %d policies" %
          ("FAILED" if n_fail else "PASSED", n_fail, len(test_files), len(policies)))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        srt = Sorter1D(CALIB_GLOB)               # ctrl=None; offline calibration
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_verify_figs")
        os.makedirs(outdir, exist_ok=True)
        for i, f in enumerate(test_files[: args.plot]):
            d = np.load(f, allow_pickle=True)[()]
            img = np.asarray(d["Images"])[LOAD_FRAME]
            srt.plot(img)                        # builds the figure
            out = os.path.join(outdir, "plan_%02d.png" % i)
            plt.savefig(out, dpi=110)
            plt.close("all")
            print("saved", out)

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
