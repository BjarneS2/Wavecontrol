"""
This script contains the Common Analysis functions shared among the plotting functions.
We still delegate to the sorter.py for all the things that defines the experiment, such
as counts-to-photons or site-counts. They have a version in CommonThings as well but
here just to keep it close to the experiment we just import from there. A few things to
flag before looking directly in the code:

Frame convention is:
    frame 0 == before collisional blockade (naturally the brightest since mutltiple atoms are in the traps)
    frame 1 == loading image -> occupancy either 1 or 0 atoms.
    frame 2 == survival      -> whatever action performed, this is the final image of the experiment

Data Layout:
    Each .npy is a pickled dictionary with "Images" of shape (3,H,W) uint16 camera counts in addition
    to global variables (up to 4 - limited by camera gui).
    Photons = (counts - 200*binning**2) * 0.1 and binning from sorter-config.json
    filename: <prefix>_<shot>_<YYYYMMDD-HHMMSS>.npy
    BE AWARE!: Shots are ordered by the integer shot index, NOT by filename -- sorted() puts _100 before _11

tooling for 1D sorter and experimental analysis
@author: Bjarne Schümann
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from CommonThings import (
    LOAD_FRAME,
    SURV_FRAME,
    Dataset,
    align_reports,
    build_plans,
    discover_runs,
    load_calibration,
    load_dataset,
    loading_drift,
    loss_distributions,
    per_site_thresholds,
    plan,
    plan_internal,
    read_reports,
    run_table,
    success_vs_k,
    survival_tables,
    wilson,
)

try:
    import Sorting.sorter as S
except Exception:  # noqa: BLE001
    S = None


def _fmt(k, n):
    """format Wilson for (k) out of (n)"""
    p, lo, hi = wilson(k, n)
    return (
        "  nan          "
        if n == 0
        else f"{p:.3f} +{hi - p:.3f}/-{p - lo:.3f} ({k}/{n})"
    )


def per_site_impact(cal_ds, run_dss, use_sorter=True):
    """
    I suspected that a global threshold from the first prototype might
    affect the classification and the subsequent results in running it
    live in the laboratoy, this function measures the impact of this
    difference in global versus site specific thresholding

    in a dict it saves things as disagreements such as:
    - Raw classification flips (global vs. per-site disagr.)
    - Total plan changes (disagr. that alter the move list)
    - Specific plan metric changes (K, t0, or target positions)
    """
    if cal_ds is None:
        raise ValueError("need the calibration dataset to fit per-site thresholds")
    thr_ps = per_site_thresholds(cal_ds.counts[:, LOAD_FRAME, :], cal_ds.cal.threshold)
    cal_ds.cal.per_site = thr_ps

    res = {
        "thresholds": thr_ps,
        "global": cal_ds.cal.threshold,
        "flips": 0,
        "site_shots": 0,
        "plan_changed": 0,
        "shots": 0,
        "k_changed": 0,
        "t0_changed": 0,
        "per_site_flips": np.zeros(cal_ds.cal.n, int),
    }
    for ds in [cal_ds] + list(run_dss):
        g = ds.occ(LOAD_FRAME, per_site=False)  # global
        p_ = ds.occ(LOAD_FRAME, per_site=True)  # per site
        diff = g != p_
        res["flips"] += int(diff.sum())
        res["site_shots"] += diff.size
        res["per_site_flips"] += diff.sum(0)
        if ds.is_calibration:
            continue
        for s in range(ds.n_shots):
            res["shots"] += 1
            if not diff[s].any():
                continue
            a = plan(g[s], use_sorter)
            b = plan(p_[s], use_sorter)
            if a.k != b.k or a.t0 != b.t0 or not np.array_equal(a.targets, b.targets):
                res["plan_changed"] += 1
                res["k_changed"] += a.k != b.k
                res["t0_changed"] += a.t0 != b.t0
    return res


def ghost_stats(run_dss, per_site=False):
    """
    The thresholding as discussed in the per-site-impact needs to be investigated,
    doing that one could look at ghost atom stats. Ghost atoms being atoms that
    were not seen in the first image but appear in the second. Essentially those
    that should not be there after moves - if no moves happened they might on the
    other hand still be there since the traps were not turned off.
    I of course would hope for this number to be 0 for all but it is what it is.


    Three disjoint categories for every frame-2 atom outside the predicted target block:

      at_origin   -- sits on the source site of an atom that was supposed to move. The
                      move failed to pick it up. This should technically be impossible to
                      maintain an atom there, since the traps should be turned off completely!
                      (for sorting - for single atom transport fine since no turn off)

      empty_in_f1  -- sits on a site that was classified EMPTY in frame 1. Nearly physically
                      impossible (the trap was switched off ONLY when there were actual moves
                      played), so this is a classification error: a missed atom in frame 1 or
                      a false positive in frame 2.

      other        -- occupied in frame 1, not a mover's origin, not in the block. Should
                      not happen if the plan is reconstructed correctly. Just catches all
                      the remaining cases that could in principle occur.
    """
    out = {}
    for ds in run_dss:
        m1, m2 = ds.occ(LOAD_FRAME, per_site), ds.occ(SURV_FRAME, per_site)
        r = {
            "shots": 0,
            "ghost": 0,
            "at_origin": 0,
            "empty_in_f1": 0,
            "other": 0,
            "shots_with_ghost": 0,
            "by_site": np.zeros(ds.cal.n, int),
            "by_k": {},
            "n_f2": 0,
        }
        for i, pl in enumerate(ds.plans):
            r["shots"] += 1
            blk = np.zeros(ds.cal.n, bool)
            if pl.k:
                blk[pl.t0 : pl.t0 + pl.k] = True
            gh = np.flatnonzero(m2[i] & ~blk)
            r["n_f2"] += int(m2[i].sum())
            bk = r["by_k"].setdefault(pl.k, [0, 0])
            bk[1] += 1
            if gh.size:
                r["shots_with_ghost"] += 1
                bk[0] += 1
            origins = {int(pl.sites[a]) for a in range(pl.k) if pl.dist[a] != 0}
            for g in gh:
                r["ghost"] += 1
                r["by_site"][g] += 1
                if int(g) in origins:
                    r["at_origin"] += 1
                elif not m1[i, g]:
                    r["empty_in_f1"] += 1
                else:
                    r["other"] += 1
        out[ds.name] = r
    return out


def print_ghosts(gh):
    """
    Print the ghost stats.
    """
    W = 100
    print("\n" + "=" * W)
    print("GHOSTS: frame-2 atoms outside the predicted block".ljust(W))
    print("-" * W)
    print(
        f"  {'run':40s} {'shots':>6} {'w/ghost':>8} {'ghosts':>7} {'/shot':>6} "
        f"{'failed move':>12} {'empty in f1':>12} {'other':>6}"
    )
    tot = {
        k: 0
        for k in (
            "shots",
            "ghost",
            "at_origin",
            "empty_in_f1",
            "other",
            "shots_with_ghost",
        )
    }
    for nm, r in gh.items():
        for k in tot:
            tot[k] += r[k]
        print(
            f"  {nm.replace('tweezerLoad1x11-', '')[:40]:40s} {r['shots']:6d} "
            f"{r['shots_with_ghost']:8d} {r['ghost']:7d} {r['ghost'] / max(1, r['shots']):6.3f} "
            f"{r['at_origin']:12d} {r['empty_in_f1']:12d} {r['other']:6d}"
        )
    print(
        f"  {'ALL':40s} {tot['shots']:6d} {tot['shots_with_ghost']:8d} {tot['ghost']:7d} "
        f"{tot['ghost'] / max(1, tot['shots']):6.3f} {tot['at_origin']:12d} "
        f"{tot['empty_in_f1']:12d} {tot['other']:6d}"
    )
    print(
        "  failed move = an atom left sitting on its source site. empty in f1 = a site\n"
        "  that was classified empty during loading yet holds an atom afterwards, which\n"
        "  is physically impossible and therefore counts classification errors."
    )


def planner_asymmetry(n=11, trials=200000, p=0.5, seed=0):
    """
    Just wanna see if there is a real more left to right moves. Should be the case
    since if occupancy is [#, #, *, *, #, #, *, *, *, *, *] the two occupancies on
    the left will be chosen for the starting window so the right ones will move to
    the left side - just a hardcoded choice since there physically should be no
    difference.

    Pure combinatorics on random masks -- no atoms, no images.
    """
    rng = np.random.default_rng(seed)
    left = right = 0
    dl = dr = 0
    for _ in range(trials):
        m = rng.random(n) < p
        pl = plan_internal(m)
        d = pl.dist
        left += int((d < 0).sum())
        right += int((d > 0).sum())
        dl += int(-d[d < 0].sum())
        dr += int(d[d > 0].sum())
    return {
        "left_moves": left,
        "right_moves": right,
        "left_frac": left / max(1, left + right),
        "mean_left_dist": dl / max(1, left),
        "mean_right_dist": dr / max(1, right),
    }


def make_summary_figures(
    outdir,
    surv,
    per_ds7,
    pooled7,
    loss,
    drift,
    ps=None,
    cal_ds=None,
    kbands=(),
    run_dss=(),
    n_sites=11,
):
    """I had claude build this Orchestrator that builds and saves every summary
    figure under its established filename, delegating each fig_* to the plotting
    script that owns it. `drift`is accepted for signature compatibility with the
    caller but unused, same as in the pre-split make_plots. Imports are local to
    avoid import cycles, since each of those scripts itself imports this module
    as `CommonAnalysis`."""
    import os

    import matplotlib.pyplot as plt
    from ExpectationsOfDelivery import fig_success_vs_k
    from FidelityOverDistance import fig_survival_vs_distance
    from LoadingDrift import fig_loading_drift
    from LossDistributions import fig_loss_distributions
    from PerSiteThresholds import (
        fig_per_site_calibration_survival,
        fig_per_site_thresholds,
    )
    from SurvivalStationaryVsMoved import fig_survival_stationary_vs_moved

    os.makedirs(outdir, exist_ok=True)
    saved = []

    fig = fig_survival_stationary_vs_moved(surv)
    p = os.path.join(outdir, "item4_survival_stationary_vs_moved.png")
    fig.savefig(p, dpi=130), plt.close(fig), saved.append(p)

    f = fig_survival_vs_distance(surv)
    if f is not None:
        p = os.path.join(outdir, "item5_survival_vs_distance.png")
        f.savefig(p, dpi=130), plt.close(f), saved.append(p)

    f = fig_success_vs_k(
        per_ds7, pooled7, kbands, "Item 7 -- defect-free block probability"
    )
    if f is not None:
        p = os.path.join(outdir, "item7_success_vs_K.png")
        f.savefig(p, dpi=130), plt.close(f), saved.append(p)

    f = fig_loss_distributions(loss)
    if f is not None:
        p = os.path.join(outdir, "item8_loss_distributions.png")
        f.savefig(p, dpi=130), plt.close(f), saved.append(p)

    f = fig_loading_drift(cal_ds, run_dss, n_sites=n_sites)
    if f is not None:
        p = os.path.join(outdir, "item10_loading_drift.png")
        f.savefig(p, dpi=130), plt.close(f), saved.append(p)

    if ps is not None:
        f = fig_per_site_thresholds(ps)
        p = os.path.join(outdir, "item9_per_site_thresholds.png")
        f.savefig(p, dpi=130), plt.close(f), saved.append(p)

    if cal_ds is not None:
        f = fig_per_site_calibration_survival(surv)
        if f is not None:
            p = os.path.join(outdir, "extra_per_site_survival.png")
            f.savefig(p, dpi=130), plt.close(f), saved.append(p)
    return saved


def print_report(surv, per_ds7, pooled7, loss, ps, aligns):
    """print report by claude - just some statistics printed out"""
    W = 78
    print("\n" + "=" * W)
    print("survival".ljust(W))
    print("-" * W)
    print("  calibration (imaging only) : " + _fmt(surv["cal"]["k"], surv["cal"]["n"]))
    print(
        "  runs, stationary atoms     : " + _fmt(surv["stat"]["k"], surv["stat"]["n"])
    )
    print(
        "  runs, moved atoms          : " + _fmt(surv["move"]["k"], surv["move"]["n"])
    )
    pc = wilson(surv["cal"]["k"], surv["cal"]["n"])[0]
    pm = wilson(surv["move"]["k"], surv["move"]["n"])[0]
    st = wilson(surv["stat"]["k"], surv["stat"]["n"])[0]
    if np.isfinite(pc) and np.isfinite(pm):
        print(
            f"\n  moving costs {100 * (st - pm):+.2f} pp relative to staying put, and "
            f"{100 * (pc - pm):+.2f} pp\n  relative to the calibration floor. "
            "Transport is only the culprit if the\n  moved/stationary gap is larger "
            "than the stationary/calibration gap."
        )
    f = surv["mover_fate"]
    print(
        f"\n  mover fate: {f['target']} arrived, {f['origin']} still at origin, "
        f"{f['gone']} gone"
    )

    print("\n" + "=" * W)
    print("survival vs distance".ljust(W))
    print("-" * W)
    for d in sorted(surv["by_dist"]):
        b = surv["by_dist"][d]
        print(
            f"  {d:2d} site(s): "
            + _fmt(b["k"], b["n"])
            + f"   left {_fmt(b['kL'], b['nL']).strip()}  right {_fmt(b['kR'], b['nR']).strip()}"
        )
    print(
        "\n  ^ MARGINAL, AND CONFOUNDED. Long moves happen in sparsely loaded shots, and"
    )
    print(
        "    survival depends strongly on K, so this column mixes two effects and can"
    )
    print("    even come out backwards. The table below is the one to quote.")
    ks = sorted({k for k, _ in surv["by_kd"]})
    ds_ = sorted({d for _, d in surv["by_kd"]})
    if ks:
        print("\n  survival of moved atoms at FIXED K (cells with n>=8):")
        print("     K  |" + "".join(f"  d={d:<9}" for d in ds_))
        for k in ks:
            row = f"    {k:2d}  |"
            for d in ds_:
                c = surv["by_kd"].get((k, d))
                row += (
                    f" {c[0] / c[1]:.2f}({c[1]:3d})"
                    if c and c[1] >= 8
                    else "           "
                )
            print(row)
        print("\n  per-atom survival vs K:")
        for k in sorted(surv["by_k"]):
            a_ = surv["by_k"][k]
            if a_["stat"][1] + a_["move"][1] < 8:
                continue
            print(
                f"    K={k:2d}  stationary {_fmt(*a_['stat']).strip():28s} "
                f"moved {_fmt(*a_['move']).strip()}"
            )

    print("\n" + "=" * W)
    print("P(defect-free block | K loaded)".ljust(W))
    print("-" * W)
    for k in sorted(pooled7):
        if pooled7[k][1]:
            print(f"  K={k:2d}: " + _fmt(*pooled7[k]))

    print("\n" + "=" * W)
    print("loss distribution vs independent model".ljust(W))
    print("-" * W)
    print(f"  fitted p_stationary={loss['p_stat']:.4f}  p_moved={loss['p_move']:.4f}")
    print(
        f"  {'K':>3} {'M':>3} {'n':>5}  {'<loss>':>7} {'var_obs':>8} {'var_indep':>9}  verdict"
    )
    for (k, m), c in sorted(loss["cells"].items()):
        if c["n"] < 5:
            continue
        mean = (c["obs"] * np.arange(len(c["obs"]))).sum() / c["n"]
        ratio = c["var_obs"] / c["var_exp"] if c["var_exp"] else np.nan
        verdict = (
            "over-dispersed"
            if ratio > 1.5
            else "under"
            if ratio < 0.67
            else "consistent"
        )
        print(
            f"  {k:3d} {m:3d} {c['n']:5d}  {mean:7.3f} {c['var_obs']:8.3f} "
            f"{c['var_exp']:9.3f}  {verdict}"
        )
    print(
        "  over-dispersion => shot-to-shot common-mode loss (MOT drift, bad dark\n"
        "  window, one flaky arm), not independent per-atom loss."
    )

    if ps is not None:
        print("\n" + "=" * W)
        print("per-site vs global thresholds".ljust(W))
        print("-" * W)
        print(
            f"  reclassified site-shots : {ps['flips']} / {ps['site_shots']} "
            f"({100 * ps['flips'] / max(1, ps['site_shots']):.3f}%)"
        )
        print(
            f"  shots whose PLAN differs: {ps['plan_changed']} / {ps['shots']} "
            f"({100 * ps['plan_changed'] / max(1, ps['shots']):.2f}%)   "
            f"[K changed {ps['k_changed']}, block moved {ps['t0_changed']}]"
        )

    if aligns:
        print("\n" + "=" * W)
        print("REPORT ALIGNMENT (a check, not a dependency)".ljust(W))
        print("-" * W)
        for name, a in aligns.items():
            if not a:
                continue
            print(
                f"  {name}: {a['aligned']} aligned, {a['exact']} exact, "
                f"{a.get('mismatched', 0)} mask mismatches"
            )
            if a.get("unreported_with_moves"):
                print(
                    f"    ! {len(a['unreported_with_moves'])} shots have moves but no "
                    f"report -> alignment or threshold is off: {a['unreported_with_moves'][:12]}"
                )
            if a.get("reported_without_moves"):
                print(
                    f"    ! {len(a['reported_without_moves'])} reported shots need no "
                    f"move: {a['reported_without_moves'][:12]}"
                )
    print("=" * W + "\n")


def print_run_table(rows, title="PER-RUN COMPARISON"):
    """same here - just some stats to compare runs - claude's print style"""
    W = 108
    print("\n" + "=" * W)
    print(title.ljust(W))
    print("-" * W)
    print(
        f"  {'run':38s} {'shots':>5} {'<K>':>5} {'load':>5}  {'stationary':>16} "
        f"{'moved':>16} {'block':>7} {'window':>8} {'misfit':>7}"
    )
    for r in rows:
        ps, plo, phi = wilson(r["stat_k"], r["stat_n"])
        ms, mlo, mhi = wilson(r["move_k"], r["move_n"])
        fs, flo, fhi = wilson(r["perfect"], r["perfect_n"])
        st = f"{ps:.3f}+-{max(phi - ps, ps - plo):.3f}" if r["stat_n"] else "     -    "
        mv = f"{ms:.3f}+-{max(mhi - ms, ms - mlo):.3f}" if r["move_n"] else "     -    "
        pf = f"{fs:.3f}" if r["perfect_n"] else "  -  "
        w = "auto" if r.get("start_window") is None else str(r["start_window"])
        mf = r.get("window_misfit", float("nan"))
        print(
            f"  {r['name'][:38]:38s} {r['shots']:5d} {r['K']:5.2f} {r['loading']:5.3f}  "
            f"{st:>16} {mv:>16} {pf:>7} {w:>8} "
            + ("      -" if mf != mf else f"{mf:7.3f}")
        )
    print(
        "  stationary = P(survive | did not move), moved = P(arrived at target | had to "
        "move),\n  block = P(whole target block occupied in frame 2). window = the "
        "start_window inferred\n  from frame 2; misfit = fraction of frame-2 atoms "
        "outside the predicted block, which\n  should be ~0 if the reconstruction of "
        "that run is faithful."
    )


def main(argv=None):
    """
    For debugging with claude I made this main script. I will not use it, but
    it helped claude understand the functions and so on. I will use the plot
    scripts under this folder where I understand what happens instead of this
    with all the parsing of arguments and cyclical imports and so on.
    """

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--images", help="flat folder holding all the run .npy files")
    ap.add_argument("--cal", help="calibration folder (default <images>/calibration)")
    ap.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="substrings selecting which run prefixes to include (default all)",
    )
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument(
        "--min-shots",
        type=int,
        default=20,
        help="ignore prefixes with fewer shots than this",
    )
    ap.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="subsample each run, for a quick pass",
    )
    ap.add_argument("--list", action="store_true", help="list the runs found and exit")
    ap.add_argument(
        "--reports", nargs="*", default=[], help="report files, globs or a folder"
    )
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the live atom_threshold (normally DO NOT)",
    )
    ap.add_argument(
        "--start-window",
        type=int,
        default=None,
        help="force this start_window for every run (implies --no-auto-window)",
    )
    ap.add_argument(
        "--no-auto-window",
        action="store_true",
        help="do not infer each run's start_window from frame 2",
    )
    ap.add_argument(
        "--per-site",
        action="store_true",
        help="redo the whole analysis with per-site thresholds (item 9 variant)",
    )
    ap.add_argument(
        "--per-site-file",
        default=None,
        help="npz from tune_thresholds.py --save; default <cal>/per_site_thresholds.npz",
    )
    ap.add_argument(
        "--no-sorter", action="store_true", help="ignore sorter.py's planner"
    )
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument(
        "--k-bands",
        nargs="*",
        default=["1-3", "4-6", "7-9"],
        metavar="LO-HI",
        help="K intervals to pool over, e.g. --k-bands 4-6 7-9 2-4",
    )
    ap.add_argument(
        "--per-run-figs",
        action="store_true",
        help="also write the item-5 and item-7 figures for every run separately",
    )
    ap.add_argument(
        "--merge",
        nargs="*",
        default=[],
        metavar="NAME:SUB1,SUB2",
        help="pool runs into one dataset, e.g. --merge 80us:bestsort-80us,80usAGAIN",
    )
    ap.add_argument(
        "--k-match",
        nargs=2,
        type=int,
        default=[4, 6],
        metavar=("LO", "HI"),
        help="also compare runs restricted to this K band, so that runs taken "
        "at different loading rates are compared at matched atom number",
    )
    a = ap.parse_args(argv)

    if not a.images:
        ap.error("--images is required")
    if not a.cal:
        a.cal = os.path.join(a.images, "calibration")

    print(f"sorter.py {'imported' if S else 'NOT importable -> internal fallbacks'}")
    groups = discover_runs(a.images, a.min_shots)
    if a.runs:
        groups = {k: v for k, v in groups.items() if any(s in k for s in a.runs)}
    if a.exclude:
        groups = {k: v for k, v in groups.items() if not any(s in k for s in a.exclude)}
    order = sorted(groups, key=lambda k: groups[k][0][1])  # chronological
    print(f"  {len(order)} run(s) in {a.images}:")
    for k in order:
        print(
            f"    {k:44s} {len(groups[k]):4d} shots   {groups[k][0][1]} .. {groups[k][-1][1]}"
        )
    if a.list:
        return 0

    cal = load_calibration(a.cal, a.threshold)
    cal_entries = sum(discover_runs(a.cal, 1, verbose=False).values(), [])
    cal_ds = load_dataset("calibration", cal_entries, cal, is_calibration=True)

    if a.per_site:
        ps_file = a.per_site_file or os.path.join(a.cal, "per_site_thresholds.npz")
        if os.path.exists(ps_file):
            d = np.load(ps_file, allow_pickle=False)
            key = next(
                k
                for k in ("per_site_threshold", "per_site_thresholds", "thresholds")
                if k in d.files
            )
            cal.per_site = np.asarray(d[key], float)
            print(f"  per-site thresholds loaded from {ps_file}")
        else:
            print("  per-site thresholds: fitting from the calibration set")
            cal.per_site = per_site_thresholds(
                cal_ds.counts[:, LOAD_FRAME, :], cal.threshold
            )

    sw = a.start_window if (a.no_auto_window or a.start_window is not None) else "auto"
    build_plans(
        cal_ds,
        a.per_site,
        not a.no_sorter,
        a.start_window if a.start_window is not None else None,  # type: ignore
    )

    run_dss = []
    for k in order:
        e = groups[k]
        if a.max_shots:
            e = e[: a.max_shots]
        ds = load_dataset(k, e, cal)
        build_plans(ds, a.per_site, not a.no_sorter, sw)
        run_dss.append(ds)

    kbands = []
    for b in a.k_bands:
        try:
            lo_, hi_ = (int(x) for x in b.replace(":", "-").split("-"))
            kbands.append((lo_, hi_))
        except Exception:  # noqa: BLE001
            print(f"  ! could not parse --k-bands entry {b!r}, skipping")

    for spec in a.merge:
        try:
            nm, subs = spec.split(":", 1)
            subs = [x for x in subs.split(",") if x]
        except Exception:  # noqa: BLE001
            print(f"  ! could not parse --merge entry {spec!r}")
            continue
        part = [d for d in run_dss if any(x in d.name for x in subs)]
        if len(part) < 2:
            print(f"  ! --merge {nm}: matched {len(part)} run(s), need >=2")
            continue
        merged = Dataset(
            nm,
            sum((d.paths for d in part), []),  # noqa: RUF017
            np.concatenate([d.shot_idx for d in part]),
            sum((d.stamps for d in part), []),  # noqa: RUF017
            np.concatenate([d.counts for d in part]),
            part[0].mean_frames,
            cal,
        )
        build_plans(merged, a.per_site, not a.no_sorter, sw)
        run_dss = [d for d in run_dss if not any(d is q for q in part)] + [merged]
        print(f"  merged {len(part)} runs into {nm!r}: {merged.n_shots} shots")

    aligns = {}
    if a.reports and run_dss:
        reports = read_reports(a.reports, cal.n)
        for ds in run_dss:
            ds.mapping, aligns[ds.name] = align_reports(ds.plans, reports, cal.n)

    surv = survival_tables(cal_ds, run_dss, a.per_site)
    per_ds7, pooled7 = success_vs_k(run_dss, a.kmax, a.per_site)
    loss = loss_distributions(run_dss, a.per_site)
    drift = loading_drift(cal_ds, run_dss, a.per_site)
    ps = per_site_impact(cal_ds, run_dss, not a.no_sorter) if run_dss else None
    rows = run_table(cal_ds, run_dss, a.per_site, a.kmax)
    rows_k = run_table(cal_ds, run_dss, a.per_site, a.kmax, tuple(a.k_match))

    print_report(surv, per_ds7, pooled7, loss, ps, aligns)
    print_run_table(rows)
    print_run_table(
        rows_k,
        f"PER-RUN COMPARISON AT MATCHED LOADING, K in [{a.k_match[0]}, {a.k_match[1]}]",
    )
    print(
        "  The loading rate drifted from 0.48 to 0.11 over the session (item 10), so K\n"
        "  covaries with whatever was being varied between runs. Survival depends much\n"
        "  more strongly on K than on move distance, so THIS table is the fair\n"
        "  comparison between transport settings; the one above is not."
    )
    os.makedirs(a.out, exist_ok=True)
    gh = ghost_stats(run_dss, a.per_site)
    from ExpectationsOfDelivery import fig_success_vs_k
    from FidelityOverDistance import fig_survival_vs_distance
    from PerRunComparison import fig_run_comparison

    saved = make_summary_figures(
        a.out, surv, per_ds7, pooled7, loss, drift, ps, cal_ds, kbands, run_dss, cal.n
    )
    print_ghosts(gh)
    asym = planner_asymmetry(cal.n)
    print(
        f"\n  planner asymmetry check on random masks (n={cal.n}, p=0.5, 2e5 draws): "
        f"{100 * asym['left_frac']:.1f}% of moves go LEFT, mean distance "
        f"{asym['mean_left_dist']:.2f} left vs {asym['mean_right_dist']:.2f} right.\n"
        "  The left/right imbalance is a property of find_best_window, not of the atoms."
    )

    if a.per_run_figs:
        sub = os.path.join(a.out, "per_run")
        os.makedirs(sub, exist_ok=True)
        import matplotlib.pyplot as plt

        for ds in run_dss:
            tag = ds.name.replace("tweezerLoad1x11-", "").replace("twezerLoad1x11-", "")
            s5 = survival_tables(cal_ds, [ds], a.per_site)
            f = fig_survival_vs_distance(s5)
            if f is not None:
                q = os.path.join(sub, f"item5_{tag}.png")
                f.savefig(q, dpi=110), plt.close(f), saved.append(q)
            pd7, pl7 = success_vs_k([ds], a.kmax, a.per_site)
            f = fig_success_vs_k(pd7, pl7, kbands, f"Item 7 -- {ds.name}")
            if f is not None:
                q = os.path.join(sub, f"item7_{tag}.png")
                f.savefig(q, dpi=110), plt.close(f), saved.append(q)

    import matplotlib.pyplot as plt

    for r_, nm in (
        (rows, "item7_per_run_comparison.png"),
        (rows_k, "item7_per_run_comparison_kmatched.png"),
    ):
        f = fig_run_comparison(r_)
        if f is not None:
            q = os.path.join(a.out, nm)
            f.savefig(q, dpi=130), plt.close(f), saved.append(q)

    summary = {
        "images": a.images,
        "calibration_folder": a.cal,
        "n_sites": int(cal.n),
        "atom_threshold": float(cal.threshold),
        "per_site_analysis": bool(a.per_site),
        "survival": {
            k: {
                "k": int(surv[k]["k"]),
                "n": int(surv[k]["n"]),
                "p": wilson(surv[k]["k"], surv[k]["n"])[0],
            }
            for k in ("cal", "stat", "move")
        },
        "survival_by_distance": {
            str(d): {"k": int(v["k"]), "n": int(v["n"])}
            for d, v in sorted(surv["by_dist"].items())
        },
        "success_vs_K": {str(k): pooled7[k] for k in pooled7 if pooled7[k][1]},
        "p_stat": loss["p_stat"],
        "p_move": loss["p_move"],
        "runs": rows,
        "runs_k_matched": rows_k,
        "ghosts": {
            k: {
                kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                for kk, vv in v.items()
                if kk != "by_k"
            }
            for k, v in gh.items()
        },
        "planner_asymmetry": asym,
        "k_bands": [list(b) for b in kbands],
        "k_match": list(a.k_match),
        "per_site": None
        if ps is None
        else {
            "flips": int(ps["flips"]),
            "site_shots": int(ps["site_shots"]),
            "plan_changed": int(ps["plan_changed"]),
            "shots": int(ps["shots"]),
            "thresholds": ps["thresholds"].tolist(),
        },
        "alignment": {
            k: {
                kk: (vv if not isinstance(vv, list) else len(vv))
                for kk, vv in v.items()
            }
            for k, v in aligns.items()
        },
    }
    json.dump(
        summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float
    )
    print("\nwrote:")
    for p in saved + [os.path.join(a.out, "summary.json")]:
        print("  " + p)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
