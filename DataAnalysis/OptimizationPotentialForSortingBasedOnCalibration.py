"""
The sorting algorithm from ../Sorting/sorter.py has essentially
two modes of operation (for the sorting). The first mode is running
in the best window mode where the algorithm will try to find the
target window that requires the least amount of transport moves.
The second mode is given a specific target window get the moves.
The "best" window is only the best if the distance does not matter,
or if target and source are not correlated or specific sites have
worse/better fidelities for imaging or that sort, which can also
be correlated given the number of loaded atoms and number of tones.

To look at whether the survival rate could be made better in
general, one needs to calibrate using a lot of runs for statistical
relevance - we look at the comparison moving all atoms to the left/
top or right/bottom to the "best" to one experimental run, making
some small computations about probabilities and given those what
the best decision would be given the exact loading positions and
statistics.

We do bar plots sticking out from the y axis (what determined the
final/target window) with the x axis being the probability of
survival in that specific mode. (This of course then has to be
validated and taken with a grain of salt, but this might be a
way to train an AI to make quick decisions on how to create a
defect free array, learning from real data and apparatus specific
responses/alignments/... ).

@author: Bjarne Schümann
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import CommonThings as C
import matplotlib.pyplot as plt

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D
MIN_N_MODEL = 10  # minimum shots per (K, d) cell before the survival model trusts it


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> list of built Dataset objects."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    names = (
        sorted(groups)
        if not runs
        else [k for k in sorted(groups) if any(r in k for r in runs)]
    )
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return run_dss


def _candidate_windows(p, n):
    return [
        ("current (live 'best window')", p.t0),
        ("forced left / top", 0),
        ("forced right / bottom", n - p.k),
    ]


def _survival_model(run_dss, shots, m2_by_ds):
    """p(K, |d|) from a pool of (dataset index, shot index) pairs, with K+-1/+-2 backoff."""
    tab = {}
    for di, i in shots:
        p = run_dss[di].plans[i]
        m2 = m2_by_ds[di]
        for a in range(p.k):
            d = abs(int(p.dist[a]))
            c = tab.setdefault((p.k, d), [0, 0])
            c[1] += 1
            c[0] += bool(m2[i, int(p.targets[a])])
    glob = (
        np.mean([c[0] / c[1] for c in tab.values() if c[1] >= MIN_N_MODEL])
        if tab
        else 0.5
    )

    def ps(k, d):
        for kk in (k, k - 1, k + 1, k - 2, k + 2):
            c = tab.get((kk, d))
            if c and c[1] >= MIN_N_MODEL:
                return c[0] / c[1]
        return glob

    return ps


def _counterfactual(run_dss, seed=0):
    """2-fold cross-validated E[every atom arrives] for each window strategy, plus the
    fraction of hops that are a single site (context for how conservative a strategy is)."""
    n = run_dss[0].cal.n
    m2_by_ds = [ds.occ(C.SURV_FRAME) for ds in run_dss]
    all_shots = [
        (di, i)
        for di, ds in enumerate(run_dss)
        for i in range(ds.n_shots)
        if ds.plans[i].k > 0
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_shots))
    half = len(idx) // 2
    res, hops = {}, {}
    for tr_idx, te_idx in ((idx[:half], idx[half:]), (idx[half:], idx[:half])):
        ps = _survival_model(run_dss, [all_shots[j] for j in tr_idx], m2_by_ds)
        for j in te_idx:
            di, i = all_shots[j]
            p = run_dss[di].plans[i]
            occ = p.sites
            for name, t0 in _candidate_windows(p, n):
                exp = float(
                    np.prod([ps(p.k, abs(int((t0 + a) - occ[a]))) for a in range(p.k)])  # type: ignore
                )
                res.setdefault(name, []).append(exp)
                d = np.abs((t0 + np.arange(p.k)) - occ)
                d = d[d > 0]
                hops.setdefault(name, []).extend(d.tolist())
    summary = {k: (float(np.mean(v)), len(v)) for k, v in res.items()}
    hop1_frac = {
        k: float(np.mean(np.array(v) == 1)) if v else 0.0 for k, v in hops.items()
    }
    return summary, hop1_frac


def _bar_figure(summary, hop1_frac):
    """Horizontal bars: strategy on y, E[every atom arrives] on x."""
    names = list(summary)
    vals = [summary[k][0] for k in names]
    cols = [
        "0.5" if "current" in k else ("tab:blue" if "left" in k else "tab:orange")
        for k in names
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(range(len(names)), vals, color=cols, edgecolor="k")
    for i, k in enumerate(names):
        ax.text(
            vals[i] + 0.01,
            i,
            f"{vals[i]:.3f}   ({100 * hop1_frac[k]:.0f}% of hops d=1, n={summary[k][1]})",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("E[every atom arrives]  (2-fold cross-validated)")
    ax.set_title("Target-window strategy vs survival potential")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot(runs=None, save=True, seed=0, images=None, name=None):
    """Compares the live 'best window' choice against always sorting to the left/top or
    right/bottom edge. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    summary, hop1_frac = _counterfactual(run_dss, seed=seed)
    fig = _bar_figure(summary, hop1_frac)
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "optimization_potential_vs_calibration")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--seed", type=int, default=0, help="cross-validation split seed")
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        for ds in _load(a.runs, a.images):
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    plot(runs=a.runs, save=not a.no_save, seed=a.seed, images=a.images)
    if a.no_save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
