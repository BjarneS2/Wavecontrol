"""
I am here to evaluate if it moving is the cost, or the imaging? Compares survival
for calibration (imaging only), stationary run atoms, and moved run atoms,
plus the fate (arrived / still at origin / gone) of atoms that had to move.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D


def fig_survival_stationary_vs_moved(surv):
    """
    survival for calibration / stationary / moved atoms, plus the fate
    (arrived / still at origin / gone) of atoms that had to move.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["calibration\n(imaging only)", "run: stationary", "run: moved"]
    keys = ["cal", "stat", "move"]
    ps_, los, his = [], [], []
    for k in keys:
        p, lo, hi = A.wilson(surv[k]["k"], surv[k]["n"])
        ps_.append(p)
        los.append(p - lo)
        his.append(hi - p)
    ax.bar(
        labels,
        ps_,
        yerr=[los, his],
        capsize=5,
        color=["0.6", "tab:blue", "tab:red"],
        edgecolor="k",
    )
    for i, k in enumerate(keys):
        ax.text(i, 0.02, f"n={surv[k]['n']}", ha="center", fontsize=8)
    ax.set_ylabel("survival  P(in frame 2 | in frame 1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Is moving the cost, or is it imaging?")
    ax.grid(axis="y", alpha=0.3)
    f = surv["mover_fate"]
    tot = max(1, sum(f.values()))
    ax2.bar(
        list(f.keys()),
        [v / tot for v in f.values()],
        color=["tab:green", "tab:orange", "0.4"],
        edgecolor="k",
    )
    ax2.set_title("fate of atoms that had to move")
    ax2.set_ylabel("fraction")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> (cal_ds, run_dss) built from the live Sorting1D calibration."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    cal_entries = sum(  # noqa
        C.discover_runs(str(C.SORTING1D_CAL), 1, verbose=False).values(), []
    )
    cal_ds = C.load_dataset(
        "calibration", cal_entries, cal, is_calibration=True, verbose=False
    )
    C.build_plans(cal_ds, verbose=False)

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
    return cal_ds, run_dss


def plot(runs=None, save=True, images=None, name=None):
    cal_ds, run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    surv = C.survival_tables(cal_ds, run_dss)
    fig = fig_survival_stationary_vs_moved(surv)
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "survival_stationary_vs_moved")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        _, run_dss = _load(a.runs, a.images)
        for ds in run_dss:
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    plot(runs=a.runs, save=not a.no_save, images=a.images)
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
