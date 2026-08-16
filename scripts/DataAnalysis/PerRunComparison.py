"""
This script will analyze how the different runs of the sorting experiment compare to each other
given K in [...]:
Run indexes are given on the y axis and the survival probabilities on the x axis. The calibration
is also shown as a reference (dotted grey line) showing the survival after just taking images or
rather it shows the fidelity of the imaging process.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D


def fig_run_comparison(rows):
    """
    Horizontal errorbar comparison of stationary vs moved survival per run, with
    the calibration floor as a vertical reference line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in rows if r["move_n"] > 0]
    if not runs:
        return None
    runs.sort(key=lambda r: -A.wilson(r["move_k"], r["move_n"])[0])
    cal = next((r for r in rows if r["move_n"] == 0 and r["stat_n"] > 0), None)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(runs) + 2.2))
    y = np.arange(len(runs))
    for tag, c, lab, off in (
        ("stat", "tab:blue", "stationary", 0.18),
        ("move", "tab:red", "moved", -0.18),
    ):
        p = [A.wilson(r[tag + "_k"], r[tag + "_n"])[0] for r in runs]
        e = [
            max(
                A.wilson(r[tag + "_k"], r[tag + "_n"])[2]
                - A.wilson(r[tag + "_k"], r[tag + "_n"])[0],
                0,
            )
            for r in runs
        ]
        ax.errorbar(
            p, y + off, xerr=e, fmt="o", ms=5, color=c, label=lab, capsize=3, lw=1
        )
    if cal is not None:
        pc = A.wilson(cal["stat_k"], cal["stat_n"])[0]
        ax.axvline(pc, color="0.4", ls="--", label=f"calibration floor {pc:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [r["name"].replace("tweezerLoad1x11-", "") for r in runs], fontsize=8
    )
    ax.set_xlabel("survival")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    return fig


def _load(runs=None, images=None, min_shots=20, exclude=None):
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
    if exclude:
        names = [k for k in names if not any(x in k for x in exclude)]
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return cal_ds, run_dss


def plot(runs=None, save=True, k_range=None, images=None, name=None, exclude=None):
    """Per-run stationary/moved survival, calibration floor as reference. Returns the Figure."""
    cal_ds, run_dss = _load(
        runs if runs is not None else DEFAULT_RUNS, images, exclude=exclude
    )
    rows = C.run_table(cal_ds, run_dss, k_range=k_range)
    fig = fig_run_comparison(rows)
    if fig is not None and save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "per_run_comparison")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--k-range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    ap.add_argument(
        "--exclude", nargs="*", default=None, help="substrings excluding run prefixes"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        _, run_dss = _load(a.runs, a.images, exclude=a.exclude)
        for ds in run_dss:
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    fig = plot(
        runs=a.runs,
        save=not a.no_save,
        k_range=a.k_range,
        images=a.images,
        exclude=a.exclude,
    )
    if fig is None:
        print("no runs with movers matched -- nothing to plot")
        return 1
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
