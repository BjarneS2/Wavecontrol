"""
This script compares per-site atom thresholds vs the single global one, how many site-shots
each site flips when switching from global to per-site classification, and
(as a related sanity check) per-site imaging survival on the calibration set alone.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D


def fig_per_site_thresholds(ps):
    """
    per-site thresholds vs the global one, and how many site-shots each
    site flips when switching from global to per-site classification
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    n = len(ps["thresholds"])
    a1.bar(np.arange(n), ps["thresholds"], color="tab:blue", edgecolor="k", alpha=0.8)
    a1.axhline(ps["global"], color="k", ls="--", label=f"global {ps['global']:.1f}")
    a1.set_xlabel("site")
    a1.set_ylabel("atom threshold")
    a1.set_title("per-site vs global threshold")
    a1.legend(fontsize=8, framealpha=0.95)
    a2.bar(
        np.arange(n), ps["per_site_flips"], color="tab:red", edgecolor="k", alpha=0.8
    )
    a2.set_xlabel("site")
    a2.set_ylabel("site-shots reclassified")
    a2.set_title(
        f"{ps['flips']} of {ps['site_shots']} site-shots flip "
        f"({100 * ps['flips'] / max(1, ps['site_shots']):.2f}%)\n"
        f"plan differs in {ps['plan_changed']}/{ps['shots']} shots"
    )
    fig.tight_layout()
    return fig


def fig_per_site_calibration_survival(surv):
    """Per-site imaging survival on the calibration set only -- look for one bad arm.
    None if the survival table has no per-site calibration breakdown."""
    if surv["cal"]["per_site_n"] is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    k_, n_ = surv["cal"]["per_site_k"], surv["cal"]["per_site_n"]
    y = [A.wilson(int(a), int(b)) for a, b in zip(k_, n_)]
    ax.errorbar(
        np.arange(len(y)),
        [v[0] for v in y],
        yerr=[[v[0] - v[1] for v in y], [v[2] - v[0] for v in y]],
        marker="o",
        capsize=3,
        lw=1,
    )
    ax.set_xlabel("site")
    ax.set_ylabel("survival (calibration)")
    ax.set_title("per-site imaging survival -- look for one bad arm")
    ax.grid(alpha=0.3)
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
    ps = A.per_site_impact(cal_ds, run_dss)
    surv = C.survival_tables(cal_ds, run_dss)
    figs = [fig_per_site_thresholds(ps), fig_per_site_calibration_survival(surv)]
    if save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        titles = ["per_site_thresholds", "per_site_calibration_survival"]
        for fig, title in zip(figs, titles):
            if fig is not None:
                C.save_figure(fig, CATEGORY, name, title)
    return figs


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
