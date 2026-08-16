"""
This script analyzes the data from the single atom transport experiment.
To make this a short explanation what happens here exactly: We take the
individual experiment runs and plot the survival of the atoms over the
distance that they have moved. All in the nice viridis color scale. But
make it so that it is plotted the following way: 1st sort by K (loaded)
and then plot given K the distance d what is the survival rate. Those
should be one curve with error bars of course. If they are flat it means
distance is not the reason for losses. Then this is done for the
individual runs where the functions of K are plotted in viridis colors
and then one can also take the average over all linear and all STA runs
and see if that is also the case for the pooled trajectories. (In the
old version this I think was just all runs automatically averaged and
evaluated together).

I think one can then also just make a curve in this file too with surv.
of the atoms given linear or sta trajectories over K. So y axis being
survival rate of the atoms, no matter how far they have traveled and the
x axis being K, the number of loaded atoms, and showing that for linear
and sta and then show 2 curves: the stationary and the moved atoms. This
plot should then also be shown for individual runs of course.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C
import numpy as np

# from plots import fig_survival_vs_distance

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D
PER_RUN_DEFAULT = True


def fig_survival_vs_distance(surv):
    """
    Two panels: survival vs hop length S at fixed K, and survival vs K. Callable
    per run or on the pooled set.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = sorted(surv["by_dist"])
    if not d:
        return None
    pc, lo, hi = A.wilson(surv["cal"]["k"], surv["cal"]["n"])
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 4.4))

    ks = sorted({k for k, _ in surv["by_kd"]})
    cmap = plt.get_cmap("viridis")
    span = max(1, (max(ks) - min(ks)) if ks else 1)
    for k in ks:
        xs = [
            dd
            for (kk, dd) in sorted(surv["by_kd"])
            if kk == k and surv["by_kd"][(kk, dd)][1] >= 8
        ]
        if len(xs) < 2:
            continue
        y, el, eh = [], [], []
        for dd in xs:
            pp, lo, hi = A.wilson(*surv["by_kd"][(k, dd)])
            y.append(pp)
            el.append(pp - lo)
            eh.append(hi - pp)
        ax2.errorbar(
            xs,
            y,
            yerr=[el, eh],
            marker="o",
            ms=4,
            capsize=3,
            lw=1.4,
            color=cmap((k - min(ks)) / span),
            label=f"K={k}",
        )
    ax2.set_xlabel("S: hop length [sites]")
    ax2.set_ylabel("survival")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, ncol=2, framealpha=0.95)
    ax2.grid(alpha=0.3)
    ax2.text(
        0.02, 0.98, "(a)", transform=ax2.transAxes, fontsize=12,
        ha="left", va="top", weight="bold",
    )

    kk_ = sorted(surv["by_k"])
    for tag, c, lab in (
        ("stat", "tab:blue", "stationary"),
        ("move", "tab:red", "moved"),
    ):
        xs, y, el, eh = [], [], [], []
        for k in kk_:
            a_, n_ = surv["by_k"][k][tag]
            if n_ < 8:
                continue
            pp, lo, hi = A.wilson(a_, n_)
            xs.append(k)
            y.append(pp)
            el.append(pp - lo)
            eh.append(hi - pp)
        ax3.errorbar(
            xs, y, yerr=[el, eh], marker="o", capsize=4, color=c, label=lab, lw=2
        )
    if np.isfinite(pc):
        ax3.axhline(pc, ls="--", color="0.5", label="calibration floor")
    ax3.set_xlabel("K: atoms loaded")
    ax3.set_ylabel("per-atom survival")
    ax3.set_ylim(0, 1.05)
    if kk_:
        ax3.set_xticks(kk_)
    ax3.legend(fontsize=8, framealpha=0.95)
    ax3.grid(alpha=0.3)
    ax3.text(
        0.02, 0.98, "(b)", transform=ax3.transAxes, fontsize=12,
        ha="left", va="top", weight="bold",
    )
    fig.tight_layout()
    return fig


def trajectory_of(name):
    """ "lin" substring -> linear trajectory, everything else is sta"""
    return "linear" if "lin" in name else "sta"


def _load(runs=None, images=None, min_shots=20):
    """--runs substrings -> (cal_ds, run_dss), plans already built for both."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    cal_entries = sum(  # noqa: RUF017
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


def plot(runs=None, save=True, per_run=PER_RUN_DEFAULT, images=None):
    cal_ds, run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    figs = []

    by_trajectory = {
        "all": run_dss,
        "linear": [ds for ds in run_dss if trajectory_of(ds.name) == "linear"],
        "sta": [ds for ds in run_dss if trajectory_of(ds.name) == "sta"],
    }
    for tag, dss in by_trajectory.items():
        if not dss:
            continue
        surv = C.survival_tables(cal_ds, dss)
        fig = fig_survival_vs_distance(surv)
        if fig is None:
            continue
        figs.append(fig)
        if save:
            run_name = C.POOLED if tag == "all" else f"{tag}_pooled"
            C.save_figure(fig, CATEGORY, run_name, "fidelity_over_distance")

    if per_run:
        for ds in run_dss:
            surv = C.survival_tables(cal_ds, [ds])
            fig = fig_survival_vs_distance(surv)
            if fig is None:
                continue
            figs.append(fig)
            if save:
                C.save_figure(fig, CATEGORY, ds.name, "fidelity_over_distance")
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument(
        "--no-per-run", action="store_true", help="skip the individual per-run figures"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        _, run_dss = _load(a.runs, a.images)
        for ds in run_dss:
            print(f"  {ds.name}  ({trajectory_of(ds.name)}, {ds.n_shots} shots)")
        return 0

    figs = plot(
        runs=a.runs, save=not a.no_save, per_run=not a.no_per_run, images=a.images
    )
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    print(f"built {len(figs)} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
