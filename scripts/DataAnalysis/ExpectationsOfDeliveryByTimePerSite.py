"""
Same observable as ExpectationsOfDelivery.py -- P(target block defect-free | K) --
but restricted to the runs that differ only in the time spent per site, so the
curves can be labelled by that time. The two 80 us runs are additionally marked
as linear / minimum jerk. The pooled black curve is the pool of exactly these
runs, and the dotted line is the calibration floor p_cal^K, i.e. what a lossless
transport would still give up to imaging loss alone. Two versions are written:
one with error bars on the pooled curve only, one with Wilson bars on every
curve (..._errorbars).

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING
FIG_NAME = "expectations_of_delivery_by_time_per_site"
MIN_N = 5  # per-run points with fewer shots than this are dropped (--min-n 1 keeps all)

# (dataset name after merging, legend label, color, marker) -- slowest first.
# Colors are the Okabe-Ito colorblind-safe set; the marker is the second encoding.
RUNS = [
    ("tweezerLoad1x11-sortbest-100us", "100 µs/site", "#0072B2", "o"),
    ("tweezerLoad1x11-80us_pooled", "80 µs/site (minimum jerk)", "#009E73", "s"),
    ("tweezerLoad1x11-sortbest-lin80us", "80 µs/site (linear)", "#D55E00", "D"),
    # ("twezerLoad1x11-sortbest-50us", "50 µs/site", "#CC79A7", "^"),
    ("tweezerLoad1x11-sortbest-lin40us", "40 µs/site", "#56B4E9", "v"),
    ("tweezerLoad1x11-sortbest-20us", "20 µs/site", "#E69F00", "P"),
]


def _load(images=None, min_shots=20):
    """(cal_ds, {name: Dataset}) for the runs in RUNS only, merge groups honoured."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    cal_entries = sum(  # noqa: RUF017
        C.discover_runs(str(C.SORTING1D_CAL), 1, verbose=False).values(), []
    )
    cal_ds = C.load_dataset(
        "calibration", cal_entries, cal, is_calibration=True, verbose=False
    )
    C.build_plans(cal_ds, verbose=False)

    wanted = {name for name, *_ in RUNS}
    sub_to_merged = {s: g[-1] for g in C.MERGE_GROUPS for s in g[:-1]}
    groups = C.discover_runs(images, min_shots, verbose=False)
    names = [n for n in sorted(groups) if sub_to_merged.get(n, n) in wanted]
    run_dss = []
    for name in names:
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    run_dss = C.merge_run_datasets(run_dss)
    return cal_ds, {ds.name: ds for ds in run_dss}


def fig_success_vs_k(
    per_ds, pooled, p_cal=np.nan, annotate_n=True, min_n=MIN_N, per_run_errors=False
):
    """P(defect-free block | K) for each selected run, the pool of them, and the
    calibration floor p_cal^K. per_run_errors adds Wilson bars to the run curves too."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, label, color, marker in RUNS:
        tab = per_ds.get(name)
        if not tab:
            continue
        ks = [k for k in sorted(tab) if tab[k][1] >= min_n]
        if not ks:
            continue
        p = [C.wilson(*tab[k]) for k in ks]
        ax.errorbar(
            ks,
            [q[0] for q in p],
            yerr=[[q[0] - q[1] for q in p], [q[2] - q[0] for q in p]]
            if per_run_errors
            else None,
            marker=marker,
            ms=5,
            lw=1.6,
            alpha=0.9,
            color=color,
            capsize=2.5,
            elinewidth=1,
            label=label,
        )

    ks = [k for k in sorted(pooled) if pooled[k][1] > 0]
    y, el, eh = [], [], []
    for k in ks:
        pp, lo, hi = C.wilson(*pooled[k])
        y.append(pp)
        el.append(pp - lo)
        eh.append(hi - pp)
    ax.errorbar(
        ks, y, yerr=[el, eh], marker="o", color="k", lw=2.4, capsize=4, label="pooled"
    )
    if np.isfinite(p_cal):
        ax.plot(
            ks,
            [p_cal**k for k in ks],
            ls=":",
            lw=2,
            color="0.35",
            label=r"calibration floor $p_\mathrm{cal}^{K}$",
        )
    if annotate_n:
        for k, yy in zip(ks, y):
            ax.annotate(
                f"{pooled[k][1]}",
                (k, yy),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="0.1",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
    ax.set_xlabel("K atoms loaded")
    ax.set_ylabel("P(target block defect-free in frame 2)")
    ax.set_ylim(-0.02, 1.02)
    if ks:
        ax.set_xticks(ks)
    h, lab = ax.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda i: "calibration" in lab[i])
    ax.legend(
        [h[i] for i in order], [lab[i] for i in order], fontsize=8, framealpha=0.95
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot(
    save=True, kmax=10, images=None, name=None, annotate_n=True, min_n=MIN_N, ext="svg"
):
    """Both variants -- pooled error bars only, and error bars on every curve.
    Returns the two Figures."""
    cal_ds, by_name = _load(images)
    missing = [n for n, *_ in RUNS if n not in by_name]
    if missing:
        print("  missing run(s): " + ", ".join(missing))
    run_dss = [by_name[n] for n, *_ in RUNS if n in by_name]
    per_ds, pooled = C.success_vs_k(run_dss, kmax=kmax)
    surv = C.survival_tables(cal_ds, [])
    p_cal = C.wilson(surv["cal"]["k"], surv["cal"]["n"])[0]
    figs = []
    for per_run_errors, title in ((False, FIG_NAME), (True, FIG_NAME + "_errorbars")):
        fig = fig_success_vs_k(
            per_ds,
            pooled,
            p_cal=p_cal,
            annotate_n=annotate_n,
            min_n=min_n,
            per_run_errors=per_run_errors,
        )
        figs.append(fig)
        if save:
            print(f"  {C.save_figure(fig, CATEGORY, name or 'time_per_site', title, ext=ext)}")
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images", default=None, help="override the Sorting1D images folder"
    )
    ap.add_argument("--kmax", type=int, default=10, help="largest K to include")
    ap.add_argument("--no-n", action="store_true", help="drop the shot counts on top")
    ap.add_argument(
        "--min-n", type=int, default=MIN_N, help="min shots for a per-run point"
    )
    ap.add_argument("--ext", default="svg", help="figure file format")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    plot(
        save=not a.no_save,
        kmax=a.kmax,
        images=a.images,
        annotate_n=not a.no_n,
        min_n=a.min_n,
        ext=a.ext,
    )
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
