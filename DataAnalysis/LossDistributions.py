"""
This script plots the heatmap of mean atoms lost (K x M), plus observed-vs-model loss-count
curves for the biggest cells. Also checks whether loss is independent per atom
or shot-to-shot common-mode (MOT drift, a flaky arm, a bad dark window).

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING
DEFAULT_RUNS = None  # None = every run in Data/tweezerImagesSorting1D


def fig_loss_distributions(loss):
    """heatmap of mean atoms lost, plus observed-vs-model loss-count curvesfor the biggest cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = loss["cells"]
    if not cells:
        return None
    ks = sorted({k for k, _ in cells})
    ms = sorted({m for _, m in cells})
    heat = np.full((len(ks), len(ms)), np.nan)
    cnt = np.zeros_like(heat)
    for (k, m), c in cells.items():
        i, j = ks.index(k), ms.index(m)
        heat[i, j] = float((c["obs"] * np.arange(len(c["obs"]))).sum() / max(1, c["n"]))
        cnt[i, j] = c["n"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    im = a1.imshow(heat, origin="lower", aspect="auto", cmap="magma")
    a1.set_xticks(range(len(ms)), ms), a1.set_yticks(range(len(ks)), ks)  # type: ignore
    a1.set_xlabel("M atoms that had to move"), a1.set_ylabel("K atoms loaded")  # type: ignore
    a1.set_title("Mean atoms lost")
    for i in range(len(ks)):
        for j in range(len(ms)):
            if cnt[i, j]:
                a1.text(
                    j,
                    i,
                    f"{heat[i, j]:.1f}\nn={int(cnt[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="w",
                )
    fig.colorbar(im, ax=a1, label="mean loss")
    big = sorted(cells.items(), key=lambda kv: -kv[1]["n"])[:6]
    for (k, m), c in big:
        x = np.arange(len(c["obs"]))
        p, lo, hi = zip(*(C.wilson(o, c["n"]) for o in c["obs"]))
        p, lo, hi = np.array(p), np.array(lo), np.array(hi)
        eb = a2.errorbar(
            x,
            p,
            yerr=[p - lo, hi - p],
            marker="o",
            ms=4,
            lw=1.4,
            capsize=3,
            label=f"K={k},M={m} (n={c['n']})",
        )
        a2.plot(
            x, c["exp"] / max(1, c["n"]), ls="--", lw=1, color=eb.lines[0].get_color()
        )
    a2.set_xlabel("atoms lost")
    a2.set_ylabel("probability")
    a2.set_title(
        f"observed (solid) vs independent model (dashed)\n"
        f"p_stat={loss['p_stat']:.3f}  p_move={loss['p_move']:.3f}"
    )
    a2.legend(fontsize=7, framealpha=0.95)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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


def plot(runs=None, save=True, images=None, name=None):
    """loss distributions vs the independent model, pooled over the selected runs. Returns the Figure."""
    run_dss = _load(runs if runs is not None else DEFAULT_RUNS, images)
    loss = C.loss_distributions(run_dss)
    fig = fig_loss_distributions(loss)
    if fig is not None and save:
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, CATEGORY, name, "loss_distributions")
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
        for ds in _load(a.runs, a.images):
            print(f"  {ds.name}  ({ds.n_shots} shots)")
        return 0

    fig = plot(runs=a.runs, save=not a.no_save, images=a.images)
    if fig is None:
        print("no loss cells -- nothing to plot")
        return 1
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
