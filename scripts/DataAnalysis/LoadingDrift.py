"""
This script simply loads all data and just looks at the loading and how / if
it drifts over the time and course of the experiment. It is only meant as a
sanity check since all data has a time stamp and one could figure out what has
an influence on the experiment and the performance of the apparatus.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonAnalysis as A
import CommonThings as C
import numpy as np

# from plots import fig_loading_drift

DEFAULT_RUNS = None  # None = every run in the chosen dataset


def fig_loading_drift(cal_ds, run_dss, per_site=False, n_sites=11, window=25):
    """Atoms loaded per shot against wall-clock time: faint points, prominent rolling mean.

    Plotted on the real time axis parsed from the filenames, so gaps between runs and the
    stretch where the MOT/laser lock was lost are where they actually were.
    """
    import matplotlib

    matplotlib.use("Agg")
    from datetime import datetime

    import matplotlib.pyplot as plt

    def secs(stamps):
        out = []
        for t in stamps:
            try:
                out.append(datetime.strptime(t, "%Y%m%d-%H%M%S"))  # noqa: DTZ007
            except Exception:  # noqa
                out.append(None)
        return out

    fig, ax = plt.subplots(figsize=(13, 4.6))
    all_ds = ([cal_ds] if cal_ds else []) + list(run_dss)
    n_runs = len(run_dss)
    palette = (
        list(plt.get_cmap("tab10").colors)
        + list(plt.get_cmap("Dark2").colors)
        + list(plt.get_cmap("Set1").colors)
    )
    colors = [palette[i % len(palette)] for i in range(max(n_runs, 1))]
    run_i = 0
    boundary_ts = []
    for i, ds in enumerate(all_ds):
        k = ds.occ(A.LOAD_FRAME, per_site).sum(1).astype(float)
        t = secs(ds.stamps)
        has_time = not any(x is None for x in t)
        if not has_time:
            t = np.arange(len(k))
        if ds.is_calibration:
            col = "0.35"
        else:
            col = colors[run_i % len(colors)]
            run_i += 1

        if has_time:
            gaps = np.array([(t[j + 1] - t[j]).total_seconds() for j in range(len(t) - 1)])
            med = np.median(gaps) if len(gaps) else 0
            breaks = np.where(gaps > max(60.0, 8 * med))[0] + 1
            if i > 0:
                boundary_ts.append(t[0])
            boundary_ts.extend(t[b] for b in breaks)
        else:
            breaks = np.array([], dtype=int)

        label = ds.name.replace("tweezerLoad1x11-", "")
        bounds = [0, *breaks.tolist(), len(k)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            seg_t, seg_k = t[s:e], k[s:e]
            ax.plot(seg_t, seg_k, ".", ms=3, alpha=0.25, color=col)
            w = max(1, min(window, max(3, len(seg_k) // 4), len(seg_k)))
            ker = np.ones(w) / w
            sm = np.convolve(seg_k, ker, "same") / np.convolve(
                np.ones_like(seg_k), ker, "same"
            )
            ax.plot(seg_t, sm, "-", lw=4.2, color=col, alpha=1.0, label=label)
            label = None
    for bt in boundary_ts:
        ax.axvline(bt, color="0.4", lw=1.8, ls=":", alpha=0.7, zorder=0)
    ax.set_ylabel(f"atoms loaded (of {n_sites})")
    ax.set_xlabel("time of day")
    ax.set_ylim(0, n_sites)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("loading rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=5, loc="lower left", framealpha=0.95)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _load_sorting(runs=None, images=None, min_shots=20):
    """--runs substrings -> (cal_ds, run_dss, n_sites) for the Sorting1D (11-site) dataset."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    cal_entries = sum(  # noqa: RUF017
        C.discover_runs(str(C.SORTING1D_CAL), 1, verbose=False).values(), []
    )
    cal_ds = C.load_dataset(
        "calibration", cal_entries, cal, is_calibration=True, verbose=False
    )

    groups = C.discover_runs(images, min_shots, verbose=False)
    names = (
        sorted(groups)
        if not runs
        else [k for k in sorted(groups) if any(r in k for r in runs)]
    )
    run_dss = [C.load_dataset(n, groups[n], cal, verbose=False) for n in names]
    run_dss = C.merge_run_datasets(run_dss)
    return cal_ds, run_dss, cal.n


def _transport_dataset(name, folder, prefix, cal, masks):
    """One transport run -> a Dataset, counted through the Voronoi site masks."""
    entries = C.discover_runs(folder, 1, verbose=False)[prefix]
    raw = C.load_stack(folder, prefix)
    im = C.to_photons(raw, cal.binning)
    n_shots, n_frames = im.shape[0], im.shape[1]
    counts = np.empty((n_shots, n_frames, len(masks)))
    for f in range(n_frames):
        counts[:, f, :] = C.roi_mask_counts(im[:, f], masks).T
    return C.Dataset(
        name,
        [p for _, _, p in entries],
        np.array([e[0] for e in entries]),
        [e[1] for e in entries],
        counts,
        im.mean(0),
        cal,
    )


def _load_transport(runs=None, images=None, min_shots=1):
    images = images or str(C.TRANSPORT_IMAGES)
    run_dss = []
    n_sites = None
    for cfg in C.TRANSPORT_CONFIGS:
        cal, masks, cfg_runs = C.load_transport_config(images, cfg, min_shots)
        n_sites = cal.n
        names = (
            sorted(cfg_runs)
            if not runs
            else [k for k in sorted(cfg_runs) if any(r in k for r in runs)]
        )
        if not names:
            continue
        dss = [_transport_dataset(n, images, n, cal, masks) for n in names]
        pool = np.concatenate([ds.counts[:, C.LOAD_FRAME, 0] for ds in dss])
        cal.threshold, _ = C.bimodal_threshold(pool)
        run_dss.extend(dss)
    return None, run_dss, n_sites


def plot(dataset="sorting", runs=None, save=True, images=None, name=None):
    """Loading fraction vs wall-clock time for one dataset. Returns the Figure."""
    loader = _load_sorting if dataset == "sorting" else _load_transport
    cal_ds, run_dss, n_sites = loader(
        runs if runs is not None else DEFAULT_RUNS, images
    )
    fig = fig_loading_drift(cal_ds, run_dss, n_sites=n_sites)  # type: ignore
    if save:
        category = C.CAT_SORTING if dataset == "sorting" else C.CAT_TRANSPORT
        name = name or (C.POOLED if not runs else "_".join(runs))
        C.save_figure(fig, category, name, "loading_drift")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["sorting", "transport"], default="sorting")
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument("--images", default=None, help="override the images folder")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    plot(dataset=a.dataset, runs=a.runs, save=not a.no_save, images=a.images)
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
