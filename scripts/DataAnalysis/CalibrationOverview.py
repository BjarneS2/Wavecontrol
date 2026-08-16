"""
Overview figure for a calibration run: the mean image with every site circled on the
left, and the pooled (all sites combined) loading-frame histogram on the right.

@author: Bjarne Schümann
"""

import argparse
import sys

import CommonThings as C
import numpy as np

CATEGORY = C.CAT_SORTING


def fig_calibration_overview(cal_ds, frame=None):
    """Left: mean frame with every site circled. Right: pooled loading histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = C.LOAD_FRAME if frame is None else frame
    cal = cal_ds.cal
    counts_frame = cal_ds.counts[:, frame, :]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    a1.imshow(cal_ds.mean_frames[frame], cmap="inferno")
    for j, (x, y) in enumerate(cal.locations):  # sorter stores (x, y)
        a1.add_patch(plt.Circle((x, y), 5, fill=False, color="lime", lw=1.2))
        a1.text(x + 10, y + 2, str(j), color="w", fontsize=8, ha="center")
    # a1.set_title(
    #     f"{cal_ds.name}  mean of frame {frame}  ({cal_ds.n_shots} shots, {cal.n} sites)",
    #     fontsize=10,
    # )
    a1.axis("off")
    a1.text(
        0.02,
        0.98,
        "(a)",
        transform=a1.transAxes,
        color="w",
        fontsize=12,
        ha="left",
        va="top",
        weight="bold",
    )

    st = C.count_stats(counts_frame)
    a2.hist(counts_frame.ravel(), bins=90, color="tab:blue")
    a2.text(
        0.02,
        0.98,
        "(b)",
        transform=a2.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        weight="bold",
    )
    a2.axvline(
        cal.threshold, color="k", ls="--", lw=1.5, label=f"global {cal.threshold:.1f}"
    )
    if np.isfinite(st["thr"]):
        a2.axvline(st["thr"], color="tab:red", lw=1.2, label=f"auto {st['thr']:.1f}")
        # for v, c in ((st["empty"], "tab:blue"), (st["filled"], "tab:orange")):
        #     a2.axvline(v, color=c, lw=0.8, alpha=0.7)
    a2.set_yscale("log")
    a2.set_xlabel("counts in ROI")
    # a2.set_title(
    #     f"pooled, all sites  (depth {st['valley_depth']:.2f}, bridge {st['bridge']:.3f})",
    #     fontsize=9,
    # )
    a2.legend(fontsize=7, framealpha=0.95)

    fig.tight_layout()
    return fig


def _load(images=None, threshold=None):
    images = images or str(C.SORTING1D_CAL)
    cal = C.load_calibration(images, threshold)
    entries = sum(C.discover_runs(images, 1, verbose=False).values(), [])
    return C.load_dataset(
        "calibration", entries, cal, is_calibration=True, verbose=False
    )


def plot(images=None, threshold=None, frame=None, save=True, name=None):
    cal_ds = _load(images, threshold)
    fig = fig_calibration_overview(cal_ds, frame)
    if save:
        C.save_figure(fig, CATEGORY, name or "calibration", "calibration_overview")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=None, help="calibration folder")
    ap.add_argument(
        "--threshold", type=float, default=None, help="override global threshold"
    )
    ap.add_argument(
        "--frame", type=int, default=None, help="default: the loading frame"
    )
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    fig = plot(a.images, a.threshold, a.frame, save=not a.no_save)
    if a.no_save:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
