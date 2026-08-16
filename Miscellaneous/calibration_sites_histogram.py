"""
Left: the calibration's mean image with every one of the 11 sites circled and
numbered. Right: the pooled (all sites, all shots) loading-frame histogram
with the live threshold marked.

Two versions of the same figure are produced, differing only in the image
colormap:
  - "viridis", matching every other real-image plot under Miscellaneous/
    (sorting_steps_illustration.py, sorting_failure_illustration.py, ...)
  - "inferno", matching how the calibration analysis itself renders the mean
    image (Own Data Analysis/CalibrationOverview.py, TunerForCalibration.py)

@author: Bjarne Schuemann
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Own Data Analysis",
    ),
)
import CommonThings as C  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.linewidth"] = 1.0
matplotlib.rcParams["font.size"] = 14

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_calibration_dataset(images=None):
    images = images or str(C.SORTING1D_CAL)
    cal = C.load_calibration(images, verbose=False)
    entries = sum(C.discover_runs(images, 1, verbose=False).values(), [])
    return C.load_dataset(
        "calibration", entries, cal, is_calibration=True, verbose=False
    )


def fig_calibration_sites_histogram(cal_ds, cmap, frame=None):
    """Left: mean frame with all sites circled + numbered. Right: pooled loading
    histogram with the live threshold."""
    frame = C.LOAD_FRAME if frame is None else frame
    cal = cal_ds.cal
    counts_frame = cal_ds.counts[:, frame, :]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    a1.imshow(cal_ds.mean_frames[frame], cmap=cmap)
    for j, (x, y) in enumerate(cal.locations):  # sorter stores (x, y)
        a1.add_patch(plt.Circle((x, y), 5, fill=False, color="lime", lw=1.2))
        a1.text(x + 10, y + 2, str(j), color="w", fontsize=10, ha="center")
    a1.axis("off")
    a1.text(
        0.02,
        0.98,
        "(a)",
        transform=a1.transAxes,
        color="w",
        ha="left",
        va="top",
        weight="bold",
    )

    a2.hist(counts_frame.ravel(), bins=90, color="tab:blue")
    a2.text(
        0.02,
        0.98,
        "(b)",
        transform=a2.transAxes,
        ha="left",
        va="top",
        weight="bold",
    )
    a2.axvline(
        cal.threshold,
        color="k",
        ls="--",
        lw=1.5,
        label=f"threshold {cal.threshold:.1f}",
    )
    a2.set_yscale("log")
    a2.set_xlabel("counts in ROI")
    a2.legend(framealpha=0.95)

    fig.tight_layout()
    return fig


def main():
    cal_ds = load_calibration_dataset()
    for cmap in ("viridis", "inferno"):
        fig = fig_calibration_sites_histogram(cal_ds, cmap)
        fig.savefig(
            os.path.join(OUT_DIR, f"calibration_sites_histogram_{cmap}.pdf"),
            facecolor="white",
        )
        fig.savefig(
            os.path.join(OUT_DIR, f"calibration_sites_histogram_{cmap}.png"),
            dpi=300,
            facecolor="white",
        )
        plt.close(fig)
        print(f"saved calibration_sites_histogram_{cmap}.png / .pdf")


if __name__ == "__main__":
    main()