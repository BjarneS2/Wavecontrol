"""
Quick diagnostic figure for one SingleAtomTransport run: photon-count histograms
for the loading frame (source, target) on top and the survival frame (source,
target) on the bottom, with the fitted bimodal threshold overlaid on all four.

@author: Bjarne Schümann
"""

import sys

import matplotlib

matplotlib.use("Agg")
import CommonThings as C
import matplotlib.pyplot as plt

RUN = "STA250us"
INI, FIN = 0, 1


def plot(run=RUN, save=True):
    """Builds the 2x2 loading/survival histogram figure for one run. Returns the Figure."""
    images = str(C.TRANSPORT_IMAGES)
    cfg = C.transport_config_of(f"tweezerLoad1x2-{run}")
    cal, masks, per_run = C.load_transport_config(images, cfg)
    name = next(k for k in per_run if k.endswith(f"-{run}"))
    load_counts, surv_counts = per_run[name]
    thr, fitp = C.bimodal_threshold(load_counts[INI])
    print(f"{name}  threshold={thr:.4f}  n={load_counts.shape[1]}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    panels = [
        (axes[0, 0], load_counts[INI], "source, loading frame"),
        (axes[0, 1], load_counts[FIN], "target, loading frame"),
        (axes[1, 0], surv_counts[INI], "source, survival frame"),
        (axes[1, 1], surv_counts[FIN], "target, survival frame"),
    ]
    for ax, counts, ttl in panels:
        ax.hist(counts, bins=40, color="tab:blue", alpha=0.8)
        ax.axvline(thr, color="k", ls="--", label=f"thr={thr:.1f}")
        ax.set_title(f"{name}\n{ttl}", fontsize=9)
        ax.set_ylabel("shots")
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("photon counts")
    axes[1, 1].set_xlabel("photon counts")
    fig.tight_layout()
    if save:
        C.save_figure(fig, C.CAT_TRANSPORT, name, "load_survival_histograms", dpi=300)
    return fig


if __name__ == "__main__":
    sys.exit(0 if plot() else 1)
