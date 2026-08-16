"""
This image is more for the purpose of calibration in the lab and for the future.
It is an optimized version of the code that was used at the time of the experiment,
allowing single site resolved setting of thresholds as well as an overall threshold.

This script will show the per site histograms, a visual image of the locations and
filled sites. Also, immediately show the loading fraction for each site given the
current thresholds in an interactive way. The user can change the thresholds and see
which is optimal (by eye) for the current calibration.

The old code saved into sorter_calibration.npz the save writes a new file never
overwriting the old one. The user can then use the new thresholds saved in
per_sites_thresholds.npz next to it. (Both files can then be used, but for using the
new thresholds, the user has to change the code for loading in the parameters and
making it handle individual thresholds for each site - see ../Sorter/sorter.py).

@author: Bjarne Schümann
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # else for pdf use "PDF"
import CommonThings as A  # contains the analysis functions for the sorter calibration and thresholding
import matplotlib.pyplot as plt


def draw_bridge_depth_bars(ax, counts_frame):
    """Per-site bridge fraction and valley depth -- lower is better on both."""
    ax.clear()
    n = counts_frame.shape[-1]
    stats = [A.count_stats(counts_frame[:, i]) for i in range(n)]
    br = [s["bridge"] if np.isfinite(s["bridge"]) else 0 for s in stats]
    dp = [s["valley_depth"] if np.isfinite(s["valley_depth"]) else 1 for s in stats]
    ax.bar(np.arange(n) - 0.2, br, 0.4, label="bridge frac", color="tab:red")
    ax.bar(np.arange(n) + 0.2, dp, 0.4, label="valley depth", color="tab:brown")
    ax.axhline(0.05, color="tab:red", ls=":", lw=0.8)
    ax.axhline(0.15, color="tab:brown", ls=":", lw=0.8)
    ax.set_title("bridges per site (lower is better)", fontsize=9)
    ax.set_xlabel("site")
    ax.legend(fontsize=7, framealpha=0.95)


def draw_loading_survival_bars(ax, counts_load, counts_surv, thr_global, thr_ps):
    """Loading and P(survive | loaded) per site, global vs per-site thresholds."""
    ax.clear()
    n = counts_load.shape[-1]
    g = A.occupancy(counts_load, np.full(n, thr_global))
    p_ = A.occupancy(counts_load, thr_ps)
    sg = A.occupancy(counts_surv, np.full(n, thr_global))
    sp = A.occupancy(counts_surv, thr_ps)
    w = 0.2
    x = np.arange(n)
    ax.bar(x - 1.5 * w, g.mean(0), w, color="0.35", label="loading, global")
    ax.bar(x - 0.5 * w, p_.mean(0), w, color="0.7", label="loading, per-site")
    sv_g = [(g[:, j] & sg[:, j]).sum() / max(1, g[:, j].sum()) for j in range(n)]
    sv_p = [(p_[:, j] & sp[:, j]).sum() / max(1, p_[:, j].sum()) for j in range(n)]
    ax.bar(x + 0.5 * w, sv_g, w, color="tab:blue", label="survival, global")
    ax.bar(x + 1.5 * w, sv_p, w, color="tab:cyan", label="survival, per-site")
    flips = int((g != p_).sum())
    ax.set_title(
        f"loading and P(survive|loaded) per site   |   "
        f"{flips}/{g.size} site-shots reclassified "
        f"({100 * flips / g.size:.2f}%)   |   mean loading "
        f"{g.mean():.3f} -> {p_.mean():.3f}",
        fontsize=9,
    )
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=4, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)


def draw_threshold_scan(ax, counts_frame, thr_global):
    """Mean fill vs a swept global threshold -- flat means the choice is safe."""
    ax.clear()
    n = counts_frame.shape[-1]
    lo, hi = np.percentile(counts_frame, [1, 99])
    ts = np.linspace(lo, hi, 60)
    kk = [A.occupancy(counts_frame, np.full(n, t)).mean() for t in ts]
    ax.plot(ts, kk, "k-")
    ax.axvline(thr_global, color="k", ls="--")
    ax.set_xlabel("global threshold")
    ax.set_ylabel("mean fill")
    ax.set_title("plateau check: flat = safe", fontsize=9)
    ax.grid(alpha=0.3)


def draw_site_overlay(ax, mean_frame, locations, highlight_site, title):
    """The mean image with every site circled, the currently-selected one in cyan."""

    ax.clear()
    ax.imshow(mean_frame, cmap="inferno")
    for j, (x, y) in enumerate(locations):  # sorter stores (x, y)
        ax.add_patch(
            plt.Circle(
                (x, y),
                5,
                fill=False,
                color="lime" if j != highlight_site else "cyan",
                lw=1.2,
            )  # type: ignore
        )
        ax.text(x, y - 8, str(j), color="w", fontsize=7, ha="center")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def draw_pooled_histogram(ax, counts_frame, thr_global):
    """All sites pooled: global threshold vs the auto valley/empty/filled peaks."""
    ax.clear()
    st = A.count_stats(counts_frame)
    ax.hist(counts_frame.ravel(), bins=90, color="0.6")
    ax.axvline(thr_global, color="k", ls="--", lw=1.5, label=f"global {thr_global:.1f}")
    if np.isfinite(st["thr"]):
        ax.axvline(st["thr"], color="tab:green", lw=1.2, label=f"auto {st['thr']:.1f}")
        for v, c in ((st["empty"], "tab:blue"), (st["filled"], "tab:orange")):
            ax.axvline(v, color=c, lw=0.8, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(
        f"pooled, all sites  (depth {st['valley_depth']:.2f}, bridge {st['bridge']:.3f})",
        fontsize=9,
    )
    ax.set_xlabel("counts in ROI")
    ax.legend(fontsize=7, framealpha=0.95)


def draw_site_histogram(ax, counts_frame, site, thr_global, thr_site):
    """One site's histogram: global threshold, this site's threshold, its auto valley."""
    ax.clear()
    col = counts_frame[:, site]
    st = A.count_stats(col)
    ax.hist(col, bins=45, color="tab:purple", alpha=0.75)
    ax.axvline(thr_global, color="k", ls="--", lw=1.3)
    ax.axvline(thr_site, color="tab:red", lw=1.6)
    if np.isfinite(st["thr"]):
        ax.axvline(st["thr"], color="tab:green", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_title(
        f"site {site}: own thr {thr_site:.1f}, auto {st['thr']:.1f}, bridge {st['bridge']:.3f}"
        + ("" if st["bimodal"] else "  UNIMODAL"),
        fontsize=9,
    )
    ax.set_xlabel("counts in ROI")


def fig_per_site_histograms(counts, thr_ps, thr_global, frame, path=None):
    """Item 6: every site's histogram, own valley vs the global threshold."""
    import matplotlib

    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = counts.shape[-1]
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow), sharex=True, squeeze=False
    )
    allc = counts[:, frame, :].ravel()
    bins = np.linspace(np.percentile(allc, 0.2), np.percentile(allc, 99.8), 45)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis("off")
            continue
        st = A.count_stats(counts[:, frame, i])
        ax.hist(counts[:, frame, i], bins=bins, color="0.65", edgecolor="none")
        ax.axvline(thr_global, color="k", ls="--", lw=1.2)
        ax.axvline(thr_ps[i], color="tab:red", ls="-", lw=1.2)
        ax.set_yscale("log")
        ax.set_title(
            f"site {i}  bridge {st['bridge']:.3f}"
            if st["bimodal"]
            else f"site {i}  UNIMODAL",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"frame {frame}: per-site count histograms   "
        f"black dashed = global {thr_global:.1f}, red = per-site valley",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path:
        fig.savefig(path, dpi=130)
        print("wrote " + path)
    return fig


def peaks_and_valley(x, bins=None):
    """Thin adapter onto sortanalysis.count_stats, keeping the old key names.

    depth  = h(threshold) / min(h(empty), h(filled)). 0 is clean separation; towards 1
             the two populations have merged.
    bridge = fraction of samples inside the ambiguous band (the middle 30% of the gap
             between the two class medians). This is the number that says a site is
             untunable rather than mistuned.
    """
    st = A.count_stats(x, bins)
    return dict(
        empty=st["empty"],
        filled=st["filled"],
        valley=st["thr"],
        depth=st["valley_depth"],
        bridge=st["bridge"],
        bimodal=st["bimodal"],
    )


def site_stats(counts, thr_global, frame):
    """counts (S,F,n) -> list of per-site dicts for the chosen frame."""
    out = []
    for i in range(counts.shape[-1]):
        st = peaks_and_valley(counts[:, frame, i])
        st["site"] = i
        st["auto_ok"] = bool(st["bimodal"]) and np.isfinite(st["valley"])
        st["thr"] = st["valley"] if st["auto_ok"] else float(thr_global)
        out.append(st)
    return out


def report(cal, counts, thr_ps, frame, load_frame, surv_frame, run_dss=None):
    n = counts.shape[-1]
    stats = site_stats(counts, cal.threshold, frame)
    g = A.occupancy(counts[:, load_frame, :], np.full(n, cal.threshold))
    p_ = A.occupancy(counts[:, load_frame, :], thr_ps)
    s_g = A.occupancy(counts[:, surv_frame, :], np.full(n, cal.threshold))
    s_p = A.occupancy(counts[:, surv_frame, :], thr_ps)

    print(f"\nframe {frame} counts, global threshold {cal.threshold:.1f}")
    print(
        f"{'site':>4} {'thr':>7} {'empty':>8} {'filled':>8} {'depth':>6} {'bridge':>7} "
        f"{'load_g':>7} {'load_ps':>7} {'surv_g':>7} {'surv_ps':>7}"
    )
    for st in stats:
        i = st["site"]
        lg, lp = g[:, i].mean(), p_[:, i].mean()
        sg = (g[:, i] & s_g[:, i]).sum() / max(1, g[:, i].sum())
        sp = (p_[:, i] & s_p[:, i]).sum() / max(1, p_[:, i].sum())
        flag = "" if st["auto_ok"] else "  <- unimodal, used global"
        print(
            f"{i:4d} {thr_ps[i]:7.1f} {st['empty']:8.1f} {st['filled']:8.1f} "
            f"{st['depth']:6.2f} {st['bridge']:7.3f} {lg:7.3f} {lp:7.3f} "
            f"{sg:7.3f} {sp:7.3f}{flag}"
        )
    bad = sorted(
        stats, key=lambda s: -(s["bridge"] if np.isfinite(s["bridge"]) else 0)
    )[:3]
    print(
        "\nworst bridges: "
        + ", ".join(
            f"site {s['site']} (bridge {s['bridge']:.3f}, depth {s['depth']:.2f})"
            for s in bad
        )
    )
    print(
        "  depth > ~0.15 or bridge > ~0.05 means the two populations are not cleanly\n"
        "  separated at that site -- no threshold fixes it, look at trap depth or\n"
        "  imaging light instead."
    )

    flips = int((g != p_).sum())
    print(
        f"\nper-site vs global on this calibration set: {flips}/{g.size} site-shots flip "
        f"({100 * flips / g.size:.3f}%)"
    )
    print(f"  mean loading  global {g.mean():.4f}   per-site {p_.mean():.4f}")
    sg = (g & s_g).sum() / max(1, g.sum())
    sp = (p_ & s_p).sum() / max(1, p_.sum())
    print(f"  mean survival global {sg:.4f}   per-site {sp:.4f}")

    if run_dss:
        changed = k_ch = t_ch = 0
        tot = 0
        for ds in run_dss:
            gg = A.occupancy(ds.counts[:, load_frame, :], np.full(n, cal.threshold))
            pp = A.occupancy(ds.counts[:, load_frame, :], thr_ps)
            for s in range(len(gg)):
                tot += 1
                if not (gg[s] != pp[s]).any():
                    continue
                a_, b_ = A.plan(gg[s]), A.plan(pp[s])
                if (
                    a_.k != b_.k
                    or a_.t0 != b_.t0
                    or not np.array_equal(a_.targets, b_.targets)
                ):
                    changed += 1
                    k_ch += a_.k != b_.k
                    t_ch += a_.t0 != b_.t0
        print(
            f"  on the runs: the PLAN differs in {changed}/{tot} shots "
            f"(K changed {k_ch}, target block moved {t_ch})"
        )
        print(
            "  ^ this is the number for the thesis. Reclassified site-shots is the raw\n"
            "    difference; a changed plan is a difference that had consequences."
        )
    return stats


def interactive(
    folder, cal, mean_frames, n_shots, counts, thr_ps, frame, load_frame, surv_frame
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    n = counts.shape[-1]
    state = {
        "frame": frame,
        "site": 0,
        "thr_g": float(cal.threshold),
        "thr_ps": np.array(thr_ps, float),
    }

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.25, 1.0],
        left=0.06,
        right=0.985,
        top=0.94,
        bottom=0.16,
        hspace=0.42,
        wspace=0.24,
    )
    ax_img = fig.add_subplot(gs[0, :])
    ax_pool = fig.add_subplot(gs[1, 0])
    ax_site = fig.add_subplot(gs[1, 1])
    ax_bridge = fig.add_subplot(gs[1, 2])
    ax_load = fig.add_subplot(gs[2, :2])
    ax_scan = fig.add_subplot(gs[2, 2])

    def draw():
        f, i = state["frame"], state["site"]
        tg, tps = state["thr_g"], state["thr_ps"]
        cf = counts[:, f, :]
        draw_site_overlay(
            ax_img,
            mean_frames[f],
            cal.locations,
            i,
            f"{os.path.basename(folder)}  mean of frame {f}  ({n_shots} shots, {n} sites)",
        )
        draw_pooled_histogram(ax_pool, cf, tg)
        draw_site_histogram(ax_site, cf, i, tg, tps[i])
        draw_bridge_depth_bars(ax_bridge, cf)
        draw_loading_survival_bars(
            ax_load, counts[:, load_frame, :], counts[:, surv_frame, :], tg, tps
        )
        draw_threshold_scan(ax_scan, cf, tg)
        fig.canvas.draw_idle()

    axg = fig.add_axes([0.08, 0.085, 0.40, 0.022])
    axs = fig.add_axes([0.08, 0.050, 0.40, 0.022])
    axp = fig.add_axes([0.08, 0.015, 0.40, 0.022])
    cmax = float(np.percentile(counts, 99.5))
    s_g = Slider(axg, "global thr", 0, cmax, valinit=state["thr_g"])
    s_i = Slider(axs, "site", 0, n - 1, valinit=0, valstep=1)
    s_p = Slider(axp, "this site thr", 0, cmax, valinit=float(thr_ps[0]))

    def on_g(v):
        state["thr_g"] = float(v)
        draw()

    def on_i(v):
        state["site"] = int(v)
        s_p.eventson = False
        s_p.set_val(float(state["thr_ps"][int(v)]))
        s_p.eventson = True
        draw()

    def on_p(v):
        state["thr_ps"][state["site"]] = float(v)
        draw()

    s_g.on_changed(on_g)
    s_i.on_changed(on_i)
    s_p.on_changed(on_p)

    axr = fig.add_axes([0.55, 0.015, 0.09, 0.10])
    radio = RadioButtons(axr, ("frame 0", "frame 1", "frame 2"), active=state["frame"])

    def on_frame(lbl):
        state["frame"] = int(lbl[-1])
        draw()

    radio.on_clicked(on_frame)

    def mk(x, label, fn):
        b = Button(fig.add_axes([x, 0.05, 0.10, 0.045]), label)
        b.on_clicked(fn)
        return b

    def auto_all(_):
        st = site_stats(counts, state["thr_g"], state["frame"])
        state["thr_ps"] = np.array([s["thr"] for s in st])
        s_p.eventson = False
        s_p.set_val(float(state["thr_ps"][state["site"]]))
        s_p.eventson = True
        draw()

    def flatten(_):
        state["thr_ps"][:] = state["thr_g"]
        s_p.eventson = False
        s_p.set_val(state["thr_g"])
        s_p.eventson = True
        draw()

    def save(_):
        p = os.path.join(folder, "per_site_thresholds.npz")
        np.savez(
            p,
            locations=cal.locations,
            threshold=state["thr_g"],
            per_site_threshold=state["thr_ps"],
            n=n,
            frame=state.get("load_frame", load_frame),
        )
        json.dump(
            {
                "atom_threshold": state["thr_g"],
                "per_site_threshold": state["thr_ps"].tolist(),
                "load_frame": int(load_frame),
                "n_expected": int(n),
            },
            open(os.path.join(folder, "per_site_thresholds.json"), "w"),
            indent=2,
        )
        print("wrote " + p + "  (the live sorter_calibration.npz was NOT touched)")

    b1 = mk(0.67, "auto all", auto_all)
    b2 = mk(0.78, "flatten", flatten)
    b3 = mk(0.89, "SAVE", save)
    fig._widgets = (s_g, s_i, s_p, radio, b1, b2, b3)  # type: ignore
    draw()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder", help="calibration folder with the npy shots + npz")
    ap.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="substrings selecting run prefixes, for the plan-change count",
    )
    ap.add_argument(
        "--images",
        default=None,
        help="folder holding the run files (default: the parent of <folder>)",
    )
    ap.add_argument("--min-shots", type=int, default=20)
    ap.add_argument(
        "--frame", type=int, default=1, help="frame shown in the histograms"
    )
    ap.add_argument("--load-frame", type=int, default=1)
    ap.add_argument("--surv-frame", type=int, default=2)
    ap.add_argument(
        "--threshold", type=float, default=None, help="override the global threshold"
    )
    ap.add_argument("--report", action="store_true", help="headless table, then exit")
    ap.add_argument(
        "--grid",
        nargs="?",
        const="per_site_histograms.png",
        help="write the per-site histogram grid and exit",
    )
    ap.add_argument(
        "--save", action="store_true", help="write per_site_thresholds.npz and exit"
    )
    a = ap.parse_args(argv)

    cal = A.load_calibration(a.folder, a.threshold)
    entries = sum(A.discover_runs(a.folder, 1, verbose=False).values(), [])
    cal_ds = A.load_dataset("calibration", entries, cal)
    counts = cal_ds.counts
    thr_ps = np.array(
        [st["thr"] for st in site_stats(counts, cal.threshold, a.load_frame)]
    )

    run_dss = []
    if a.runs:
        groups = A.discover_runs(
            a.images or os.path.dirname(os.path.normpath(a.folder)), a.min_shots
        )
        for k in sorted(groups):
            if any(r in k for r in a.runs):
                run_dss.append(A.load_dataset(k, groups[k], cal))

    if a.report or a.save or a.grid:
        report(cal, counts, thr_ps, a.frame, a.load_frame, a.surv_frame, run_dss)
        if a.grid:
            fig_per_site_histograms(
                counts,
                thr_ps,
                cal.threshold,
                a.frame,
                a.grid if os.path.isabs(a.grid) else os.path.join(a.folder, a.grid),
            )
        if a.save:
            p = os.path.join(a.folder, "per_site_thresholds.npz")
            np.savez(
                p,
                locations=cal.locations,
                threshold=cal.threshold,
                per_site_threshold=thr_ps,
                n=cal.n,
                frame=a.load_frame,
            )
            json.dump(
                {
                    "atom_threshold": float(cal.threshold),
                    "per_site_threshold": thr_ps.tolist(),
                    "load_frame": int(a.load_frame),
                    "n_expected": int(cal.n),
                },
                open(os.path.join(a.folder, "per_site_thresholds.json"), "w"),
                indent=2,
            )
            print("wrote " + p + "  (sorter_calibration.npz NOT touched)")
        return 0

    import matplotlib.pyplot as plt

    interactive(
        a.folder,
        cal,
        cal_ds.mean_frames,
        cal_ds.n_shots,
        counts,
        thr_ps,
        a.frame,
        a.load_frame,
        a.surv_frame,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
