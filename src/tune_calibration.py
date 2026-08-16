"""
tune_calibration.py
Interactive tuning of the two calibration thresholds, then write a new sorter_calibration.npz.

TOUCHES NOTHING. It imports sorter.py and calls its functions; it does not modify sorter.py,
the GUI, or the Controller. No camera and no AWG are needed -- everything runs from the .npy
shots already on disk, so you can retune from your desk as often as you like.

THE TWO THRESHOLDS, which are unrelated:

  DETECT_FRAC   "where are the traps".  Applied ONCE to the MEAN of all shots. locate_sites()
                smooths, keeps pixels above frac * (brightest pixel in the frame), labels
                blobs of >= MIN_AREA px and takes each centroid. Because it is RELATIVE to
                the brightest pixel, a trap dimmer than that by more than the fraction simply
                vanishes -- which is how an 11-tone array reports 10 sites.

  ATOM_THRESHOLD  "is THIS site occupied in THIS shot".  site_counts() sums photons in a
                ROI_SIZE box at each site, masked by the union of the per-site Gaussian fits.
                auto_threshold() fits a double Gaussian to the pooled histogram of every
                (shot, site) count and sits at the valley between the empty and filled peaks.

Site detection and occupancy classification may use DIFFERENT frames: the brightest frame is
usually the most robust for finding the traps, while occupancy has to be read from the frame
your sequence designates as the loading image. Hence --detect-frame and --load-frame.

USAGE
    python tune_calibration.py                          # folder picker, interactive
    python tune_calibration.py <folder>                 # interactive on that folder
    python tune_calibration.py <folder> --scan          # headless: print the frac sweep table
    python tune_calibration.py <folder> --frac 0.30 --save   # headless: write the npz

Interactive controls: two sliders (DETECT_FRAC, ATOM_THRESHOLD), a frame radio button for
detection, and a Save button that writes <folder>/sorter_calibration.npz and records the
chosen atom_threshold in <folder>/sorter_config.json -- so run_sort_live.py picks it up with
no code change at all.

@author: tooling for Bjarne Schuemann's 1D sorter
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

import sorter as S  # noqa: E402


# ---------------------------------------------------------------------------- helpers
def _quiet(fn, *a, **kw):
    """Call fn with its prints swallowed (locate_sites is chatty; we redraw instead)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def load_all_frames(glob_pat, n_frames=3, binning=None):
    """(n_frames, M, H, W) photon-converted stack. One pass over the files per frame."""
    binning = S.BINNING if binning is None else binning
    return [S.load_frames(glob_pat, frame=f, binning=binning) for f in range(n_frames)]


def detect(mean_img, frac, n_expected, first_site=None, border_px=None, min_area=None):
    """locate_sites at a given frac. Returns (pts, message)."""
    kw = dict(n_expected=n_expected, frac=frac)
    if first_site is not None:
        kw["first_site"] = first_site
    if border_px is not None:
        kw["border_px"] = border_px
    if min_area is not None:
        kw["min_area"] = min_area
    try:
        pts = _quiet(S.locate_sites, mean_img, **kw)
    except Exception as e:  # noqa: BLE001
        return None, "detection failed: %s" % e
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) if len(pts) > 2 else np.array([])
    msg = "%d sites" % len(pts)
    if n_expected is not None:
        msg += " (expected %d)%s" % (n_expected, "" if len(pts) == n_expected else "  <-- MISMATCH")
    if d.size:
        msg += "\nspacing [px]: %s\nmedian %.2f, spread %.2f" % (
            np.array2string(np.round(d, 1), max_line_width=80), np.median(d), d.max() - d.min())
    return pts, msg


def counts_for(pts, stacks, load_frame, surv_frame):
    """Per-shot per-site counts on the loading and survival frames, from one shared ROI mask.

    The mask is fitted on the MEAN of the loading frame, then applied unchanged to both, so
    loading and survival are measured through exactly the same aperture.
    """
    mean_load = stacks[load_frame].mean(0)
    active, popts = _quiet(S.fit_site_masks, mean_load, pts)
    c_load = np.array([S.site_counts(im, pts, active) for im in stacks[load_frame]])
    c_surv = (np.array([S.site_counts(im, pts, active) for im in stacks[surv_frame]])
              if surv_frame is not None and surv_frame < len(stacks) else None)
    return active, popts, c_load, c_surv


def rates(c_load, c_surv, thr):
    """Per-site loading fraction and CONDITIONAL survival, with binomial errors.

    Survival is P(occupied in survival | occupied in loading), computed shot by shot -- not
    the ratio of the two mean fill fractions. The conditional form is what you actually want:
    it cannot be poisoned by a site that never loads, and its error bar is honest.
    """
    occ_l = c_load > thr
    load = occ_l.mean(0)
    err_load = np.sqrt(load * (1 - load) / len(c_load))
    if c_surv is None:
        return load, err_load, None, None
    occ_s = c_surv > thr
    n_l = occ_l.sum(0)
    surv = np.where(n_l > 0, (occ_l & occ_s).sum(0) / np.maximum(n_l, 1), np.nan)
    err_surv = np.where(n_l > 0, np.sqrt(surv * (1 - surv) / np.maximum(n_l, 1)), np.nan)
    return load, err_load, surv, err_surv


# ---------------------------------------------------------------------------- headless
def scan(mean_imgs, n_expected, fracs=(0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)):
    print("\nDETECT_FRAC sweep -- number of sites found on the mean image")
    print("      frac:  " + "".join("%6.2f" % f for f in fracs))
    for j, m in enumerate(mean_imgs):
        row = []
        for f in fracs:
            pts, _ = detect(m, f, n_expected)
            row.append("   err" if pts is None else "%6d" % len(pts))
        print("  frame %d:  %s   (peak %.1f ph)" % (j, "".join(row), m.max()))
    print("\nPick a frac in the middle of the widest plateau that equals your tone count.")


def report(pts, c_load, c_surv, thr, thr_src, load_frame, surv_frame):
    load, e_load, surv, e_surv = rates(c_load, c_surv, thr)
    print("\n  atom threshold %.1f (%s)" % (thr, thr_src))
    print("  loading frame %d, survival frame %s" % (load_frame, surv_frame))
    print("  per-site loading : %s" % np.round(load, 2))
    if surv is not None:
        print("  per-site survival: %s" % np.round(surv, 2))
    print("  mean loading %.3f (%.2f atoms/shot of %d)" % (load.mean(), load.sum(), len(pts)))
    if surv is not None:
        print("  mean survival %.3f" % np.nanmean(surv))
    return load, e_load, surv, e_surv


def write_calibration(folder, mean_img, pts, active, popts, thr, cfg):
    cal = S.Calibration(
        mean_img, pts, active, thr, popts,
        spacing_um=cfg.get("spacing_um", S.SPACING_UM),
        center=cfg.get("center", S.CENTER_ARRAY),
        f_start_hz=cfg.get("f_start_hz", S.F_START_HZ),
        um_per_MHz=cfg.get("um_per_MHz", S.UM_PER_MHZ),
    )
    path = os.path.join(folder, S.CALIB_CACHE)
    cal.save(path)
    cfg = dict(cfg)
    cfg["atom_threshold"] = float(thr)
    cfg["n_expected"] = int(len(pts))
    S.save_config(folder, cfg)
    print("\n  wrote %s  (%d sites, threshold %.1f)" % (path, len(pts), thr))
    print("  wrote %s  (atom_threshold, n_expected)" % os.path.join(folder, S.CONFIG_NAME))
    print("  run_sort_live.py will now load these with no code change.")
    return cal


# ---------------------------------------------------------------------------- interactive
def interactive(folder, glob_pat, stacks, n_expected, load_frame, surv_frame, frac0, cfg):
    state = {"frac": frac0, "detect_frame": load_frame, "pts": None, "thr": None,
             "auto_thr": None, "active": None, "popts": None,
             "c_load": None, "c_surv": None}
    mean_imgs = [s.mean(0) for s in stacks]

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], width_ratios=[1.05, 1.25, 1.0],
                          left=0.05, right=0.985, top=0.93, bottom=0.20, hspace=0.32, wspace=0.26)
    ax_img, ax_hist, ax_txt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_rate, ax_scan = fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    ax_f = fig.add_axes([0.09, 0.115, 0.55, 0.028])
    ax_t = fig.add_axes([0.09, 0.065, 0.55, 0.028])
    s_frac = Slider(ax_f, "DETECT_FRAC", 0.08, 0.80, valinit=frac0, valstep=0.01)
    s_thr = Slider(ax_t, "ATOM_THRESHOLD", 0.0, 1.0, valinit=0.5)  # rescaled after first fit
    ax_radio = fig.add_axes([0.70, 0.045, 0.11, 0.11])
    radio = RadioButtons(ax_radio, ["detect: f%d" % i for i in range(len(stacks))],
                         active=load_frame)
    b_auto = Button(fig.add_axes([0.83, 0.105, 0.07, 0.045]), "auto thr")
    b_save = Button(fig.add_axes([0.91, 0.105, 0.075, 0.045]), "SAVE npz")
    b_save.label.set_fontweight("bold")

    # one-off: sites-vs-frac curve for the detection frame, so the plateau is visible
    def draw_scan():
        ax_scan.clear()
        fr = np.arange(0.10, 0.801, 0.02)
        n = []
        for f in fr:
            p, _ = detect(mean_imgs[state["detect_frame"]], f, None)
            n.append(np.nan if p is None else len(p))
        ax_scan.plot(fr, n, "-o", ms=3, color="#3070c0")
        if n_expected:
            ax_scan.axhline(n_expected, color="#d04040", ls="--", lw=1,
                            label="expected %d" % n_expected)
            ax_scan.legend(fontsize=7, loc="upper right")
        ax_scan.axvline(state["frac"], color="k", lw=1.2)
        ax_scan.set_xlabel("DETECT_FRAC", fontsize=8)
        ax_scan.set_ylabel("sites found", fontsize=8)
        ax_scan.set_ylim(0, max(3, np.nanmax(n) * 1.1))
        ax_scan.tick_params(labelsize=7)
        ax_scan.set_title("plateau finder (frame %d)" % state["detect_frame"], fontsize=8)

    def recompute_detection():
        pts, msg = detect(mean_imgs[state["detect_frame"]], state["frac"], n_expected)
        state["pts"], state["msg"] = pts, msg
        if pts is None:
            return
        state["active"], state["popts"], state["c_load"], state["c_surv"] = counts_for(
            pts, stacks, load_frame, surv_frame)
        flat = state["c_load"].flatten()
        try:
            a, _fit = _quiet(S.auto_threshold, flat)
        except Exception:  # noqa: BLE001
            a = 0.5 * (flat.min() + flat.max())
        state["auto_thr"] = float(a)
        lo, hi = float(flat.min()), float(flat.max())
        s_thr.valmin, s_thr.valmax = lo, hi
        s_thr.ax.set_xlim(lo, hi)
        if state["thr"] is None or not (lo <= state["thr"] <= hi):
            state["thr"] = float(a)
        s_thr.set_val(state["thr"])   # triggers redraw

    def redraw(_=None):
        state["thr"] = float(s_thr.val)
        pts = state["pts"]
        m = mean_imgs[state["detect_frame"]]

        ax_img.clear()
        ax_img.imshow(m, cmap="viridis", vmin=np.percentile(m, 20), vmax=np.percentile(m, 99.8))
        if pts is not None:
            for i, (x, y) in enumerate(pts):
                ax_img.add_patch(plt.Circle((x, y), S.ROI_SIZE / 2, fill=False,
                                            ec="#ff5555", lw=1.3))
                ax_img.text(x + S.ROI_SIZE / 2 + 1, y, str(i), color="#ffcc55",
                            fontsize=7, va="center")
        ax_img.set_title("mean of %d shots, frame %d  |  frac %.2f"
                         % (len(stacks[0]), state["detect_frame"], state["frac"]), fontsize=9)
        ax_img.tick_params(labelsize=7)

        ax_hist.clear()
        if pts is not None:
            flat = state["c_load"].flatten()
            ax_hist.hist(flat, bins=60, color="#8fb8e0", edgecolor="none",
                         label="loading (frame %d)" % load_frame)
            if state["c_surv"] is not None:
                ax_hist.hist(state["c_surv"].flatten(), bins=60, histtype="step",
                             color="#d06030", lw=1.3,
                             label="survival (frame %d)" % surv_frame)
            ax_hist.axvline(state["thr"], color="k", lw=1.6,
                            label="threshold %.1f" % state["thr"])
            ax_hist.axvline(state["auto_thr"], color="#40a040", ls=":", lw=1.4,
                            label="auto %.1f" % state["auto_thr"])
            ax_hist.set_yscale("log")
            ax_hist.legend(fontsize=7)
        ax_hist.set_xlabel("ROI photon count", fontsize=8)
        ax_hist.set_title("occupancy histogram (all shots x all sites)", fontsize=9)
        ax_hist.tick_params(labelsize=7)

        ax_rate.clear()
        txt = state.get("msg", "")
        if pts is not None:
            load, e_load, surv, e_surv = rates(state["c_load"], state["c_surv"], state["thr"])
            idx = np.arange(len(pts))
            ax_rate.bar(idx - 0.19, load, 0.36, yerr=e_load, capsize=2,
                        color="#4c9be8", label="loading")
            if surv is not None:
                ax_rate.bar(idx + 0.19, surv, 0.36, yerr=e_surv, capsize=2,
                            color="#e8804c", label="survival | loaded")
            ax_rate.axhline(load.mean(), color="#4c9be8", ls="--", lw=1)
            ax_rate.set_xticks(idx)
            ax_rate.set_ylim(0, 1.05)
            ax_rate.legend(fontsize=7, ncol=2)
            txt += "\n\nthreshold %.1f%s\nmean loading %.3f  (%.2f atoms/shot)" % (
                state["thr"], "  [auto]" if abs(state["thr"] - state["auto_thr"]) < 1e-9 else "",
                load.mean(), load.sum())
            if surv is not None:
                txt += "\nmean survival %.3f" % np.nanmean(surv)
        ax_rate.set_xlabel("site index (0 = %s)" % S.FIRST_SITE, fontsize=8)
        ax_rate.set_ylabel("fraction", fontsize=8)
        ax_rate.tick_params(labelsize=7)

        ax_txt.clear()
        ax_txt.axis("off")
        ax_txt.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace",
                    transform=ax_txt.transAxes)
        ax_scan.lines and ax_scan.axvline(state["frac"], color="k", lw=1.2)
        fig.canvas.draw_idle()

    def on_frac(v):
        state["frac"] = float(v)
        recompute_detection()
        draw_scan()
        redraw()

    def on_frame(label):
        state["detect_frame"] = int(label.split("f")[-1])
        recompute_detection()
        draw_scan()
        redraw()

    def on_auto(_):
        s_thr.set_val(state["auto_thr"])

    def on_save(_):
        if state["pts"] is None:
            print("nothing to save - detection failed")
            return
        write_calibration(folder, mean_imgs[load_frame], state["pts"], state["active"],
                          state["popts"], state["thr"], cfg)

    s_frac.on_changed(on_frac)
    s_thr.on_changed(lambda v: redraw())
    radio.on_clicked(on_frame)
    b_auto.on_clicked(on_auto)
    b_save.on_clicked(on_save)

    recompute_detection()
    draw_scan()
    redraw()
    fig.suptitle("calibration tuner  --  %s" % folder, fontsize=10)
    plt.show()
    return state


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--prefix", default=None, help="default: sorter.CALIB_PREFIX")
    ap.add_argument("--n", type=int, default=None, help="expected sites (default sorter.N_SITES)")
    ap.add_argument("--frac", type=float, default=None, help="DETECT_FRAC (default sorter's)")
    ap.add_argument("--load-frame", type=int, default=None, help="default sorter.LOAD_FRAME")
    ap.add_argument("--detect-frame", type=int, default=None, help="default = load frame")
    ap.add_argument("--frames", type=int, default=3, help="frames per shot")
    ap.add_argument("--scan", action="store_true", help="headless: print the frac sweep and exit")
    ap.add_argument("--save", action="store_true", help="headless: write the npz and exit")
    a = ap.parse_args(argv)

    folder = a.folder or S._pick_folder("Select the calibration folder")
    cfg = S.load_config(folder)
    prefix = a.prefix or cfg.get("calib_prefix", S.CALIB_PREFIX)
    n_expected = a.n if a.n is not None else cfg.get("n_expected", S.N_SITES)
    load_frame = a.load_frame if a.load_frame is not None else S.LOAD_FRAME
    frac = a.frac if a.frac is not None else S.DETECT_FRAC
    glob_pat = os.path.join(folder, prefix + "*.npy")

    print("folder     : %s" % folder)
    print("glob       : %s" % glob_pat)
    print("n_expected : %s   load frame: %d   start frac: %.2f" % (n_expected, load_frame, frac))

    stacks = load_all_frames(glob_pat, n_frames=a.frames,
                             binning=cfg.get("binning", S.BINNING))
    print("loaded     : %d shots x %d frames of %s"
          % (len(stacks[0]), len(stacks), stacks[0].shape[1:]))
    surv_frame = load_frame + 1 if load_frame + 1 < len(stacks) else None
    mean_imgs = [s.mean(0) for s in stacks]

    if a.scan:
        scan(mean_imgs, n_expected)
        return 0

    detect_frame = a.detect_frame if a.detect_frame is not None else load_frame

    if a.save:
        pts, msg = detect(mean_imgs[detect_frame], frac, n_expected)
        print("\n  " + (msg or "").replace("\n", "\n  "))
        if pts is None:
            return 1
        active, popts, c_load, c_surv = counts_for(pts, stacks, load_frame, surv_frame)
        thr = cfg.get("atom_threshold")
        if thr is None:
            thr, _ = _quiet(S.auto_threshold, c_load.flatten())
            src = "auto, double-Gaussian valley"
        else:
            src = "from sorter_config.json"
        report(pts, c_load, c_surv, float(thr), src, load_frame, surv_frame)
        write_calibration(folder, mean_imgs[load_frame], pts, active, popts, float(thr), cfg)
        return 0

    interactive(folder, glob_pat, stacks, n_expected, load_frame, surv_frame, frac, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
