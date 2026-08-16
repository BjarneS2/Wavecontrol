"""
Same figure as SpeedOfTrajectories.py, but with a 1/T ("inverse velocity") and an
exponential model fit per trajectory kind, overlaid on the velocity-vs-survival
scatter panel.

@author: Bjarne Schümann
"""

import argparse
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import CommonThings as C
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.linewidth"] = 1.0

try:  # after importing CommonThings the sorter folder should be accessible
    import sorter as S  # type: ignore
except Exception:  # noqa: BLE001
    S = None

CATEGORY = C.CAT_TRANSPORT
DEFAULT_RUNS = (
    None  # None = every measurement run in Data/tweezerImagesSingleAtomTransport
)
SPACING_UM_BY_CONFIG = {
    "46um": 4.6,
    "default": 32.25,
}  # measured, not derived from pixels
SHORT_TRANSFER_CONFIG = "46um"  # 4.6um-spacing hops, excluded from the ALL_RUNS plot
FAULTY_RUNS = {"tweezerLoad1x2-STA200us"}  # bad data taking, excluded from ALL_RUNS plot

_DUR_RE = re.compile(r"(\d+(?:_\d+)?)us")

KIND_STYLE = {
    "linear": ("tab:blue", "o"),
    "min-jerk": ("tab:red", "^"),
    "optimised": ("tab:green", "s"),
    "optimised + amp": ("tab:orange", "D"),
    "opt + amp, half offset": ("tab:purple", "P"),
    "other": ("0.5", "x"),
}


def mean_velocity(d_sites, tau, spacing_um=None):
    """Mean velocity of one min-jerk hop. Can be over multiple sites d.
    Just the distance travelled over the hop duration:
        v_mean = d * spacing / tau
    """
    sp = (S.SPACING_UM if spacing_um is None else spacing_um) * 1e-6  # type: ignore
    return np.asarray(d_sites, float) * sp / tau


def inv_time_model(v, a, b):
    """p(v) = a/v + b, i.e. a 1/T model re-expressed in velocity (T = d/v)."""
    return a / v + b


def exp_model(v, amp, rate, offset):
    """p(v) = amp * exp(-rate * v) + offset."""
    return amp * np.exp(-rate * v) + offset


def _run_kind(name):
    if "OptimizedWithAmpHalfFinalOffset" in name:
        return "opt + amp, half offset"
    if "OptimizedWithAmp" in name:
        return "optimised + amp"
    if "Optimized" in name:
        return "optimised"
    if "STA" in name:
        return "min-jerk"
    if "Linear" in name:
        return "linear"
    return "other"


def _duration_us(name):
    m = _DUR_RE.search(name)
    return float(m.group(1).replace("_", ".")) if m else None


def _load_runs(images=None, runs=None, min_shots=1):
    """
    Load both SingleAtomTransport calibration sessions.
    """
    images = images or str(C.TRANSPORT_IMAGES)
    per_run = {}
    for cfg in C.TRANSPORT_CONFIGS:
        _, _, cfg_runs = C.load_transport_config(images, cfg, min_shots)
        if runs:
            cfg_runs = {k: v for k, v in cfg_runs.items() if any(r in k for r in runs)}
        if not cfg_runs:
            continue
        pool = np.concatenate([lc[0] for lc, _ in cfg_runs.values()])
        thr, _ = C.bimodal_threshold(pool)
        for name, (lc, sc) in cfg_runs.items():
            per_run[name] = (lc, sc, thr)
    return per_run


def _run_fates(load_counts, surv_counts, thr):
    """(n loaded, n arrived at target, n left behind at source, n gone entirely)."""
    lo = load_counts[0] > thr
    at_target = surv_counts[1] > thr
    at_origin = surv_counts[0] > thr
    nl = int(lo.sum())
    arr = int((lo & at_target).sum())
    left_behind = int((lo & ~at_target & at_origin).sum())
    return nl, arr, left_behind, nl - arr - left_behind


def _fit_curves(rows, kind):
    sub = [r for r in rows if r["kind"] == kind]
    v = np.array([r["v_mean"] for r in sub])
    p = np.array([r["arr"] / r["nl"] for r in sub])
    grid = np.linspace(v.min(), v.max(), 200)
    curves = {}
    try:
        popt, _ = curve_fit(inv_time_model, v, p, p0=[p[np.argmin(v)] * v.min(), 0.0])
        curves["1/T"] = (grid, inv_time_model(grid, *popt), popt)
    except RuntimeError:
        pass
    try:
        popt, _ = curve_fit(
            exp_model, v, p, p0=[p.max(), 1.0 / v.mean(), 0.0], maxfev=20000
        )
        curves["exp"] = (grid, exp_model(grid, *popt), popt)
    except RuntimeError:
        pass
    return curves


def _figure(rows, ax):
    """mean transport velocity vs survival, colored/marked by kind, with 1/T and
    exponential fits overlaid per kind."""
    seen = set()
    for r in rows:
        color, marker = KIND_STYLE.get(r["kind"], KIND_STYLE["other"])
        label = r["kind"] if r["kind"] not in seen else None
        seen.add(r["kind"])
        p, lo, hi = C.wilson(r["arr"], r["nl"])
        ax.errorbar(
            r["v_mean"],
            [p],
            yerr=[[p - lo], [hi - p]],
            marker=marker,
            ms=8,
            capsize=3,
            color=color,
            label=label,
        )
        ax.annotate(
            f"{r['T']:g}us",
            (r["v_mean"], p),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    fit_styles = {"1/T": "--", "exp": ":"}
    for kind in seen:
        color, _ = KIND_STYLE.get(kind, KIND_STYLE["other"])
        curves = _fit_curves(rows, kind)
        for name, (grid, fit, _) in curves.items():
            ax.plot(
                grid,
                fit,
                fit_styles[name],
                color=color,
                lw=1.6,
                label=f"{kind} {name} fit",
            )
    ax.set_xlabel("mean transport velocity [m/s]")
    ax.set_ylabel("P(arrived at target | loaded at source)")
    ax.set_ylim(0, 1.02)
    ax.text(-0.12, 1.02, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.legend(fontsize=7.5, framealpha=0.95)
    ax.grid(alpha=0.3)


def _stacked_figure(rows, ax):
    """Stacked bar of arrived/lost/left-behind probabilities, grouped by duration
    then kind (linear before min-jerk), same colors as the scatter for arrivals."""
    order = sorted(rows, key=lambda r: (-r["T"], r["kind"] != "linear"))
    x = np.arange(len(order))
    arrived = np.array([r["arr"] / r["nl"] for r in order])
    left_behind = np.array([r["left_behind"] / r["nl"] for r in order])
    lost = np.array([r["lost"] / r["nl"] for r in order])
    colors = [KIND_STYLE.get(r["kind"], KIND_STYLE["other"])[0] for r in order]
    ax.bar(x, arrived, color=colors)
    ax.bar(x, lost, bottom=arrived, color="0.6")
    ax.bar(x, left_behind, bottom=arrived + lost, color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['kind']}\n{r['T']:g}us" for r in order], fontsize=8)
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.02)
    ax.text(-0.12, 1.02, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.legend(
        handles=[
            Patch(color="0.6", label="lost"),
            Patch(color="gold", label="left behind"),
        ],
        fontsize=8.5,
        framealpha=0.95,
    )
    ax.grid(alpha=0.3)


def plot(runs=None, save=True, images=None):
    """Builds the per-run velocity-vs-survival scatter (with fits) next to the
    outcome breakdown bar chart. Returns the Figure."""
    combined = runs is None
    per_run = _load_runs(images, runs if runs is not None else DEFAULT_RUNS)
    rows = []
    for name, (load_counts, surv_counts, thr) in per_run.items():
        if combined and C.transport_config_of(name) == SHORT_TRANSFER_CONFIG:
            continue
        if combined and name in FAULTY_RUNS:
            continue
        nl, arr, left_behind, lost = _run_fates(load_counts, surv_counts, thr)
        T = _duration_us(name)
        if T is None or nl == 0:
            continue
        sp = SPACING_UM_BY_CONFIG[C.transport_config_of(name)]
        v_mean = float(mean_velocity(1, T * 1e-6, spacing_um=sp))
        rows.append(
            {
                "name": name,
                "T": T,
                "kind": _run_kind(name),
                "v_mean": v_mean,
                "nl": nl,
                "arr": arr,
                "left_behind": left_behind,
                "lost": lost,
            }
        )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _figure(rows, ax1)
    _stacked_figure(rows, ax2)
    fig.tight_layout()
    if save:
        name = C.POOLED if not runs else "_".join(runs)
        C.save_figure(fig, CATEGORY, name, "speed_of_trajectories_fits")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="substrings selecting run prefixes"
    )
    ap.add_argument(
        "--images", default=None, help="override the transport images folder"
    )
    ap.add_argument("--list", action="store_true", help="list matching runs and exit")
    ap.add_argument(
        "--no-save", action="store_true", help="show interactively instead of saving"
    )
    a = ap.parse_args(argv)

    if a.list:
        per_run = _load_runs(a.images, a.runs)
        print(f"spacing_um by session: {SPACING_UM_BY_CONFIG}")
        for name in sorted(per_run):
            _, _, thr = per_run[name]
            cfg = C.transport_config_of(name)
            print(
                f"  {name}  T={_duration_us(name)}us  kind={_run_kind(name)}  "
                f"session={cfg}  spacing={SPACING_UM_BY_CONFIG[cfg]:.2f}um  thr={thr:.1f}"
            )
        return 0

    plot(runs=a.runs, save=not a.no_save, images=a.images)
    if a.no_save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
