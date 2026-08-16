"""
This script is for common functions and parameters for the
analysis and plotting functions to come back to.

@author: Bjarne Schümann
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.linewidth"] = 1.0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # the real thing always wins over the fallbacks below
    import Sorting.sorter as S
except Exception:  # pragma: no cover  # noqa: BLE001
    S = None

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
FIGURES_DIR = REPO_ROOT / "Figures"

SORTING1D_IMAGES = DATA_DIR / "tweezerImagesSorting1D"
SORTING1D_CAL = SORTING1D_IMAGES / "calibration"
SORTING1D_REPORTS = DATA_DIR / "reportforSorting1D"
TRANSPORT_IMAGES = DATA_DIR / "tweezerImagesSingleAtomTransport"

CAT_SORTING = "Sorting"
CAT_TRANSPORT = "Transportation"
POOLED = "ALL_RUNS"  # run_name -- for pooled/multi-run figures
DEFAULT_DPI = 300  # nice resolution images


TRANSPORT_CONFIGS = {
    "46um": {"reference": "46umMephisto", "r_pix": 3.0},
    "default": {"reference": "Mephisto", "r_pix": 4.0},
}

LOAD_FRAME = 1
SURV_FRAME = 2
DETECT_FRAME = 0
SHOT_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)_(?P<ts>\d{8}-\d{6})\.npy$")
_MASKISH = ("mask", "occupancy", "occupied", "loaded", "filled", "sites")
_MOVEISH = ("moves", "move_list", "moveseq", "sort_moves")


# to save figures I wanna use always the same procedure to save conventions made here:
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def _safe_name(s):
    """Strip characters Windows can't put in a path component."""
    return _UNSAFE_PATH_CHARS.sub("_", str(s)).strip()


def save_figure(fig, category, run_name, title, **savefig_kwargs):
    """fig -> Figures/<category>/<run_name>/<title>.png, dirs made as needed."""
    out = (
        FIGURES_DIR
        / _safe_name(category)
        / _safe_name(run_name)
        / f"{_safe_name(title)}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **{"dpi": DEFAULT_DPI, **savefig_kwargs})
    return out


# For the sorting experiment we are going to need some functions that are able to do some
# very repetitive things handled by these functions
def to_photons(im, binning):
    """
    Integer input only. Float input is assumed to be photons already and passed
    through.
    """
    im = np.asarray(im)
    if not np.issubdtype(im.dtype, np.integer):
        return im.astype(np.float64)
    if S is not None:
        return S._counts_to_photons(im, binning)
    return (im.astype(np.int32) - 200 * int(binning) ** 2) * 0.1


def read_images(path, binning):
    """One shot -> (n_frames, H, W) photon image, from the pickled {"Images": ...} dict."""
    a = np.load(path, allow_pickle=True)
    if a.dtype == object or a.shape == ():
        d = a[()]
        im = d["Images"] if isinstance(d, dict) else d
    else:
        im = a
    return to_photons(np.asarray(im), binning)


#: run-name groups that must always be treated/counted as ONE pooled run. Each tuple is
#: (sub_name, sub_name, ..., merged_name). tweezerLoad1x11-sortbest-lin80us is a DIFFERENT
#: run and must never be swept in here.
MERGE_GROUPS = (
    (
        "tweezerLoad1x11-bestsort-80us",
        "tweezerLoad1x11-sortbest-80usAGAIN",
        "tweezerLoad1x11-80us_pooled",
    ),
)


def merge_run_names(names, groups=MERGE_GROUPS):
    """Run names -> same list with every configured group collapsed to its merged name
    (order-preserving, deduplicated)."""
    sub_to_merged = {s: g[-1] for g in groups for s in g[:-1]}
    out, seen = [], set()
    for n in names:
        m = sub_to_merged.get(n)
        if m is None:
            out.append(n)
        elif m not in seen:
            out.append(m)
            seen.add(m)
    return out


def merge_run_datasets(run_dss, groups=MERGE_GROUPS, **build_plans_kwargs):
    """Datasets whose .name matches a configured group -> collapsed into one pooled
    Dataset (paths/shot_idx/stamps/counts concatenated, plans rebuilt). Datasets outside
    any group pass through unchanged; order-preserving."""
    build_plans_kwargs.setdefault("verbose", False)
    run_dss = list(run_dss)
    for *subs, merged_name in groups:
        part = [d for d in run_dss if d.name in subs]
        if len(part) < 2:
            continue
        idx = min(i for i, d in enumerate(run_dss) if d.name in subs)
        merged = Dataset(
            merged_name,
            sum((d.paths for d in part), []),  # noqa: RUF017
            np.concatenate([d.shot_idx for d in part]),
            sum((d.stamps for d in part), []),  # noqa: RUF017
            np.concatenate([d.counts for d in part]),
            part[0].mean_frames,
            part[0].cal,
        )
        build_plans(merged, **build_plans_kwargs)
        part_ids = {id(d) for d in part}
        run_dss = [d for d in run_dss if id(d) not in part_ids]
        run_dss.insert(idx, merged)
    return run_dss


def discover_runs(folder, min_shots=1, verbose=True):
    """Flat folder -> {prefix: [(shot_index, timestamp, path), ...]} sorted by shot index.

    Files that do not match the naming scheme are grouped under the folder name and kept
    in filename order, so the tool still works on a folder holding a single run.
    """
    groups = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        m = SHOT_RE.match(os.path.basename(p))
        if m:
            groups.setdefault(m["prefix"], []).append((int(m["idx"]), m["ts"], p))
        else:
            g = groups.setdefault(os.path.basename(os.path.normpath(folder)), [])
            g.append((len(g), "", p))
    for k in groups:  # noqa: PLC0206
        groups[k].sort(key=lambda t: t[0])

    small = {k: v for k, v in groups.items() if len(v) < min_shots}
    groups = {k: v for k, v in groups.items() if len(v) >= min_shots}
    if verbose and small:
        print(
            f"  skipped {len(small)} group(s) below --min-shots: "
            + ", ".join(f"{k} ({len(v)})" for k, v in small.items())
        )
    return groups


@dataclass
class Calibration:
    """
    This just contains all the information from the calibration.
    """

    locations: np.ndarray  # (n,2) as (x, y) = (column, row), trap-index order
    active: np.ndarray  # (H,W) bool, union of the per-site Gaussian fits
    threshold: float  # the atom_threshold that was LIVE
    n: int
    mean_img: np.ndarray | None = None
    binning: int = 2
    roi: int = 10
    source: str = ""
    per_site: np.ndarray | None = None

    def thr(self, per_site=False):
        if per_site:
            if self.per_site is None:
                raise ValueError(
                    "per-site thresholds requested but none are set -- fit them first"
                )
            return np.asarray(self.per_site, float)
        return np.full(self.n, float(self.threshold))


# Now we need to fill our data holding object with some from the saved calibraiton:
def load_calibration(folder, threshold=None, verbose=True):
    """Read the npz + json that were LIVE during the runs. Nothing is refitted here."""
    npz = os.path.join(folder, "sorter_calibration.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"no sorter_calibration.npz in {folder}")
    cfg = {}
    cfg_path = os.path.join(folder, "sorter_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))

    if S is not None:
        c = S.load_calibration(npz)
        cal = Calibration(
            np.asarray(c.locations, float),
            np.asarray(c.active, bool),
            float(c.threshold),
            int(c.n),
            np.asarray(c.mean_img, float),
            int(cfg.get("binning", getattr(S, "BINNING", 2))),
            int(getattr(S, "ROI_SIZE", 10)),
            "sorter.load_calibration",
        )
    else:
        d = np.load(npz, allow_pickle=False)
        loc = np.asarray(d["locations"], float)
        cal = Calibration(
            loc,
            np.asarray(d["active"], bool),
            float(d["threshold"]),
            len(loc),
            np.asarray(d["mean_img"], float),
            int(cfg.get("binning", 2)),
            10,
            npz,
        )
    if threshold is not None:
        cal.threshold = float(threshold)

    if verbose:
        print(
            f"  calibration: n={cal.n}  live atom_threshold={cal.threshold:.3f}  "
            f"binning={cal.binning}  active={int(cal.active.sum())} px  ({cal.source})"
        )
        if S is not None:
            u = S.array_axis(cal.locations)  # type: ignore
            proj = (cal.locations - cal.locations.mean(0)) @ u
            if not np.all(np.diff(proj) > 0):
                print(
                    "  ! locations are NOT monotonic along the array axis -- site "
                    "indices in the npz do not match trap order; stop and recalibrate"
                )
        n_exp = cfg.get("n_expected")
        if n_exp and int(n_exp) != cal.n:
            print(f"  ! npz has {cal.n} sites but the config expected {n_exp}")
    return cal


def _site_counts_fallback(img, locations, active, roi):
    """
    Given locations, what are the counts in the region of interest given the active area.
    All pixels outside active area set to 0.
    """
    H, W = img.shape
    masked = np.where(active, img, 0.0)
    out = np.empty(len(locations))
    for i, (xc, yc) in enumerate(locations):
        c0 = max(0, min(int(round(xc - roi / 2)), W - roi))
        r0 = max(0, min(int(round(yc - roi / 2)), H - roi))
        out[i] = masked[r0 : r0 + roi + 1, c0 : c0 + roi + 1].sum()
    return out


def counts_from_images(im, cal):
    """Call site counts for multiple frames."""
    func = (
        S.site_counts if S is not None else _site_counts_fallback
    )  # had some import errors
    return np.array(
        [func(im[f], cal.locations, cal.active, cal.roi) for f in range(len(im))]
    )


def occupancy(counts, thr):
    """counts > threshold"""
    return counts > np.asarray(thr, float)


def count_stats(x, bins=None):
    """Robust description of one count histogram.
    -> dict(thr, empty, filled, valley_depth, bridge, bimodal, n)

    Due to only few actual occurences per site (say ~50) the fits might not converge
    or converge to something "random". For two tallest peak searches one might latch
    onto some noise peaks with that low numbers of shots.
    https://en.wikipedia.org/wiki/Otsu%27s_method
    The Otsu method still works, not the best but good enough. Slide threshold and get
    some mid-gap value (if gap is really empty).
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    out = dict(  # noqa: C408
        thr=np.nan,
        empty=np.nan,
        filled=np.nan,
        valley_depth=np.nan,
        bridge=np.nan,
        bimodal=False,
        n=int(x.size),
    )
    if x.size < 20 or np.ptp(x) <= 0:
        return out
    nb = int(np.clip(2 * np.sqrt(x.size), 24, 80))
    h, edges = np.histogram(x, bins=nb)
    c = 0.5 * (edges[:-1] + edges[1:])

    tot = h.sum()
    w0 = np.cumsum(h)[:-1].astype(float)
    w1 = tot - w0
    cum = np.cumsum(h * c)
    m0 = np.divide(cum[:-1], w0, out=np.zeros_like(w0), where=w0 > 0)
    m1 = np.divide(cum[-1] - cum[:-1], w1, out=np.zeros_like(w1), where=w1 > 0)
    sb = np.where((w0 > 0) & (w1 > 0), w0 * w1 * (m0 - m1) ** 2, -1.0)
    thr = float(edges[int(np.argmax(sb)) + 1])

    lo, hi = x[x < thr], x[x >= thr]
    if lo.size < 3 or hi.size < 3:
        return out
    e, f = float(np.median(lo)), float(np.median(hi))
    mad = 1.4826 * float(np.median(np.abs(lo - e)))
    if mad <= 0:
        mad = max(float(lo.std()), 1e-9)

    k = max(3, nb // 15)
    hs = np.convolve(h.astype(float), np.ones(k) / k, mode="same")
    band_idx = np.flatnonzero((c >= e) & (c <= f))
    if band_idx.size:
        sub = hs[band_idx]
        flat = band_idx[sub <= sub.min() + 1e-12]  # mid-gap when the gap is empty
        thr = float(c[flat[len(flat) // 2]])

    ti = int(np.argmin(np.abs(c - thr)))
    ei = int(np.argmin(np.abs(c - e)))
    fi = int(np.argmin(np.abs(c - f)))
    depth = float(hs[ti] / max(1e-9, min(hs[ei], hs[fi])))
    band = 0.15 * (f - e)
    out.update(
        thr=thr,
        empty=e,
        filled=f,
        valley_depth=depth,
        bridge=float(np.mean((x > thr - band) & (x < thr + band))),
        bimodal=bool((f - e) / mad > 4.0),
    )
    return out


def valley_threshold(x, bins=None):
    """
    Threshold for one pooled / per-site histogram;
    np.nan if the site is unimodal not bimodal.
    """
    st = count_stats(x, bins)
    return st["thr"] if st["bimodal"] else np.nan


@dataclass
class Plan:
    mask: np.ndarray  # (n,) bool, from frame 1 (loading)
    sites: np.ndarray  # (K,) currently occupied indices, ascending
    t0: int
    targets: np.ndarray  # (K,) destin. idx of each atom
    order: list  # execution order, should avoid crossings / crashes
    source: str = "internal"

    @property
    def dist(self):
        return self.targets - self.sites

    @property
    def moved(self):
        return self.targets != self.sites

    @property
    def k(self):
        return len(self.sites)

    @property
    def n_moves(self):
        return int(self.moved.sum())


def best_window(sites, n):
    """Essentially the same as find_best_window from sorter.py"""
    # if S is not None:
    #    return S.find_best_window(sites)
    k = len(sites)
    disp = np.clip(np.asarray(sites) - np.arange(k), 0, n - k)
    starts, counts = np.unique(disp, return_counts=True)
    return int(starts[int(np.argmax(counts))])


def execution_order(sites, targets):
    """sorter.plan_moves order: left-movers by ascending source, then right-movers by
    descending source, so no path crosses and no traps collide"""
    idx = range(len(sites))
    left = sorted((i for i in idx if targets[i] < sites[i]), key=lambda i: sites[i])
    right = sorted((i for i in idx if targets[i] > sites[i]), key=lambda i: -sites[i])
    return left + right


def plan_internal(mask):
    mask = np.asarray(mask, bool)
    sites = np.flatnonzero(mask)
    if sites.size == 0:
        return Plan(mask, sites, 0, sites.copy(), [])
    t0 = best_window(sites, len(mask))
    targets = t0 + np.arange(len(sites))
    return Plan(mask, sites, t0, targets, execution_order(sites, targets))


def plan_with_sorter(mask, start_window=None):
    """sorter.plan_moves(mask) -> (target_sites, [(src, dst), ...] in execution order)."""
    if S is None:
        return None
    mask = np.asarray(mask, bool)
    sites = np.flatnonzero(mask)
    targets, moves = S.plan_moves(mask, start_window=start_window)
    targets = np.asarray(targets, int)
    if sites.size == 0:
        return Plan(mask, sites, 0, sites.copy(), [], "sorter.plan_moves")
    if targets.shape != sites.shape:
        raise ValueError(
            f"plan_moves returned {targets.shape} targets for {sites.shape} atoms"
        )
    where = {int(s): i for i, s in enumerate(sites)}
    order = [where[int(src)] for src, _ in moves]
    return Plan(mask, sites, int(targets[0]), targets, order, "sorter.plan_moves")


def plan(mask, use_sorter=True, xcheck=None, start_window=None):
    """Plan one shot. sorter's planner is the truth; cross-checks it."""
    A = plan_with_sorter(mask, start_window) if use_sorter else None
    if A is None:
        return plan_internal(mask)
    if xcheck is not None:
        B = plan_internal(mask)
        if not np.array_equal(A.targets, B.targets):
            xcheck.append(int(np.asarray(mask).sum()))
    return A


def _find_mask(obj, n):
    """
    Depth-first hunt for something that looks like an n-site
    occupancy mask in a dict or obj with _MASKISH.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in str(k).lower() for t in _MASKISH):
                m = _as_mask(v, n)
                if m is not None:
                    return m
        for v in obj.values():
            m = _find_mask(v, n)
            if m is not None:
                return m
        return None
    if isinstance(obj, (list, tuple)):
        m = _as_mask(obj, n)
        if m is not None:
            return m
        for v in obj:
            m = _find_mask(v, n)
            if m is not None:
                return m
    if isinstance(obj, str):
        s = re.sub(r"[^01]", "", obj)
        return _as_mask(list(s), n) if len(s) == n else None
    return None


def _as_mask(v, n):
    try:
        a = np.asarray([int(x) for x in v])
    except Exception:  # noqa: BLE001
        return None
    if a.size == n and set(np.unique(a)) <= {0, 1}:
        return a.astype(bool)
    if a.size and a.max() < n and a.size < n and set(np.unique(a)) - {0, 1}:
        m = np.zeros(n, bool)  # looks like a list of occupied indices
        m[a] = True
        return m
    return None


def _find_moves(obj):
    """same as find mask but for moves"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in str(k).lower() for t in _MOVEISH):
                return v
        for v in obj.values():
            r = _find_moves(v)
            if r is not None:
                return r
    return None


# This just says how the Report is bein read/written, or rather with which info
@dataclass
class Report:
    path: str
    mask: np.ndarray | None
    moves: object = None


def read_reports(paths, n, verbose=True):
    """Parse report files in sorted order. json / jsonl / npz / npy / txt / csv."""
    files = []
    for p in paths:
        files.extend(
            sorted(glob.glob(p))
            if any(c in p for c in "*?[")
            else (sorted(glob.glob(os.path.join(p, "*"))) if os.path.isdir(p) else [p])
        )
    out = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext in (".json", ".jsonl", ".txt", ".csv", ".log", ""):
                text = open(f, "r", errors="replace").read()
                try:
                    objs = [json.loads(text)]
                except Exception:
                    objs = []
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            objs.append(json.loads(line))
                        except Exception:
                            objs.append(line)
                for o in objs:
                    m = _find_mask(o, n)
                    if m is not None:
                        out.append(
                            Report(
                                f, m, _find_moves(o) if isinstance(o, dict) else None
                            )
                        )
            elif ext == ".npz":
                d = np.load(f, allow_pickle=False)
                m = _find_mask({k: d[k].tolist() for k in d.files}, n)
                if m is not None:
                    out.append(Report(f, m))
            elif ext == ".npy":
                m = _as_mask(np.load(f, allow_pickle=False).ravel().tolist(), n)
                if m is not None:
                    out.append(Report(f, m))
        except Exception as e:
            if verbose:
                print(f"  ! could not parse {f}: {e}")
    if verbose:
        print(f"  reports: {len(out)} masks parsed from {len(files)} files")
    return out


def align_reports(plans, reports, n, tol=3):
    """Align reports (a SUBSEQUENCE of the runs) to shots by mask fingerprint.
    Subsequence since only runs that got classified and need moving are actually
    reported, the rest is not. We then look for the longest commom subsequence to
    be able to aling a run with a report.
    Returns (mapping shot -> report index or -1, diagnostics).
    """
    R, Sh = len(reports), len(plans)
    if R == 0 or Sh == 0:
        return np.full(Sh, -1, int), {"aligned": 0, "exact": 0}
    A = np.array([p.mask for p in plans], bool)
    B = np.array([r.mask for r in reports], bool)
    bits = (A.astype(int) @ B.astype(int).T) + ((~A).astype(int) @ (~B).astype(int).T)
    sc = 3.0 * bits - 3.0 * (n - tol)  # >0 iff fewer than `tol` bits differ
    dp = np.zeros((Sh + 1, R + 1))
    back = np.zeros((Sh + 1, R + 1), np.int8)
    for i in range(1, Sh + 1):
        row, prev = dp[i], dp[i - 1]
        for j in range(1, R + 1):
            take = prev[j - 1] + sc[i - 1, j - 1]
            skip_shot = prev[j]
            skip_rep = row[j - 1]
            if take >= skip_shot and take >= skip_rep:
                row[j], back[i, j] = take, 1
            elif skip_shot >= skip_rep:
                row[j], back[i, j] = skip_shot, 2
            else:
                row[j], back[i, j] = skip_rep, 3
    mapping = np.full(Sh, -1, int)
    i, j = Sh, R
    while i > 0 and j > 0:
        b = back[i, j]
        if b == 1:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    hit = mapping >= 0
    exact = sum(
        1
        for s_ in range(Sh)
        if hit[s_] and np.array_equal(plans[s_].mask, reports[mapping[s_]].mask)
    )
    unreported_with_moves = [
        s_ for s_ in range(Sh) if not hit[s_] and plans[s_].n_moves > 0
    ]
    reported_without_moves = [
        s_ for s_ in range(Sh) if hit[s_] and plans[s_].n_moves == 0
    ]
    return mapping, {
        "aligned": int(hit.sum()),
        "exact": exact,
        "mismatched": int(hit.sum()) - exact,
        "reports_unused": int(R - hit.sum()),
        "unreported_with_moves": unreported_with_moves,
        "reported_without_moves": reported_without_moves,
    }


def window_misfit(masks, m2, cal_n, start_window):
    """Fraction of frame 2 (survival) atoms sitting OUTSIDE the predicted target block.
    This would either mean ghost atoms - atoms that were not detected in the first image
    and then the coincendence occured that the atoms that were found are already sorted
    leading to no change or turn off in other traps -> we don't lose the undetected one
    but if gets found in the second image. It could also mean something goes wrong in the
    execution of the moves and how the experiment was supposed to work. So it is also kind´
    of a sanity / safety check."""
    out = tot = 0
    for i, m in enumerate(masks):
        p = plan(m, start_window=start_window)
        if p.k == 0:
            continue
        blk = np.zeros(cal_n, bool)
        blk[p.t0 : p.t0 + p.k] = True
        out += int((m2[i] & ~blk).sum())
        tot += int(m2[i].sum())
    return out / max(1, tot)


def detect_start_window(ds, per_site=False, verbose=True):
    """Which start_window did this run actually use? Read it off frame 2.
    this could be misleading if lots of atoms are lost or as if we found
    before some atoms that were misfits / ghosts."""
    masks, m2 = ds.occ(LOAD_FRAME, per_site), ds.occ(SURV_FRAME, per_site)
    cands = [None] + list(range(ds.cal.n))
    mis = {c: window_misfit(masks, m2, ds.cal.n, c) for c in cands}
    best = min(mis, key=lambda c: (mis[c], c is not None))  # prefer None on a tie
    if verbose:
        tag = (
            "auto (move-minimising)" if best is None else f"forced start_window={best}"
        )
        print(
            f"    window: {tag}; outside-block atoms {mis[best]:.3f} (auto {mis[None]:.3f})"
        )
        if best is not None and mis[None] > 5 * max(mis[best], 1e-3):
            print(
                f"    ! this run did NOT use the move-minimising window; using {best}"
            )
    return best, mis


@dataclass
class Dataset:
    name: str
    paths: list
    shot_idx: np.ndarray
    stamps: list
    counts: np.ndarray  # (n_shots, n_frames, n_sites)
    mean_frames: np.ndarray  # (n_frames, H, W)
    cal: Calibration
    is_calibration: bool = False
    reports: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    align: dict = field(default_factory=dict)
    mapping: np.ndarray | None = None
    xcheck: list = field(default_factory=list)
    start_window: int | None = None
    window_misfit: float = np.nan

    def occ(self, frame, per_site=False):
        return occupancy(self.counts[:, frame, :], self.cal.thr(per_site))

    @property
    def n_shots(self):
        return self.counts.shape[0]


def load_dataset(name, entries, cal, is_calibration=False, verbose=True):
    """entries: [(shot_index, timestamp, path), ...] already in shot order.

    Only the counts and the per-frame mean image are kept -- the full frame stack of a
    single run can be hundreds of MB, and there can be a dozen or more runs in one pass.
    """
    n_f = None
    counts = []
    acc = None
    for _, _, p in entries:
        im = read_images(p, cal.binning)
        if n_f is None or acc is None:
            n_f = im.shape[0]
            acc = np.zeros(im.shape, float)
        counts.append(counts_from_images(im, cal))
        acc += im
    if acc is None:
        raise ValueError(
            "Something happened that should be impossible - acc variable is still None."
        )
    counts = np.asarray(counts)
    ds = Dataset(
        name,
        [e[2] for e in entries],
        np.array([e[0] for e in entries]),
        [e[1] for e in entries],
        counts,
        acc / max(1, len(entries)),
        cal,
        is_calibration,
    )
    if verbose:
        print(f"  {name}: {ds.n_shots} shots, {n_f} frames")
    return ds


# the idea to make misfits and ghosts less likely:
# optimizing thresholds for individual per trap possonian statistic photons counts
def per_site_thresholds(counts, fallback, verbose=True):
    n = counts.shape[-1]
    thr = np.empty(n)
    fell_back = []
    for i in range(n):
        t = valley_threshold(counts[..., i])
        if not np.isfinite(t):
            t = fallback
            fell_back.append(i)
        thr[i] = t
    if verbose:
        print(
            "  per-site thresholds: " + np.array2string(thr, precision=1, separator=" ")
        )
        if fell_back:
            print(f"    (sites {fell_back} not bimodal -> global {fallback:.1f})")
    return thr


def build_plans(ds, per_site=False, use_sorter=True, start_window="auto", verbose=True):
    """Classify frame 1 with the LIVE threshold, then re-plan each shot."""
    if start_window == "auto":
        start_window, mis = detect_start_window(ds, per_site, verbose)
        ds.window_misfit = mis[start_window]
    ds.start_window = start_window
    masks = ds.occ(LOAD_FRAME, per_site)
    ds.xcheck = []
    ds.plans = [plan(m, use_sorter, ds.xcheck, start_window) for m in masks]
    if verbose:
        src = ds.plans[0].source if ds.plans else "n/a"
        k = np.array([p.k for p in ds.plans])
        mv = np.array([p.n_moves for p in ds.plans])
        print(
            f"    planner={src}  <K>={k.mean():.2f}/{ds.cal.n} "
            f"(loading {k.mean() / ds.cal.n:.3f})  <moves>={mv.mean():.2f}  "
            f"{int((mv == 0).sum())}/{ds.n_shots} shots needed no move"
        )
        if ds.xcheck:
            print(
                f"    ! the reimplementation disagreed with sorter.plan_moves on "
                f"{len(ds.xcheck)} shots"
            )
    return ds


# since wald interval struggles to give reasonable error bars for extremely low or high probabilities
# we need to use wilson interval instead which curves and correctly stays within 0 and 1 for probs..
def wilson(k, n, z=1.0):
    """(p, lo, hi). z=1 -> 1 sigma. More honest at p near 0 or 1, unlike sqrt(p(1-p)/n) (wald interval)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (p, min(lo, p), max(hi, p))


def survival_tables(cal_ds, run_dss, per_site=False, k_range=None):
    """
    We wanna reduce redundant computation and the redefinition of the same
    function in all of the plotting scripts keeping those to a minimum therefore:
    Survival tables shared by several plots computed here.

    Calibration:  same site, frame1 -> frame2, nothing moved  = the imaging-loss floor.
    Runs:         stationary atoms checked at their site, movers at their TARGET.
    Movers are additionally split into at-target / still-at-origin / gone.
    """
    out = {
        "cal": {"k": 0, "n": 0, "per_site_k": None, "per_site_n": None},
        "stat": {"k": 0, "n": 0},
        "move": {"k": 0, "n": 0},
        "by_dist": {},
        "by_kd": {},
        "by_k": {},
        "mover_fate": {"target": 0, "origin": 0, "gone": 0},
    }

    if cal_ds is not None:
        m1, m2 = cal_ds.occ(LOAD_FRAME, per_site), cal_ds.occ(SURV_FRAME, per_site)
        out["cal"]["k"] = int((m1 & m2).sum())
        out["cal"]["n"] = int(m1.sum())
        out["cal"]["per_site_k"] = (m1 & m2).sum(0)
        out["cal"]["per_site_n"] = m1.sum(0)

    for ds in run_dss:
        m2 = ds.occ(SURV_FRAME, per_site)
        for s, p in enumerate(ds.plans):
            blk = np.zeros(ds.cal.n, bool)
            if p.k:
                blk[p.t0 : p.t0 + p.k] = True
            if p.k == 0 or (k_range and not (k_range[0] <= p.k <= k_range[1])):
                continue
            bk = out["by_k"].setdefault(p.k, {"stat": [0, 0], "move": [0, 0]})
            for a in range(p.k):
                site, tgt, d = int(p.sites[a]), int(p.targets[a]), int(p.dist[a])
                alive = bool(m2[s, tgt])
                bk["stat" if d == 0 else "move"][1] += 1
                bk["stat" if d == 0 else "move"][0] += alive
                if d != 0:
                    c = out["by_kd"].setdefault((p.k, abs(d)), [0, 0])
                    c[1] += 1
                    c[0] += alive
                if d == 0:
                    out["stat"]["n"] += 1
                    out["stat"]["k"] += alive
                else:
                    out["move"]["n"] += 1
                    out["move"]["k"] += alive
                    b = out["by_dist"].setdefault(
                        abs(d), {"k": 0, "n": 0, "kL": 0, "nL": 0, "kR": 0, "nR": 0}
                    )
                    b["n"] += 1
                    b["k"] += alive
                    side = "R" if d > 0 else "L"
                    b["n" + side] += 1
                    b["k" + side] += alive
                    if alive:
                        out["mover_fate"]["target"] += 1
                    elif m2[s, site] and not blk[site]:
                        out["mover_fate"]["origin"] += 1
                    else:
                        out["mover_fate"]["gone"] += 1
    return out


def success_vs_k(run_dss, kmax=10, per_site=False):
    """

    P(perfect sort | K loaded) per dataset. Perfect = every site of the K-wide
    target block occupied in frame 2."""
    per_ds = {}
    for ds in run_dss:
        m2 = ds.occ(SURV_FRAME, per_site)
        tab = {k: [0, 0] for k in range(1, kmax + 1)}
        for s, p in enumerate(ds.plans):
            if not (1 <= p.k <= kmax):
                continue
            tab[p.k][1] += 1
            tab[p.k][0] += bool(m2[s, p.t0 : p.t0 + p.k].all())
        per_ds[ds.name] = tab
    pooled = {k: [0, 0] for k in range(1, kmax + 1)}
    for tab in per_ds.values():
        for k, (a, b) in tab.items():
            pooled[k][0] += a
            pooled[k][1] += b
    return per_ds, pooled


def loss_distributions(run_dss, per_site=False):
    """P(lose m | K, M) plus the independent-loss model it should be compared against."""
    rows = []
    ks = ns = km = nm = 0
    for ds in run_dss:
        m2 = ds.occ(SURV_FRAME, per_site)
        for s, p in enumerate(ds.plans):
            blk = np.zeros(ds.cal.n, bool)
            if p.k:
                blk[p.t0 : p.t0 + p.k] = True
            if p.k == 0:
                continue
            surv = int(m2[s].sum())
            rows.append((p.k, p.n_moves, max(0, p.k - surv)))
            for a in range(p.k):
                alive = bool(m2[s, int(p.targets[a])])
                if p.dist[a] == 0:
                    ns += 1
                    ks += alive
                else:
                    nm += 1
                    km += alive
    rows = np.array(rows, int) if rows else np.zeros((0, 3), int)
    p_s = ks / ns if ns else np.nan
    p_m = km / nm if nm else np.nan

    def model(k, m):
        """Can be compressed to:
        def model(k, m, p_s, p_m):
            a = binom.pmf(np.arange(k - m + 1), k - m, 1 - p_s)
            b = binom.pmf(np.arange(m + 1), m, 1 - p_m)
        return np.convolve(a, b)
        """
        from math import comb

        a = np.array(  # say: binomial for success probability of move
            [
                comb(k - m, i) * (1 - p_s) ** i * p_s ** (k - m - i)
                for i in range(k - m + 1)
            ]
        )
        b = np.array(  # say: binomial for survival/loss when measurement
            [comb(m, i) * (1 - p_m) ** i * p_m ** (m - i) for i in range(m + 1)]
        )
        return np.convolve(a, b)

    cells = {}
    for k, m in sorted({(int(a), int(b)) for a, b, _ in rows}):
        sel = rows[(rows[:, 0] == k) & (rows[:, 1] == m)]
        obs = np.bincount(sel[:, 2], minlength=k + 1)[: k + 1].astype(float)
        exp = model(k, m) * len(sel)
        cells[(k, m)] = {
            "n": len(sel),
            "obs": obs,
            "exp": exp,
            "var_obs": float(sel[:, 2].var()) if len(sel) > 1 else np.nan,
            "var_exp": float((k - m) * p_s * (1 - p_s) + m * p_m * (1 - p_m)),
        }
    return {"rows": rows, "p_stat": p_s, "p_move": p_m, "cells": cells}


def loading_drift(cal_ds, run_dss, per_site=False):
    """Loading rate over shot index -- per dataset."""
    out = {}
    for ds in ([cal_ds] if cal_ds else []) + list(run_dss):
        m1 = ds.occ(LOAD_FRAME, per_site)
        out[ds.name] = m1.sum(1).astype(float) / ds.cal.n
    return out


def run_table(cal_ds, run_dss, per_site=False, kmax=10, k_range=None):
    """
    One row per run: shots, mean K, loading, stationary/moved survival, block-perfect
    rate, mover fate (target or still at source?), and the detected start_window"""
    rows = []
    for ds in run_dss:
        s = survival_tables(None, [ds], per_site, k_range)
        m2 = ds.occ(SURV_FRAME, per_site)
        perfect = tot = 0
        for i, p in enumerate(ds.plans):
            if p.k == 0 or (k_range and not (k_range[0] <= p.k <= k_range[1])):
                continue
            tot += 1
            perfect += bool(m2[i, p.t0 : p.t0 + p.k].all())
        k = np.array(
            [p.k for p in ds.plans if not k_range or k_range[0] <= p.k <= k_range[1]],
            float,
        )
        k = k if k.size else np.array([np.nan])
        rows.append(
            {
                "name": ds.name,
                "shots": ds.n_shots,
                "K": k.mean(),
                "loading": k.mean() / ds.cal.n,
                "stat_k": s["stat"]["k"],
                "stat_n": s["stat"]["n"],
                "move_k": s["move"]["k"],
                "move_n": s["move"]["n"],
                "perfect": perfect,
                "perfect_n": tot,
                "arrived": s["mover_fate"]["target"],
                "at_origin": s["mover_fate"]["origin"],
                "gone": s["mover_fate"]["gone"],
                "start_window": ds.start_window,
                "window_misfit": float(ds.window_misfit),
            }
        )
    if cal_ds is not None:
        s = survival_tables(cal_ds, [], per_site, k_range)
        rows.insert(
            0,
            {
                "name": "CALIBRATION (no moves)",
                "shots": cal_ds.n_shots,
                "K": np.mean([p.k for p in cal_ds.plans]),
                "loading": np.mean([p.k for p in cal_ds.plans]) / cal_ds.cal.n,
                "stat_k": s["cal"]["k"],
                "stat_n": s["cal"]["n"],
                "move_k": 0,
                "move_n": 0,
                "perfect": 0,
                "perfect_n": 0,
                "arrived": 0,
                "at_origin": 0,
                "gone": 0,
                "start_window": None,
                "window_misfit": float("nan"),
            },
        )
    return rows


# If there are datasets that do not have a sorter_calibration.npz locate the sites from
# reference images instead -- counts / threshold with same conventions --
# SingleAtomTransport for example and the correspond. dataset fall into this category.
# (use also nearest-site Voronoi mask, not sorter's ROI box).
def locate_sites(img, thr=100):
    """Locates atoms in a reference image by thresholding and center of mass."""
    im = img - img.min()
    im = im / im.max() * 255
    im = ndimage.gaussian_filter(im, 0.8)
    label, n = ndimage.label(im > thr)  # type: ignore
    pts = []
    for k in range(1, n + 1):
        if (label == k).sum() > 2:
            cy, cx = ndimage.center_of_mass(label == k)
            if 5 < cy < im.shape[1] - 5:  # type: ignore
                pts.append((cx, cy))
    return np.array(sorted(pts, key=lambda p: p[1]))


def derive_calibration(locate_img, r_pix, thr=100, mean_img=None, binning=2):
    """Calibration is built from locate_sites instead of a live npz, for the no-npz
    conventions (SingleAtomTransport, and any future hop-scan dataset like it)."""
    locs = locate_sites(locate_img, thr)
    active = np.zeros(locate_img.shape, bool)
    for m in site_masks(locate_img.shape, locs, r_pix):
        active |= m
    return Calibration(
        locs,
        active,
        float("nan"),
        len(locs),
        mean_img if mean_img is not None else locate_img,
        binning,
        r_pix,
        "derived-locate",
    )


def site_masks(shape, loc, r):
    """Per-site boolean masks: nearest-site Voronoi cell, capped at radius r."""
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    d = np.stack([(X - s[0]) ** 2 + (Y - s[1]) ** 2 for s in loc])
    own = d.argmin(0)
    return [(d[i] <= r * r) & (own == i) for i in range(len(loc))]


def roi_mask_counts(images, masks):
    return np.array([[im[m].sum() for im in images] for m in masks])


def transport_config_of(run_name):
    """Which SingleAtomTransport calibration session a run belongs to -- the 46um-spacing
    session (46um-tagged and Optimized* runs) or the default-spacing session (everything
    else).
    Hardcoded here simply for my analysis purposes - not applicable in a general sense.
    See hardcoded TRANSPORT_CONFIGS."""
    return "46um" if ("46um" in run_name or "Optimized" in run_name) else "default"


def load_transport_config(folder, config, min_shots=1):
    """One SingleAtomTransport calibration session -> (Calibration, masks, {run: (load_counts,
    surv_counts)}).

    The reference stack is matched by exact suffix so "46umMephisto" never picked by "Mephisto"
    session, or vice versa."""
    spec = TRANSPORT_CONFIGS[config]
    groups = discover_runs(folder, min_shots, verbose=False)
    ref_key = next(k for k in groups if k.endswith("-" + spec["reference"]))
    ref_raw = load_stack(folder, ref_key)
    ref_img = ref_raw[:, LOAD_FRAME].std(0) + ref_raw[:, SURV_FRAME].std(0)
    cal = derive_calibration(ref_img, spec["r_pix"], binning=2)
    masks = site_masks(ref_img.shape, cal.locations, spec["r_pix"])
    names = sorted(
        k
        for k in groups
        if k != ref_key
        and "Mephisto" not in k
        and "TiSaph" not in k
        and transport_config_of(k) == config
    )
    per_run = {}
    for name in names:
        raw = load_stack(folder, name)
        im = to_photons(raw, cal.binning)
        per_run[name] = (
            roi_mask_counts(im[:, LOAD_FRAME], masks),
            roi_mask_counts(im[:, SURV_FRAME], masks),
        )
    return cal, masks, per_run


def load_stack(folder, prefix, min_shots=1):
    """One run's raw image stack, in numeric shot order (discover_runs, not filename sort)."""
    groups = discover_runs(folder, min_shots, verbose=False)
    entries = groups.get(prefix)
    if entries is None:
        raise FileNotFoundError(f"no run matching prefix {prefix!r} in {folder}")
    return np.array(
        [np.load(p, allow_pickle=True)[()]["Images"] for _, _, p in entries]
    ).astype(np.int32)


def bimodal_pdf(k, A, m1, m2, s1, s2):
    """Probability density function:
    Two-Gaussian mixture used to fit a raw count histogram."""
    return (1 - A) * norm.pdf(k, m1, s1) + A * norm.pdf(k, m2, s2)


def bimodal_threshold(pool):
    """Threshold between two peaks of a bimodal count distribution, via curve_fit.
    Returns (threshold, fit_params).
    Applied when there is enough data and there is no need for Otsu's method.
    Fallback should be Otsu's method if fits fail or convergence yields nonsense."""
    bins = np.arange(pool.min(), pool.max() + 2, 2) - 0.5
    ent, edg = np.histogram(pool, bins=bins, density=True)
    ctr = 0.5 * (edg[1:] + edg[:-1])
    p, _ = curve_fit(
        bimodal_pdf,
        ctr,
        ent,
        p0=[0.5, pool.max() / 6, pool.max() * 0.7, 5, 12],
        maxfev=40000,
    )
    A, m1, m2, s1, s2 = p
    g = np.linspace(min(m1, m2), max(m1, m2), 4000)
    d = (1 - A) * norm.pdf(g, m1, s1) - A * norm.pdf(g, m2, s2)
    return float(g[np.where(np.diff(np.sign(d)))[0][0]]), p
