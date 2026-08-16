"""
Sort_Given_Mask.py
Sort a stochastically loaded NxM atom array into an arbitrary boolean Mask
(a smiley, a letter, any shape) instead of a filled KxK block.

The Mask is the target set. Given a measured occupancy (binary NxM) the moves
that fill every Mask site with the fewest single-atom relocations are computed
with the HCA solver (Sheng et al., PRR 3, 023008 (2021)); HCA already routes
collision-free roads and fills defects inside-out, which is what keeps both
scarce masks (isolated target sites) and dense masks (clustered target sites)
cheap. Only the target shape is generalised here - everything else is reused
from HCA.py.

Each move is one steerable tweezer carrying an atom src->dst along its road,
column -> X-AOD (CH0), row -> Y-AOD (CH1). build_sequence stitches the moves
into one moving-spot trajectory (x(t), y(t), amp(t)); the grid plots step
through the moves exactly like the 1D sorter / HCA.

@author: Bjarne Schümann
18.06.2026
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from HCA import HCA, _draw_path, _PAL


SPACING_UM         = 3.0
STEP_TIME_S        = 20e-6
WAYPOINTS_PER_STEP = 15
REPOSITION_TIME_S  = 1e-6     # dark hop between moves: power off -> jump freq -> power on
HOLD_S             = 2.5

SERIAL     = 24909
F_START_HZ = 91.0e6
MAX_AMPV   = 0.65
MAX_TOTAL  = 0.8
CHANNEL_X  = 0
CHANNEL_Y  = 1


# ============================================================================
#  SOLVER: HCA into an arbitrary mask
# ============================================================================
class MaskHCA(HCA):
    """HCA whose target is an arbitrary boolean NxM mask rather than a KxK block.
    occupancy and mask are NxM (0/1 or bool). solve() returns the move list."""

    def __init__(self, occupancy, mask, allow_outside=False, incremental_dist=True):
        rows = [list(r) for r in np.asarray(occupancy).astype(int)]
        super().__init__(rows, 1, corner="top-left",
                         allow_outside=allow_outside, incremental_dist=incremental_dist)
        self._set_mask(mask)

    def _set_mask(self, mask):
        mask = np.asarray(mask).astype(bool)
        if mask.shape != (self.N, self.M):
            raise ValueError("mask shape %s != grid shape %s"
                             % (mask.shape, (self.N, self.M)))
        self.mask = mask
        self.target = bytearray(self.N * self.M)
        self.target_cells = []
        for r in range(self.N):
            base = r * self.M
            for c in range(self.M):
                if mask[r, c]:
                    self.target[base + c] = 1
                    self.target_cells.append(base + c)
        self.n_target = len(self.target_cells)
        self.target_origin = (0, 0)

    def solve(self, allow_outside=None, store_paths=True, validate=False):
        n_atoms = sum(self._g0)
        if n_atoms < self.n_target:
            raise ValueError("infeasible: %d atoms for %d mask sites"
                             % (n_atoms, self.n_target))
        realK, self.K = self.K, 0          # bypass the K*K block feasibility check
        try:
            return super().solve(allow_outside=allow_outside,
                                  store_paths=store_paths, validate=validate)
        finally:
            self.K = realK

    def unused_traps(self):
        """Cells still holding an atom after the sort that are not mask sites - the
        leftover reservoir tones to switch off once the pattern is complete."""
        final = self.replay_states()[-1]
        return [(r, c) for r in range(self.N) for c in range(self.M)
                if final[r][c] and not self.mask[r, c]]


def solve_mask(occupancy, mask, allow_outside=False, validate=True):
    s = MaskHCA(occupancy, mask, allow_outside=allow_outside)
    s.solve(validate=validate)
    return s


def feasible(occupancy, mask):
    return int(np.asarray(occupancy).astype(bool).sum()) >= int(np.asarray(mask).astype(bool).sum())


def best_placement(occupancy, pattern, allow_outside=False):
    """Slide `pattern` over the array and keep the offset needing the fewest moves
    (the 2D analogue of the 1D find_best_window). O(offsets * solve); use for shapes
    whose absolute position is free, not for a centred smiley."""
    occ = np.asarray(occupancy).astype(bool)
    N, M = occ.shape
    ph, pw = pattern.shape
    if pattern.sum() > occ.sum():
        raise ValueError("infeasible: not enough atoms for the pattern")
    best = None
    for r0 in range(N - ph + 1):
        for c0 in range(M - pw + 1):
            try:
                s = MaskHCA(occ, place_pattern(pattern, N, M, (r0, c0)), allow_outside=allow_outside)
                s.solve(store_paths=False)
            except RuntimeError:
                continue
            free = (r0 > 0) + (r0 + ph < N) + (c0 > 0) + (c0 + pw < M)
            key = (len(s.moves), -free)
            if best is None or key < best[0]:
                best = (key, (r0, c0))
    if best is None:
        raise RuntimeError("no feasible placement for this pattern")
    return best[1], best[0][0]


# ============================================================================
#  MASK PATTERNS
# ============================================================================
_FONT = {
    " ": (".....", ".....", ".....", ".....", "....."),
    "A": (".###.", "#...#", "#####", "#...#", "#...#"),
    "B": ("####.", "#...#", "####.", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "###..", "#....", "#####"),
    "F": ("#####", "#....", "###..", "#....", "#...."),
    "G": (".####", "#....", "#..##", "#...#", ".####"),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "#####"),
    "J": ("..###", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "###..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "####.", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "####.", "#..#.", "#...#"),
    "S": (".####", "#....", ".###.", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "Y": ("#...#", ".#.#.", "..#..", "..#..", "..#.."),
    "Z": ("#####", "...#.", "..#..", ".#...", "#####"),
    "0": (".###.", "#..##", "#.#.#", "##..#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "..##.", ".#...", "#####"),
    "3": ("####.", "....#", ".###.", "....#", "####."),
    "4": ("#..#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "####."),
    "6": (".###.", "#....", "####.", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#..."),
    "8": (".###.", "#...#", ".###.", "#...#", ".###."),
    "9": (".###.", "#...#", ".####", "....#", ".###."),
}


def glyph(ch):
    rows = _FONT.get(ch.upper(), _FONT[" "])
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def render_text(text):
    parts = []
    for i, ch in enumerate(text):
        if i:
            parts.append(np.zeros((5, 1), dtype=bool))
        parts.append(glyph(ch))
    return np.hstack(parts) if parts else np.zeros((5, 0), dtype=bool)


def place_pattern(pattern, N, M, where="center"):
    pattern = np.asarray(pattern).astype(bool)
    ph, pw = pattern.shape
    if ph > N or pw > M:
        raise ValueError("pattern %dx%d does not fit in %dx%d" % (ph, pw, N, M))
    r0, c0 = ((N - ph) // 2, (M - pw) // 2) if where == "center" else where
    out = np.zeros((N, M), dtype=bool)
    out[r0:r0 + ph, c0:c0 + pw] = pattern
    return out


def text_mask(text, N, M, where="center"):
    return place_pattern(render_text(text), N, M, where)


_SMILEY8 = ("..####..",
            ".#....#.",
            "#.#..#.#",
            "#......#",
            "#.#..#.#",
            "#..##..#",
            ".#....#.",
            "..####..")


def smiley(N, M):
    if (N, M) == (8, 8):
        return np.array([[ch == "#" for ch in row] for row in _SMILEY8], dtype=bool)
    rr, cc = np.mgrid[0:N, 0:M]
    cr, cc0 = (N - 1) / 2.0, (M - 1) / 2.0
    R = min(N, M) / 2.0 - 0.5
    d = np.hypot(rr - cr, cc - cc0)
    m = np.abs(d - R) < 0.7                                       # face ring
    for s in (-1, 1):                                             # eyes
        m |= np.hypot(rr - (cr - 0.30 * R), cc - (cc0 + s * 0.35 * R)) < max(0.6, 0.12 * R)
    m |= (np.abs(d - 0.55 * R) < 0.7) & (rr > cr + 0.10 * R)      # smile
    return m & (d <= R + 0.3)


# ============================================================================
#  TRAJECTORIES  (column -> X / CH0, row -> Y / CH1)
# ============================================================================
def site_grid_positions(N, M, spacing_um=SPACING_UM, center=True):
    xs = np.arange(M) * float(spacing_um)
    ys = np.arange(N) * float(spacing_um)
    if center:
        xs = xs - xs.mean()
        ys = ys - ys.mean()
    return xs, ys


def min_jerk(p0, p1, n):
    s = np.linspace(0.0, 1.0, n)
    return p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def _exterior(r, c, xs, ys, N, M, pad):
    x, y = xs[c], ys[r]
    if c == 0:        x = xs[0] - pad
    elif c == M - 1:  x = xs[-1] + pad
    if r == 0:        y = ys[0] - pad
    elif r == N - 1:  y = ys[-1] + pad
    return x, y


def _path_waypoints(path, xs, ys, N, M, pad):
    pts = []
    for k, node in enumerate(path):
        if node == "OUT":
            pr, pc = path[k - 1]
            nr, nc = path[k + 1]
            pts.append(_exterior(pr, pc, xs, ys, N, M, pad))
            pts.append(_exterior(nr, nc, xs, ys, N, M, pad))
        else:
            r, c = node
            pts.append((xs[c], ys[r]))
    return pts


def _collapse_collinear(pts):
    """Drop interior waypoints on a straight run so a multi-site move in one
    direction becomes a single min-jerk segment, not n stop-and-go steps."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        d1x, d1y = bx - ax, by - ay
        d2x, d2y = cx - bx, cy - by
        cross = d1x * d2y - d1y * d2x
        dot = d1x * d2x + d1y * d2y
        if abs(cross) > 1e-9 or dot <= 0:        # turn (or reversal) -> keep the corner
            out.append(pts[i])
    out.append(pts[-1])
    return out


def build_move_xy(path, xs, ys, N, M, step_time_s=STEP_TIME_S,
                  wpps=WAYPOINTS_PER_STEP, spacing_um=SPACING_UM, t0=0.0):
    """One move's (t, x, y): min-jerk between turning points of the road, segment
    duration scaled by its length in site units (collinear runs are merged into one
    continuous go). 'OUT' hops detour one site beyond the edge."""
    pts = _collapse_collinear(_path_waypoints(path, xs, ys, N, M, pad=spacing_um))
    t = [t0]
    X = [pts[0][0]]
    Y = [pts[0][1]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        steps = max(1, int(round(np.hypot((x1 - x0) / spacing_um, (y1 - y0) / spacing_um))))
        n = max(2, steps * wpps)
        seg_t = np.linspace(t[-1], t[-1] + steps * step_time_s, n)[1:]
        t.extend(seg_t.tolist())
        X.extend(min_jerk(x0, x1, n)[1:].tolist())
        Y.extend(min_jerk(y0, y1, n)[1:].tolist())
    return np.asarray(t), np.asarray(X), np.asarray(Y)


def build_sequence(sorter, xs, ys, step_time_s=STEP_TIME_S, wpps=WAYPOINTS_PER_STEP,
                   spacing_um=SPACING_UM, reposition_time_s=REPOSITION_TIME_S, amp=1.0):
    """Stitch the moves into one moving-tweezer trajectory. Between moves the spot
    does a dark hop (~reposition_time_s): power off, step the frequency straight to
    the next pickup, power on - no slow min-jerk transit. Returns (t, X, Y, A)."""
    N, M = sorter.N, sorter.M
    t = np.zeros(0)
    X = np.zeros(0)
    Y = np.zeros(0)
    A = np.zeros(0)
    prev = None
    cur = 0.0
    for mv, p in zip(sorter.moves, sorter.paths):
        (sr, sc), (dr, dc) = mv
        if p is None:
            p = [(sr, sc), (dr, dc)]
        if prev is not None:
            tau = reposition_time_s / 3.0          # off (at dst) -> jump freq -> on (at src)
            t = np.concatenate([t, [cur + tau, cur + 2 * tau, cur + 3 * tau]])
            X = np.concatenate([X, [prev[0], xs[sc], xs[sc]]])
            Y = np.concatenate([Y, [prev[1], ys[sr], ys[sr]]])
            A = np.concatenate([A, [0.0, 0.0, amp]])
            cur = cur + 3 * tau
        mt, mx, my = build_move_xy(p, xs, ys, N, M, step_time_s, wpps, spacing_um, t0=cur)
        sl = slice(0 if t.size == 0 else 1, None)
        t = np.concatenate([t, mt[sl]])
        X = np.concatenate([X, mx[sl]])
        Y = np.concatenate([Y, my[sl]])
        A = np.concatenate([A, np.full(len(mt[sl]), amp)])
        cur = t[-1]
        prev = (mx[-1], my[-1])
    return t, X, Y, A


# ============================================================================
#  PLOTTING  (step through the moves, same idea as the 1D sorter / HCA)
# ============================================================================
def _draw_state_mask(ax, g2d, target_cells, M, move=None, path=None, title="", off_cells=None):
    from matplotlib.patches import Circle, Rectangle
    N, Mc = len(g2d), len(g2d[0])
    ax.set_facecolor(_PAL["bg"])
    for cell in target_cells:
        r, c = cell // M, cell % M
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=_PAL["zone_face"],
                               edgecolor=_PAL["zone_edge"], lw=1.0, zorder=0))
    for r in range(N):
        for c in range(Mc):
            if g2d[r][c]:
                ax.add_patch(Circle((c, r), 0.33, facecolor=_PAL["atom"], edgecolor="none", zorder=2))
            else:
                ax.add_patch(Circle((c, r), 0.16, facecolor="none", edgecolor=_PAL["empty"], lw=1.1, zorder=1))
    for r, c in (off_cells or []):                       # traps switched off this step
        ax.add_patch(Circle((c, r), 0.33, facecolor="none", edgecolor="#cf6679",
                            lw=1.6, ls=(0, (1, 1)), zorder=3))
        ax.plot([c - 0.18, c + 0.18], [r - 0.18, r + 0.18], color="#cf6679", lw=1.4, zorder=3)
        ax.plot([c - 0.18, c + 0.18], [r + 0.18, r - 0.18], color="#cf6679", lw=1.4, zorder=3)
    if move is not None:
        (sr, sc), (dr, dc) = move
        if path:
            _draw_path(ax, path, N, Mc)
        else:
            ax.annotate("", xy=(dc, dr), xytext=(sc, sr),
                        arrowprops=dict(arrowstyle="-|>", color=_PAL["road"], lw=2.2,
                                        shrinkA=7, shrinkB=7), zorder=3)
        ax.add_patch(Circle((sc, sr), 0.30, facecolor="none", edgecolor=_PAL["src"], lw=1.8, zorder=4))
        ax.add_patch(Circle((dc, dr), 0.33, facecolor=_PAL["road"], edgecolor="none", zorder=5))
    ax.set_xlim(-1, Mc)
    ax.set_ylim(-1, N)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)


def _off_state(states, off):
    g = [list(row) for row in states[-1]]
    for r, c in off:
        g[r][c] = 0
    return g


def plot_move_grid(sorter, include_initial=False, save=None, max_panels=49, turn_off=True):
    import math
    states = sorter.replay_states()
    panels = []
    if include_initial:
        panels.append((states[0], None, None, None, "initial"))
    for k, mv in enumerate(sorter.moves, 1):
        p = sorter.paths[k - 1] if k - 1 < len(sorter.paths) else None
        panels.append((states[k], mv, p, None, "move %d" % k))
    off = sorter.unused_traps() if turn_off else []
    if off:
        panels.append((_off_state(states, off), None, None, off, "turn off %d unused" % len(off)))
    if not panels:
        print("mask already complete - nothing to plot.")
        return None
    if len(panels) > max_panels:
        print("%d panels is a lot; use plot_slider(sorter)." % len(panels))
        panels = panels[:max_panels]
    n = len(panels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (g2d, mv, p, offc, title) in zip(axes, panels):
        _draw_state_mask(ax, g2d, sorter.target_cells, sorter.M, mv, p, title, off_cells=offc)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("mask sort: %d moves, %d sites, allow_outside=%s"
                 % (len(sorter.moves), sorter.n_target, sorter.allow_outside), fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()
    return fig


def plot_slider(sorter, turn_off=True):
    from matplotlib.widgets import Slider
    states, moves, paths = sorter.replay_states(), sorter.moves, sorter.paths
    n = len(moves)
    off = sorter.unused_traps() if turn_off else []
    off_step = n + 1 if off else None
    off_state = _off_state(states, off) if off else None
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    plt.subplots_adjust(bottom=0.14)

    def render(step):
        ax.clear()
        if off_step is not None and step == off_step:
            _draw_state_mask(ax, off_state, sorter.target_cells, sorter.M,
                             title="turn off %d unused trap(s)" % len(off), off_cells=off)
        else:
            mv = moves[step - 1] if step > 0 else None
            p = paths[step - 1] if step > 0 and step - 1 < len(paths) else None
            title = "initial" if step == 0 else \
                "after move %d:  %s -> %s" % (step, moves[step - 1][0], moves[step - 1][1])
            _draw_state_mask(ax, states[step], sorter.target_cells, sorter.M, mv, p, title)
        fig.canvas.draw_idle()

    sax = plt.axes((0.16, 0.04, 0.68, 0.04))
    slider = Slider(sax, "step", 0, max(off_step or n, 1), valinit=0, valstep=1)
    slider.on_changed(lambda v: render(int(v)))
    render(0)
    plt.show()
    return fig, slider


def plot_sequence(t, X, Y, A, xs, ys, n_moves):
    fig, (axp, axa) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6.0))
    for v in np.concatenate([xs, ys]):
        axp.axhline(v, color="#ededed", lw=0.6, zorder=0)
    axp.plot(t * 1e3, X, lw=1.6, label="x  (CH0)", zorder=3)
    axp.plot(t * 1e3, Y, lw=1.6, label="y  (CH1)", zorder=3)
    axa.plot(t * 1e3, A, lw=1.6, color="#888888")
    axp.set_ylabel("position [$\\mu$m]")
    axp.legend(loc="upper right")
    axa.set_ylabel("tweezer amp")
    axa.set_xlabel("time [ms]")
    axp.set_title("mask sort: %d moves, %.3f ms transport, step=%.0f us/site"
                  % (n_moves, t[-1] * 1e3 if t.size else 0.0, STEP_TIME_S * 1e6))
    fig.tight_layout()
    plt.show()
    return fig


# ============================================================================
#  OFFLINE TEST + STATS
# ============================================================================
def _print_grid(g, mask):
    for r in range(g.shape[0]):
        line = []
        for c in range(g.shape[1]):
            ch = "#" if g[r, c] else "."
            line.append("[%s]" % ch if mask[r, c] else " %s " % ch)
        print("".join(line))


def emulate(N=8, M=8, pattern="smiley", fill_prob=0.7, seed=0,
            allow_outside=False, show=True):
    rng = np.random.default_rng(seed)
    occ = rng.random((N, M)) < fill_prob
    mask = smiley(N, M) if pattern == "smiley" else text_mask(pattern, N, M)
    n_t, n_a = int(mask.sum()), int(occ.sum())
    print("array %dx%d, atoms=%d, mask sites=%d" % (N, M, n_a, n_t))
    print("loaded (#=atom, [.]=mask site):")
    _print_grid(occ, mask)
    if n_a < n_t:
        print("infeasible: not enough atoms loaded for this mask.")
        return None

    s = MaskHCA(occ, mask, allow_outside=allow_outside)
    t0 = time.perf_counter()
    s.solve(validate=True)
    dt = time.perf_counter() - t0
    print("HCA: %d moves, solved+validated in %.1f us" % (len(s.moves), dt * 1e6))
    for k, (mv, p) in enumerate(zip(s.moves, s.paths), 1):
        print("  move %2d:  %s -> %s   via %s" % (k, mv[0], mv[1], p))
    off = s.unused_traps()
    print("after sort: %d unused trap(s) to switch off" % len(off))
    if not s.moves and not off:
        print("mask already complete - nothing to do.")
        return s

    if show:
        xs, ys = site_grid_positions(N, M, SPACING_UM)
        plot_slider(s)
        plot_move_grid(s, include_initial=True)
        if s.moves:
            plot_sequence(*build_sequence(s, xs, ys), xs, ys, len(s.moves))
    return s


def monte_carlo(N=12, M=16, pattern="smiley", fill_prob=0.65, trials=300,
                seed0=0, allow_outside=False):
    mask = smiley(N, M) if pattern == "smiley" else text_mask(pattern, N, M)
    n_t = int(mask.sum())
    moves, us, feas, failed = [], [], 0, 0
    for sd in range(seed0, seed0 + trials):
        occ = np.random.default_rng(sd).random((N, M)) < fill_prob
        if occ.sum() < n_t:
            continue
        feas += 1
        s = MaskHCA(occ, mask, allow_outside=allow_outside)
        t0 = time.perf_counter()
        try:
            s.solve()
        except (RuntimeError, ValueError):       # HCA density limit: too few empties to maneuver
            failed += 1
            continue
        us.append((time.perf_counter() - t0) * 1e6)
        moves.append(len(s.moves))
    if not moves:
        print("no feasible loads at p=%.2f for %d mask sites." % (fill_prob, n_t))
        return
    moves, us = np.array(moves), np.array(us)
    print("MC %dx%d '%s' (%d sites), p=%.2f, ao=%s: feasible %d/%d, solved %d (%d crowded-outs)"
          % (N, M, pattern, n_t, fill_prob, allow_outside, feas, trials, len(moves), failed))
    print("  <moves>=%.2f (min %d, max %d),  solve %.1f us avg / %.1f us max"
          % (moves.mean(), moves.min(), moves.max(), us.mean(), us.max()))


# ============================================================================
#  LAB HOOK (template - dual-channel sync is setup-specific)
# ============================================================================
def _acquire_mask(N, M):
    print("enter %d rows of %d bits (blank line to quit):" % (N, M))
    rows = []
    for r in range(N):
        raw = input("row %d: " % r).strip()
        if raw == "":
            return None
        bits = [int(ch) for ch in raw if ch in "01"]
        if len(bits) != M:
            print("  expected %d bits, got %d" % (M, len(bits)))
            return _acquire_mask(N, M)
        rows.append(bits)
    return np.array(rows, dtype=int)


def run_in_the_lab_idea(mask, acquire=None, allow_outside=False):
    """Template lab loop: image -> occupancy, solve into `mask`, execute moves
    one steerable tweezer at a time (x on CH0, y on CH1). The synchronous
    dual-channel trigger for a single moving spot is setup-specific; wire the two
    move() calls to fire together on your trigger before running on hardware."""
    from Controller import AWGController            # AWGController repo must be importable

    N, M = mask.shape
    xs, ys = site_grid_positions(N, M, SPACING_UM)
    ctrl = AWGController(serial_number=SERIAL, f_start_hz=F_START_HZ,
                         max_channel_amp_v=MAX_AMPV, max_total_amplitude=MAX_TOTAL,
                         core_mapping="16/5", realtime_priority=True)
    ctrl.connect()
    try:
        while True:
            occ = acquire(N, M) if acquire else _acquire_mask(N, M)
            if occ is None:
                break
            if not feasible(occ, mask):
                print("  not enough atoms for the mask - re-arming.")
                continue
            try:
                s = MaskHCA(occ, mask, allow_outside=allow_outside)
                s.solve(validate=True)
            except (RuntimeError, ValueError) as e:
                print("  solve failed (%s) - re-arming." % e)
                continue
            off = s.unused_traps()
            print("  %d atoms, %d moves, %d traps off after" % (int(np.sum(occ)), len(s.moves), len(off)))
            for mv, p in zip(s.moves, s.paths):
                mt, mx, my = build_move_xy(p, xs, ys, N, M)
                ctrl.move(mt, mx[np.newaxis, :], channel=CHANNEL_X,
                          amplitudes=MAX_TOTAL, force_trigger=True)
                ctrl.move(mt, my[np.newaxis, :], channel=CHANNEL_Y,
                          amplitudes=MAX_TOTAL, force_trigger=True)
                time.sleep(mt[-1] + 1e-3)
                ctrl.stop()
            # `off` are the leftover reservoir sites to extinguish so only the mask
            # stays lit. How you drop them is holding-hardware specific (per-site SLM
            # tone off, or re-program_static the kept sites); wire it to your setup.
            _ = off
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    # monte_carlo(N=8, M=8, pattern="smiley", fill_prob=0.7)
    # monte_carlo(N=8, M=8, pattern="A", fill_prob=0.6)
    # emulate(N=8, M=8, pattern="smiley", fill_prob=0.7, seed=3)
    emulate(N=8, M=8, pattern="H", fill_prob=0.5, seed=1)
    emulate(N=8, M=8, pattern="A", fill_prob=0.5, seed=1)
    emulate(N=8, M=8, pattern="N", fill_prob=0.5, seed=1)
