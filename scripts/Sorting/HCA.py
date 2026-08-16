"""
Heuristic Cluster Algorithm (HCA) for atom-array sorting.
After Sheng et al., Phys. Rev. Research 3, 023008 (2021).

Every move carries one atom src -> dst along an explicit road, never a teleport.
The road is axis-aligned and only crosses empty traps; it can bend around
obstacles and, with allow_outside=True, loop through free space outside the array.

  moves[k] = ((sr,sc),(dr,dc))        endpoints
  paths[k] = [(sr,sc), ..., (dr,dc)]  the road; 'OUT' = an around-the-outside hop
  solve(validate=True) replays everything and checks no road crosses an atom.

allow_outside True permits exterior detours (physical for a steerable spot);
False threads only interior empty traps and relocates blockers instead.

Target window: corner is a corner name, an explicit (rs,cs), or "best"
(max overlap, ties broken by centrality then ring atoms). find_best_origin
instead tries every window and keeps the one needing the fewest moves.
"""

from collections import deque

NEI = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
OUT = -2          # the "outside the array" node. Must stay an invalid cell index:
                  # cells are 0..N*M-1, so 0 would alias (0,0) and corrupt routing.


# ============================================================================
#  SOLVER
# ============================================================================
class HCA:
    def __init__(self, grid, K, corner="top-left", allow_outside=True, incremental_dist=True):
        rows = list(grid)
        self.N, self.M, self.K = len(rows), len(rows[0]), int(K)
        if self.K > self.N or self.K > self.M:
            raise ValueError("K must be <= N and <= M")
        assert not (0 <= OUT < self.N * self.M), \
            "OUT must not collide with a real cell index — keep OUT = -2 (never 0)."
        self.g = bytearray(self.N * self.M)
        for r in range(self.N):
            base, row = r * self.M, rows[r]
            for c in range(self.M):
                self.g[base + c] = 1 if row[c] else 0
        self._g0 = bytes(self.g)
        self._build_target(corner)
        self._build_neighbours()
        self.moves, self.paths = [], []
        self.allow_outside = allow_outside
        self.incremental_dist = incremental_dist

    # ---- target / geometry --------------------------------------------
    def _build_target(self, corner):
        N, M, K = self.N, self.M, self.K
        if isinstance(corner, (tuple, list)):
            rs, cs = int(corner[0]), int(corner[1])
            if not (0 <= rs <= N - K and 0 <= cs <= M - K):
                raise ValueError("explicit origin out of range")
        elif corner == "best":
            rs, cs = self._best_origin_overlap()
        elif corner in CORNERS:
            rs = 0 if corner.startswith("top") else N - K
            cs = 0 if corner.endswith("left") else M - K
        else:
            raise ValueError("corner must be a CORNER name, 'best', or (rs, cs)")
        self.corner = corner
        self.target_origin = (rs, cs)
        self.target = bytearray(N * M)
        self.target_cells = []
        for r in range(rs, rs + K):
            base = r * M
            for c in range(cs, cs + K):
                self.target[base + c] = 1
                self.target_cells.append(base + c)

    def _best_origin_overlap(self):
        N, M, K, g = self.N, self.M, self.K, self.g
        P = [[0] * (M + 1) for _ in range(N + 1)]
        for r in range(N):
            rowsum, base = 0, r * M
            for c in range(M):
                rowsum += g[base + c]
                P[r + 1][c + 1] = P[r][c + 1] + rowsum
        ws = lambda r0, c0, r1, c1: P[r1][c1] - P[r0][c1] - P[r1][c0] + P[r0][c0]
        best_key, best = None, (0, 0)
        for rs in range(N - K + 1):
            for cs in range(M - K + 1):
                ov = ws(rs, cs, rs + K, cs + K)
                free = (rs > 0) + (rs + K < N) + (cs > 0) + (cs + K < M)
                r0, c0 = max(rs - 1, 0), max(cs - 1, 0)
                r1, c1 = min(rs + K + 1, N), min(cs + K + 1, M)
                ring = ws(r0, c0, r1, c1) - ov
                key = (ov, free, ring)
                if best_key is None or key > best_key:
                    best_key, best = key, (rs, cs)
        return best

    def _build_neighbours(self):
        N, M = self.N, self.M
        nb = [None] * (N * M)
        bnd = bytearray(N * M)
        bcells = []
        for r in range(N):
            for c in range(M):
                i:int = r * M + c
                lst:list = []
                if r > 0:     
                    lst.append(i - M)
                if r < N - 1: 
                    lst.append(i + M)
                if c > 0:     
                    lst.append(i - 1)
                if c < M - 1: 
                    lst.append(i + 1)
                nb[i] = lst # type: ignore
                if r == 0 or c == 0 or r == N - 1 or c == M - 1:
                    bnd[i] = 1
                    bcells.append(i)
        self.nb, self.boundary, self._boundary_cells = nb, bnd, bcells

    # ---- connectivity --------------------------------------------------
    def _dist_from_outside(self):
        """BFS distance through empty traps from any perimeter gap.
        Reachable empty -> >=1; occupied or sealed-off empty -> -1."""
        g, nb = self.g, self.nb
        dist = [-1] * len(g)
        dq = deque()
        for i in self._boundary_cells:
            if g[i] == 0:
                dist[i] = 1
                dq.append(i)
        while dq:
            i = dq.popleft()
            d = dist[i] + 1
            for j in nb[i]:
                if g[j] == 0 and dist[j] == -1:
                    dist[j] = d
                    dq.append(j)
        return dist

    def _dist_from_reservoir(self):
        """BFS road-distance through empty traps from the nearest reservoir atom.
        Empty cell reachable from a reservoir -> >=1 (1 == reservoir on a neighbour);
        boxed-in empty or occupied -> -1. Used to fill defects from the inside out:
        a defect far from (or walled off from) the reservoir is filled before an
        entrance-adjacent one, so an early fill never seals a deeper defect."""
        g, nb, tgt = self.g, self.nb, self.target
        dist = [-1] * len(g)
        dq = deque()
        for i in range(len(g)):
            if g[i] == 1 and not tgt[i]:                # reservoir atom
                for j in nb[i]:
                    if g[j] == 0 and dist[j] == -1:
                        dist[j] = 1
                        dq.append(j)
        while dq:
            i = dq.popleft()
            d = dist[i] + 1
            for j in nb[i]:
                if g[j] == 0 and dist[j] == -1:
                    dist[j] = d
                    dq.append(j)
        return dist

    @staticmethod
    def _fill_order_key(resd, dist, i):
        """Priority for filling defect `i`, larger == filled first. A defect with
        no reservoir road (boxed/surrounded, resd<=0) goes first, then the one whose
        nearest reservoir is farthest; ties broken by outside-depth, then index."""
        rd = resd[i]
        return (rd <= 0, rd if rd > 0 else 0, dist[i], i)

    def _empty_component(self, seed):
        g, nb = self.g, self.nb
        seen, dq, comp = {seed}, deque([seed]), [seed]
        while dq:
            i = dq.popleft()
            for j in nb[i]:
                if g[j] == 0 and j not in seen:
                    seen.add(j)
                    dq.append(j)
                    comp.append(j)
        return comp

    def _chain(self, prev, start):
        M = self.M
        seq, cur = [], start
        while cur is not None:
            seq.append("OUT" if cur == OUT else (cur // M, cur % M))
            cur = prev[cur]
        return seq

    def _route(self, src, dst, grid=None):
        """Shortest collision-free road src->dst; src is the pickup, every other
        step must be empty. Returns the path (cells, 'OUT' for exterior hops) or None."""
        g = self.g if grid is None else grid
        nb, bnd, ao = self.nb, self.boundary, self.allow_outside
        prev = {src: None}
        dq = deque([src])
        found = False
        while dq:
            i = dq.popleft()
            if i == dst:
                found = True
                break
            if i == OUT:
                for j in self._boundary_cells:
                    if j not in prev and (g[j] == 0 or j == dst):
                        prev[j] = OUT # type: ignore
                        dq.append(j)
                continue
            for j in nb[i]:
                if j not in prev and (g[j] == 0 or j == dst):
                    prev[j] = i
                    dq.append(j)
            if ao and bnd[i] and OUT not in prev:
                prev[OUT] = i
                dq.append(OUT)
        if not found:
            return None
        seq = self._chain(prev, dst)
        seq.reverse()
        return seq

    def _nearest_source_path(self, d):
        """Nearest reservoir atom that can reach defect d by a clean road.
        Returns (src_index, path[src..d]) or None."""
        g, nb, bnd, tgt, ao = self.g, self.nb, self.boundary, self.target, self.allow_outside
        prev = {d: None}
        dq = deque([d])
        while dq:
            i = dq.popleft()
            if i == OUT:
                for j in self._boundary_cells:
                    if j in prev:
                        continue
                    if g[j] == 1 and not tgt[j]:           # reservoir atom on perimeter
                        prev[j] = OUT # type: ignore
                        return j, self._chain(prev, j)
                    if g[j] == 0:
                        prev[j] = OUT  # type: ignore
                        dq.append(j)
                continue
            for j in nb[i]:                                 # source adjacent to this empty
                if g[j] == 1 and not tgt[j] and j not in prev:
                    prev[j] = i
                    return j, self._chain(prev, j)
            for j in nb[i]:
                if g[j] == 0 and j not in prev:
                    prev[j] = i
                    dq.append(j)
            if ao and bnd[i] and OUT not in prev:
                prev[OUT] = i
                dq.append(OUT)
        return None

    # ---- main loop -----------------------------------------------------
    def solve(self, allow_outside=None, store_paths=True, validate=False):
        if sum(self.g) < self.K * self.K:
            raise ValueError("Infeasible: %d atoms for %d target sites"
                             % (sum(self.g), self.K * self.K))
        if allow_outside is not None:        # None -> keep the value set in __init__
            self.allow_outside = allow_outside
        self._store = store_paths
        self.g = bytearray(self._g0)
        self.res_atoms = {i for i in range(len(self.g))
                          if self.g[i] == 1 and not self.target[i]}
        self.moves.clear()
        self.paths.clear()
        if self.incremental_dist:
            self._run_loop_fast()
        else:
            self._run_loop_default()
        if validate:
            self.validate()
        return self.moves

    def _run_loop_default(self):
        """Recompute distances once per defect handled. Straightforward, slower."""
        g = self.g
        guard, maxg = 0, 12 * len(g) + 100
        while True:
            guard += 1
            if guard > maxg:
                raise RuntimeError("HCA did not converge")
            dist = self._dist_from_outside()
            resd = self._dist_from_reservoir()
            deepest, dkey, closed = -1, None, -1
            for i in self.target_cells:
                if g[i] == 0:
                    if dist[i] > 0:
                        key = self._fill_order_key(resd, dist, i)
                        if dkey is None or key > dkey:
                            dkey, deepest = key, i
                    elif closed == -1:
                        closed = i
            if deepest == -1 and closed == -1:
                break
            if closed != -1:
                self._open_closed(closed)
            elif not self._fill_one(deepest):              # only fails if allow_outside=False
                self._open_toward_source(deepest)

    def _run_loop_fast(self):
        """Fill every reachable defect deepest-first under one distance snapshot,
        recomputing only after a crack. Moves are still routed live so the result
        stays valid, but the sequence (and sometimes the count) differs from the
        default loop. Main speed-up on large arrays."""
        g = self.g
        guard, maxg = 0, 12 * len(g) + 100
        while True:
            guard += 1
            if guard > maxg:
                raise RuntimeError("HCA did not converge")
            dist = self._dist_from_outside()
            resd = self._dist_from_reservoir()
            closed, reach = -1, []
            for i in self.target_cells:
                if g[i] == 0:
                    if dist[i] > 0:
                        reach.append(i)
                    elif closed == -1:
                        closed = i
            if not reach and closed == -1:
                break
            if closed != -1:                  # crack first, then recompute the snapshot
                self._open_closed(closed)
                continue
            reach.sort(key=lambda i: self._fill_order_key(resd, dist, i), reverse=True)
            for d in reach:
                if g[d] != 0:                 # an earlier fill already took this cell
                    continue
                if not self._fill_one(d):     # earlier fills sealed it, or allow_outside=False
                    self._open_toward_source(d)
                    break                     # topology changed; get a fresh snapshot

    # ---- primitive operations -----------------------------------------
    def _fill_one(self, d):
        res = self._nearest_source_path(d)
        if res is None:
            return False
        src, path = res
        self._do_move(src, d, path)
        return True

    def _open_closed(self, seed):
        """Crack a sealed pocket by relocating its wall atoms into it."""
        comp = self._empty_component(seed)
        dist = self._dist_from_outside()
        wall = self._crack(comp, lambda i: self.g[i] == 0 and dist[i] > 0)
        for w in wall:
            self._relocate(w)

    def _open_toward_source(self, d):
        """Fallback when allow_outside=False: crack a wall toward the nearest source."""
        comp = self._empty_component(d)
        g, nb, tgt = self.g, self.nb, self.target

        def goal(i):
            if g[i] != 0:
                return False
            return any(g[j] == 1 and not tgt[j] for j in nb[i])

        for w in self._crack(comp, goal):
            self._relocate(w)

    def _crack(self, comp, goal_fn):
        """0-1 BFS out of `comp`, cost = atoms crossed, stopping at the first cell
        satisfying goal_fn. Returns the occupied cells on that path."""
        g, nb = self.g, self.nb
        INF = 1 << 30
        cost = [INF] * len(g)
        par = [-1] * len(g)
        done = bytearray(len(g))
        comp_set = set(comp)
        dq = deque()
        for i in comp:
            cost[i] = 0
            dq.append(i)
        goal = -1
        while dq:
            i = dq.popleft()
            if done[i]:
                continue
            done[i] = 1
            if i not in comp_set and goal_fn(i):
                goal = i
                break
            ci = cost[i]
            for j in nb[i]:
                w = 1 if g[j] == 1 else 0
                nd = ci + w
                if nd < cost[j]:
                    cost[j] = nd
                    par[j] = i
                    (dq.appendleft if w == 0 else dq.append)(j)
        if goal == -1:
            raise RuntimeError("cannot open region (array too dense for this mode?)")
        path, cur = [], goal
        while cur != -1:
            path.append(cur)
            cur = par[cur]
        path.reverse()
        return [i for i in path if g[i] == 1]

    def _relocate(self, w):
        """Move obstacle `w` to the deepest target cell it can reach (fills a defect
        and frees `w`); failing that, park it in the nearest reservoir cell."""
        g, nb, tgt = self.g, self.nb, self.target
        dw = {w: 0}
        dq = deque([w])
        tcells, rcells = [], []
        while dq:
            i = dq.popleft()
            di = dw[i]
            for j in nb[i]:
                if g[j] == 0 and j not in dw:
                    dw[j] = di + 1
                    dq.append(j)
                    (tcells if tgt[j] else rcells).append(j)
        if tcells:
            dest = max(tcells, key=dw.__getitem__)
        elif rcells:
            dest = min(rcells, key=dw.__getitem__)
        else:
            raise RuntimeError("nowhere to relocate obstacle atom")
        self._do_move(w, dest)

    # ---- bookkeeping ---------------------------------------------------
    def _do_move(self, src, dst, path=None):
        g = self.g
        if g[src] != 1 or g[dst] != 0:
            raise RuntimeError("invalid move generated")
        if self._store and path is None:
            path = self._route(src, dst)
        g[src] = 0
        g[dst] = 1
        if not self.target[src]:
            self.res_atoms.discard(src)
        if not self.target[dst]:
            self.res_atoms.add(dst)
        M = self.M
        self.moves.append(((src // M, src % M), (dst // M, dst % M)))
        self.paths.append(path if self._store else None)

    # ---- utilities -----------------------------------------------------
    def grid(self):
        M = self.M
        return [[self.g[r * M + c] for c in range(M)] for r in range(self.N)]

    def replay_states(self):
        M = self.M
        g = bytearray(self._g0)
        snap = lambda: [[g[r * M + c] for c in range(M)] for r in range(self.N)]
        out = [snap()]
        for (sr, sc), (dr, dc) in self.moves:
            g[sr * M + sc] = 0
            g[dr * M + dc] = 1
            out.append(snap())
        return out

    def validate(self):
        """Replay every move and assert a clean road exists, the stored road crosses
        no occupied trap, and the target ends defect-free."""
        M = self.M
        g = bytearray(self._g0)
        for k, ((sr, sc), (dr, dc)) in enumerate(self.moves):
            s, d = sr * M + sc, dr * M + dc
            assert g[s] == 1 and g[d] == 0, "move %d endpoints inconsistent" % k
            assert self._route(s, d, grid=g) is not None, \
                "move %d: no collision-free road exists" % k
            p = self.paths[k] if k < len(self.paths) else None
            if p is not None:
                assert p[0] == (sr, sc) and p[-1] == (dr, dc), "stored road endpoints wrong"
                for cell in p[1:-1]:
                    if cell != "OUT":
                        rr, cc = cell
                        assert g[rr * M + cc] == 0, \
                            "move %d: stored road crosses an occupied trap %s" % (k, cell)
            g[s] = 0
            g[d] = 1
        for i in self.target_cells:
            assert g[i] == 1, "target left with a defect"
        return True

    def print_state(self):
        M = self.M
        for r in range(self.N):
            line = []
            for c in range(M):
                i = r * M + c
                ch = "#" if self.g[i] else "."
                line.append("[%s]" % ch if self.target[i] else " %s " % ch)
            print("".join(line))


# ============================================================================
#  CONVENIENCE + BEST PLACEMENT
# ============================================================================
def hca_sort(grid, K, corner="top-left", allow_outside=True, validate=False):
    h = HCA(grid, K, corner)
    h.solve(allow_outside=allow_outside, validate=validate)
    return h.moves


class HCA1D(HCA):
    """1D variant: take a 1xN row, count its K atoms, target a filled 1xK block.
    Only the target shape changes; the KxK builder is bypassed.

    corner = left/right (or a corner name) -> block flush to that edge
             "best" -> the window needing the fewest moves (every window solved and
                       compared, ties broken by more free edges). This is a routing
                       count, not overlap, so it's resolved at solve() time with the
                       allow_outside actually in use.
             int / (r, c) -> explicit start column."""

    def __init__(self, row, corner="best", **kw):
        row = list(row)
        K = sum(1 for x in row if x)
        super().__init__([row], 1, corner="top-left", **kw)   # K=1 just to pass __init__
        self.K = K
        self._row = row
        self.corner = corner
        self._best_pending = (corner == "best")
        self._set_target(0 if self._best_pending else self._origin_1d(corner, K))

    def _set_target(self, cs):
        if not (0 <= cs <= self.M - self.K):
            raise ValueError("origin out of range")
        self.target = bytearray(self.N * self.M)
        self.target_cells = list(range(cs, cs + self.K))
        for c in self.target_cells:
            self.target[c] = 1
        self.target_origin = (0, cs)

    def _origin_1d(self, corner, K):
        M = self.M
        if isinstance(corner, (tuple, list)):
            return int(corner[-1])
        if isinstance(corner, int):
            return corner
        if corner in CORNERS or corner in ("left", "right"):
            return 0 if corner.endswith("left") else M - K
        raise ValueError("corner must be left/right/CORNER name, 'best', or column")

    def _best_origin_1d(self, allow_outside):
        K, M, row = self.K, self.M, self._row
        best = None
        for cs in range(M - K + 1):
            h = HCA1D(row, cs, allow_outside=allow_outside, # type: ignore
                      incremental_dist=self.incremental_dist) # type: ignore
            try:
                h.solve(store_paths=False)
            except RuntimeError:               # this window isn't routable in this mode
                continue
            free = (cs > 0) + (cs + K < M)
            key = (len(h.moves), -free)
            if best is None or key < best[0]:
                best = (key, cs)
        if best is None:
            raise RuntimeError("no feasible target window (array too dense for this mode?)")
        return best[1]

    def solve(self, *a, **k):
        if self._best_pending:                 # now we know the real allow_outside
            ao = k.get("allow_outside", a[0] if a else None)
            self._set_target(self._best_origin_1d(self.allow_outside if ao is None else ao))
            self._best_pending = False
        realK, self.K = self.K, 1              # feasibility check uses K*K; want 1xK
        try:
            return super().solve(*a, **k)
        finally:
            self.K = realK


def find_best_origin(grid, K, mode="moves", allow_outside=True):
    if mode == "overlap":
        return HCA(grid, K, "best").target_origin
    rows = list(grid)
    N, M = len(rows), len(rows[0])
    best = None
    for rs in range(N - K + 1):
        for cs in range(M - K + 1): 
            h = HCA(grid, K, (rs, cs)) # type: ignore
            h.solve(allow_outside=allow_outside, store_paths=False)
            free = (rs > 0) + (rs + K < N) + (cs > 0) + (cs + K < M)
            key = (len(h.moves), -free)
            if best is None or key < best[0]:
                best = (key, (rs, cs))
    return best[1] # type: ignore


# ============================================================================
#  PLOTTING  (matplotlib only; solver above is dependency-free)
# ============================================================================
_PAL = dict(atom="#2b3a67", empty="#c8cdda", road="#e8543f", src="#e8543f",
            zone_face="#ffefd3", zone_edge="#e0a85a", bg="#fbfbfd")


def _draw_path(ax, path, N, M):
    """Draw the road: solid inside the array, dashed loop for 'OUT' hops."""
    def ext(rc):                                   # push a point just past its edge
        r, c = rc
        x, y = c, r
        if r == 0:        y = -0.9
        elif r == N - 1:  y = N - 0.1
        if c == 0:        x = -0.9
        elif c == M - 1:  x = M - 0.1
        return x, y

    solid = []
    k = 0
    while k < len(path):
        node = path[k]
        if node == "OUT":
            prev_r, next_r = path[k - 1], path[k + 1]
            if solid:
                xs, ys = zip(*solid)
                ax.plot(xs, ys, color=_PAL["road"], lw=2.0, zorder=3, solid_capstyle="round")
                solid = []
            ep, en = ext(prev_r), ext(next_r)
            ax.plot([prev_r[1], ep[0], en[0], next_r[1]],
                    [prev_r[0], ep[1], en[1], next_r[0]],
                    color=_PAL["road"], lw=1.7, ls=(0, (4, 3)), zorder=3)
            k += 1
            continue
        solid.append((node[1], node[0]))
        k += 1
    if solid:
        xs, ys = zip(*solid)
        ax.plot(xs, ys, color=_PAL["road"], lw=2.0, zorder=3, solid_capstyle="round")


def _draw_state(ax, g2d, origin, K, move=None, path=None, title=""):
    from matplotlib.patches import Circle, Rectangle
    N, M = len(g2d), len(g2d[0])
    rs, cs = origin
    ax.set_facecolor(_PAL["bg"])
    ax.add_patch(Rectangle((cs - 0.5, rs - 0.5), K, K, facecolor=_PAL["zone_face"],
                           edgecolor=_PAL["zone_edge"], lw=1.6, zorder=0))
    for r in range(N):
        for c in range(M):
            if g2d[r][c]:
                ax.add_patch(Circle((c, r), 0.33, facecolor=_PAL["atom"], edgecolor="none", zorder=2))
            else:
                ax.add_patch(Circle((c, r), 0.16, facecolor="none", edgecolor=_PAL["empty"], lw=1.1, zorder=1))
    if move is not None:
        (sr, sc), (dr, dc) = move
        if path:
            _draw_path(ax, path, N, M)
        else:
            ax.annotate("", xy=(dc, dr), xytext=(sc, sr),
                        arrowprops=dict(arrowstyle="-|>", color=_PAL["road"], lw=2.2,
                                        shrinkA=7, shrinkB=7), zorder=3)
        ax.add_patch(Circle((sc, sr), 0.30, facecolor="none", edgecolor=_PAL["src"], lw=1.8, zorder=4))
        ax.add_patch(Circle((dc, dr), 0.33, facecolor=_PAL["road"], edgecolor="none", zorder=5))
    ax.set_xlim(-1, M); ax.set_ylim(-1, N)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)


def plot_move_grid(sorter, include_initial=False, save=None, max_panels=49):
    import math, numpy as np, matplotlib.pyplot as plt  # noqa: E401
    states = sorter.replay_states()
    panels = []
    if include_initial:
        panels.append((states[0], None, None, "initial"))
    for k, mv in enumerate(sorter.moves, 1):
        p = sorter.paths[k - 1] if k - 1 < len(sorter.paths) else None
        panels.append((states[k], mv, p, "move %d" % k))
    if not panels:
        print("Target already defect-free — nothing to plot."); return None
    if len(panels) > max_panels:
        print("%d panels is a lot; use plot_slider(sorter)." % len(panels))
        panels = panels[:max_panels]
    n = len(panels)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (g2d, mv, p, title) in zip(axes, panels):
        _draw_state(ax, g2d, sorter.target_origin, sorter.K, mv, p, title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("HCA: %d moves, origin %s, K=%d, allow_outside=%s"
                 % (len(sorter.moves), sorter.target_origin, sorter.K, sorter.allow_outside),
                 fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()
    return fig


def plot_slider(sorter):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    states, moves, paths = sorter.replay_states(), sorter.moves, sorter.paths
    n = len(moves)
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    plt.subplots_adjust(bottom=0.16)

    def render(step):
        ax.clear()
        mv = moves[step - 1] if step > 0 else None
        p = paths[step - 1] if step > 0 and step - 1 < len(paths) else None
        title = "initial" if step == 0 else \
            "after move %d:  %s -> %s" % (step, moves[step - 1][0], moves[step - 1][1])
        _draw_state(ax, states[step], sorter.target_origin, sorter.K, mv, p, title)
        fig.canvas.draw_idle()

    sax = plt.axes((0.16, 0.05, 0.68, 0.04))
    slider = Slider(sax, "step", 0, max(n, 1), valinit=0, valstep=1)
    slider.on_changed(lambda v: render(int(v)))
    render(0)
    plt.show()
    return fig, slider


# ============================================================================
#  DEMO
# ============================================================================
if __name__ == "__main__":
    import random, time # noqa: E401

    def random_grid(N, M, p, seed):
        rng = random.Random(seed)
        return [[1 if rng.random() < p else 0 for _ in range(M)] for _ in range(N)]

    g, K = random_grid(10, 10, 0.55, seed=7), 5

    # solve two ways and SHOW the real roads
    s_out = HCA(g, K, "best"); s_out.solve(allow_outside=True, validate=True)
    s_in = HCA(g, K, "best"); s_in.solve(allow_outside=False, validate=True)

    print("origin:", s_out.target_origin)
    print("allow_outside=True  -> N_m = %d" % len(s_out.moves))
    print("allow_outside=False -> N_m = %d" % len(s_in.moves))
    print("\nexample roads (allow_outside=True), 'OUT' = looped around the array:")
    for mv, p in list(zip(s_out.moves, s_out.paths))[:4]:
        print("  %s -> %s   via %s" % (mv[0], mv[1], p))

    runs = 1000
    grids = [random_grid(10, 10, 0.55, seed=i) for i in range(runs)]
    t0 = time.perf_counter()
    for gg in grids:
        HCA(gg, 5, "best").solve(allow_outside=False)
    print("\n10x10/K=5 'best'+solve+paths: avg %.1f us (%d runs)"
          % ((time.perf_counter() - t0) / runs * 1e6, runs))

    try:
        # plot_move_grid(s_out, include_initial=True, save="hca_sequence.png")
        plot_slider(s_out)         # interactive: needs a GUI backend
    except Exception as e:
        print("plotting skipped (%s)" % e)