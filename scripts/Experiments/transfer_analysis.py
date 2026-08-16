import glob, json
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm

DATA = "tweezerImages"   # <- point at the folder holding the .npy shots
SIZE, BIN = 10, 2
LOAD, SURV = 1, 2
CAL = "tweezerLoad1x2-Mephisto"
ORDER = [("Linear500us", 500, "lin"), ("STA500us", 500, "sta"),
         ("STA400us", 400, "sta"), ("Linear400us", 400, "lin"),
         ("Linear300us", 300, "lin"), ("STA300us", 300, "sta"),
         ("STA250us", 250, "sta"), ("Linear250us", 250, "lin"),
         ("Linear200us", 200, "lin"), ("STA200us", 200, "sta"),
         ("STA200usNew", 200, "sta"), ("STA162_5us", 162.5, "sta"),
         ("Linear162_5us", 162.5, "lin")]


def files(p):
    return sorted(glob.glob(f"{DATA}/tweezerLoad1x2-{p}_*.npy"),
                  key=lambda f: (f.rsplit("_", 1)[1]))


def stack(p):
    fs = files(p)
    return (np.array([np.load(f, allow_pickle=True)[()]["Images"] for f in fs]).astype(np.int32),
            [f.rsplit("_", 1)[1][:-4] for f in fs])


def ph(d):
    return (d - 200 * BIN ** 2) * 0.1


def g2(xs, a, xo, yo, sx, sy, th, off):
    x, y = xs
    A = np.cos(th) ** 2 / (2 * sx ** 2) + np.sin(th) ** 2 / (2 * sy ** 2)
    B = -np.sin(2 * th) / (4 * sx ** 2) + np.sin(2 * th) / (4 * sy ** 2)
    C = np.sin(th) ** 2 / (2 * sx ** 2) + np.cos(th) ** 2 / (2 * sy ** 2)
    return (off + a * np.exp(-(A * (x - xo) ** 2 + 2 * B * (x - xo) * (y - yo)
                               + C * (y - yo) ** 2))).ravel()


def locate(img, thr=100):
    im = img - img.min(); im = im / im.max() * 255
    im = ndimage.gaussian_filter(im, 0.8)
    lab, n = ndimage.label(im > thr)
    pts = []
    for k in range(1, n + 1):
        if (lab == k).sum() > 2:
            cy, cx = ndimage.center_of_mass(lab == k)
            if 5 < cy < im.shape[1] - 5:
                pts.append((cx, cy))
    return np.array(pts)


def build_mask(img, loc):
    x = np.arange(img.shape[0]); y = np.arange(img.shape[1])
    X, Y = np.meshgrid(x, y)
    m = np.zeros(img.shape, bool); ps = []
    for s in loc:
        ly, lx = int(s[0] - SIZE / 2), int(s[1] - SIZE / 2)
        sub = img[lx:lx + SIZE, ly:ly + SIZE]
        p0 = (sub.max() - np.median(img), s[1], s[0], 3, 3, 0, np.median(img))
        xf, yf = np.meshgrid(x[lx:lx + SIZE], y[ly:ly + SIZE])
        p, _ = curve_fit(g2, (xf, yf), sub.ravel(order="F"), p0=p0, maxfev=5000)
        m |= g2((X, Y), *p).reshape(img.shape, order="F") > p[0] * 0.3 + p[-1]
        ps.append(p)
    return np.logical_not(m), ps


def cnt(arr, mask, loc):
    out = np.zeros([len(loc), len(arr)])
    for j, im in enumerate(np.copy(arr)):
        im[mask] = 0
        for i, b in enumerate(loc):
            bx, by = int(b[0] - SIZE / 2), int(b[1] - SIZE / 2)
            out[i, j] = im[by:by + SIZE + 1, bx:bx + SIZE + 1].sum()
    return out


def wil(k, n, z=1.0):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, h


cal, cal_ts = stack("Mephisto")
ref = cal[:, LOAD].std(0) + cal[:, SURV].std(0)
loc = locate(ref)
mask, ps = build_mask(ref, loc)
print("spots (x,y):", np.round(loc, 2).tolist(),
      " sep %.2f px" % np.hypot(*(loc[1] - loc[0])))
print("gaussian widths sx,sy [px]:", [(round(abs(p[3]), 2), round(abs(p[4]), 2)) for p in ps])

data = {}
for name, T, kind in ORDER:
    s, ts = stack(name)
    data[name] = (cnt(ph(s[:, LOAD]), mask, loc), cnt(ph(s[:, SURV]), mask, loc), ts)
cL, cS = cnt(ph(cal[:, LOAD]), mask, loc), cnt(ph(cal[:, SURV]), mask, loc)

sm = [np.mean([data[n][0][i].mean() for n, _, _ in ORDER]) for i in range(len(loc))]
ini, fin = int(np.argmax(sm)), int(np.argmin(sm))
print("mean loading counts per site:", np.round(sm, 1), "-> ini", ini, "fin", fin)

pool = np.concatenate([data[n][0][ini] for n, _, _ in ORDER])
bins = np.arange(pool.min(), pool.max() + 2, 2) - 0.5
ent, edg = np.histogram(pool, bins=bins, density=True)
ctr = 0.5 * (edg[1:] + edg[:-1])


def bim(k, A, m1, m2, s1, s2):
    return (1 - A) * norm.pdf(k, m1, s1) + A * norm.pdf(k, m2, s2)


fp, _ = curve_fit(bim, ctr, ent, p0=[.5, pool.max() / 6, pool.max() * .7, 5, 12], maxfev=30000)
A, m1, m2, s1, s2 = fp
grid = np.linspace(min(m1, m2), max(m1, m2), 4000)
d = (1 - A) * norm.pdf(grid, m1, s1) - A * norm.pdf(grid, m2, s2)
cross = grid[np.where(np.diff(np.sign(d)))[0][0]]
print(f"pooled histogram: empty {min(m1,m2):.1f}+-{s1:.1f}  filled {max(m1,m2):.1f}+-{s2:.1f}"
      f"  crossing {cross:.1f}  filled fraction {A:.2f}")

for THR in (45.0, round(float(cross), 1)):
    print(f"\n================ threshold {THR} photons ================")
    hdr = (f"{'run':<16}{'T[us]':>7}{'v[um/ms]':>9}{'N':>5}{'load':>7}"
           f"{'arrived':>9}{'+-':>6}{'source':>8}{'lost':>7}{'t_start':>10}")
    print(hdr); print("-" * len(hdr))
    out = []
    for name, T, kind in ORDER:
        L, S, ts = data[name]
        lo = L[ini] > THR
        af, ai = S[fin] > THR, S[ini] > THR
        n, nl = L.shape[1], int(lo.sum())
        arr = int((lo & af).sum()); src = int((lo & ai & ~af).sum())
        lost = nl - arr - src
        pl, _ = wil(nl, n); pa, ea = wil(arr, nl)
        pu, _ = wil(src, nl); px, _ = wil(lost, nl)
        print(f"{name:<16}{T:>7.1f}{32.25/T*1000:>9.1f}{n:>5}{pl:>7.3f}"
              f"{pa:>9.3f}{ea:>6.3f}{pu:>8.3f}{px:>7.3f}{ts[0][9:13]:>10}")
        out.append(dict(run=name, T=T, kind=kind, n=n, nl=nl, p_load=pl,
                        arrived=arr, p_arr=pa, e_arr=ea, src=src, p_src=pu,
                        lost=lost, p_lost=px, t0=ts[0]))
    lo = cL[ini] > THR
    keep = int((lo & (cS[ini] > THR)).sum()); nl = int(lo.sum())
    p, e = wil(keep, nl)
    print(f"{'REF Mephisto':<16}{'-':>7}{'-':>9}{cL.shape[1]:>5}{nl/cL.shape[1]:>7.3f}"
          f"{p:>9.3f}{e:>6.3f}{'(hold+image survival at source)':>20}")
    if THR == 45.0:
        out_rows, ref_survival = out, p
        json.dump(out, open("ftp_rows.json", "w"), indent=1)


# ------------------------------------------------------------------ figure
def figure(rows, ref=0.939, out="transfer_first_pass.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

    for kind, c, m, lbl in (("lin", "tab:blue", "o", "linear"),
                            ("sta", "tab:red", "s", "min-jerk (STA)")):
        xs = sorted([x for x in rows if x["kind"] == kind], key=lambda x: x["T"])
        a.errorbar([x["T"] for x in xs], [x["p_arr"] for x in xs],
                   yerr=[x["e_arr"] for x in xs], marker=m, color=c,
                   capsize=3, ls="-", label=lbl)
    a.axhline(ref, ls="--", c="k", lw=1)
    a.text(505, ref + 0.012, "hold + imaging reference", fontsize=8, ha="right")
    a.set_xlabel(r"transport duration $T$ [$\mu$s]")
    a.set_ylabel("P(arrived at target | loaded)")
    a.set_ylim(-0.02, 1.02); a.legend(); a.grid(alpha=0.3)

    xs = sorted(rows, key=lambda x: (-x["T"], x["kind"]))
    idx = np.arange(len(xs))
    bar = [x["p_arr"] for x in xs]
    src = [x["p_src"] for x in xs]
    lost = [x["p_lost"] for x in xs]
    b.bar(idx, bar, 0.7, label="arrived",
          color=["tab:blue" if x["kind"] == "lin" else "tab:red" for x in xs])
    b.bar(idx, src, 0.7, bottom=bar, color="orange", label="at source")
    b.bar(idx, lost, 0.7, bottom=np.array(bar) + np.array(src),
          color="grey", label="lost")
    b.set_xticks(idx)
    b.set_xticklabels([x["run"] for x in xs], rotation=90, fontsize=7)
    b.set_ylabel("fraction of loaded atoms"); b.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)
    return fig


figure(out_rows, ref=ref_survival)
