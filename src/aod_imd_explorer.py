#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
 AOD INTERMODULATION EXPLORER
=======================================================================

Analytic multi-tone intermodulation-distortion calculator for an
acousto-optic deflector / modulator with a (1 - cos) power response.

PHYSICS
-------
Bragg-regime AOD, single tone:

    eta(V) = sin^2( pi V / 2 V_pi ) = 1/2 [ 1 - cos( pi V / V_pi ) ]

The (1-cos) is the POWER response, therefore the FIELD response is its
square root, and the field is what superposes for multi-tone drive:

    E_diff / E_in  =  sqrt(eta)  =  sin( u(t) )

    u(t) = a cos(w1 t + p1) + b cos(w2 t + p2) + c cos(w3 t + p3)

    a_k = (pi/2) * sqrt(P_k / P_pi)      <-- RF power per tone

So the AOD is a memoryless sin() nonlinearity.  Taylor:

    sin u = u - u^3/6 + u^5/120 - ...    (odd powers only -> odd IMD only)

Exact solution via Jacobi-Anger (used by this code, no truncation in
amplitude, only in mixing order):

    sin u = 2 * sum_{N odd, half-space} (-1)^((N-1)/2)
                 J_m(a) J_n(b) J_p(c) cos( (m w1 + n w2 + p w3) t
                                          + m p1 + n p2 + p p3 )

    with N = m + n + p.

Because many (m,n,p) triplets are frequency-degenerate (especially for
equally spaced tones such as 70/80/90 MHz), the observed line amplitude
is a COHERENT sum of phasors -> tuning the tone phases redistributes
the distortion between spurs.  That is the knob this GUI exposes.

USAGE
-----
    python aod_imd_explorer.py            # launch interactive GUI
    python aod_imd_explorer.py --verify   # analytic vs brute-force FFT
    python aod_imd_explorer.py --png out.png   # static render, no GUI

Tip: untick "phase scan panel" for snappier slider dragging - that panel
re-evaluates the whole spectrum ~150 times per redraw.

Text boxes accept expressions, e.g.  pi/2,  0.9*sqrt(2),  360/7.

REQUIRES: numpy, scipy, matplotlib
=======================================================================
"""

import argparse
import sys

import numpy as np
from scipy.special import jv

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons


# =====================================================================
#  ANALYTIC ENGINE
# =====================================================================

class IMDEngine:
    """
    Exact Bessel-series evaluation of the spectrum of

        sin( a cos(w1 t + p1) + b cos(w2 t + p2) + c cos(w3 t + p3) )

    truncated at total mixing order |m|+|n|+|p| <= order_max.
    """

    def __init__(self, order_max=9):
        self.order_max = None
        self.set_order(order_max)

    # -- enumerate the surviving (m, n, p) triplets ---------------------
    def set_order(self, order_max):
        order_max = int(order_max)
        if order_max == self.order_max:
            return
        self.order_max = order_max
        R = order_max

        m_list, n_list, p_list = [], [], []
        for m in range(-R, R + 1):
            for n in range(-R + abs(m), R - abs(m) + 1):
                rem = R - abs(m) - abs(n)
                if rem < 0:
                    continue
                for p in range(-rem, rem + 1):
                    # only N = m+n+p odd survives Im{ exp(i u) }
                    # (note |m|+|n|+|p| has the same parity as m+n+p)
                    if (m + n + p) % 2 == 0:
                        continue
                    m_list.append(m)
                    n_list.append(n)
                    p_list.append(p)

        self.m = np.asarray(m_list, dtype=np.int64)
        self.n = np.asarray(n_list, dtype=np.int64)
        self.p = np.asarray(p_list, dtype=np.int64)

        N = self.m + self.n + self.p
        # (-1)^((N-1)/2) for odd N, valid for negative N as well
        self.sgn = np.round(np.sin(N * np.pi / 2.0))
        self.order = np.abs(self.m) + np.abs(self.n) + np.abs(self.p)
        self.ks = np.arange(-R, R + 1)

    # -- evaluate ------------------------------------------------------
    def spectrum(self, freqs, amps, phases, amp_floor=1e-11):
        """
        freqs  : (f1,f2,f3) in MHz
        amps   : (a,b,c)    dimensionless drive (pi/2 = single-tone saturation)
        phases : (p1,p2,p3) in radians

        returns f[MHz], A (field amplitude, coherent), order (lowest
        contributing mixing order), all sorted by frequency, f > 0 only.
        """
        R = self.order_max
        f1, f2, f3 = freqs
        a, b, c = amps
        q1, q2, q3 = phases

        # Bessel look-up tables J_k(x) for k = -R..R
        Ja = jv(self.ks, a)
        Jb = jv(self.ks, b)
        Jc = jv(self.ks, c)

        coef = self.sgn * Ja[self.m + R] * Jb[self.n + R] * Jc[self.p + R]

        f = self.m * f1 + self.n * f2 + self.p * f3
        psi = self.m * q1 + self.n * q2 + self.p * q3

        keep = (f > 1e-9) & (np.abs(coef) > amp_floor)
        if not np.any(keep):
            return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))

        fk = np.round(f[keep], 6)
        z = coef[keep] * np.exp(1j * psi[keep])
        ok = self.order[keep]

        uf, inv = np.unique(fk, return_inverse=True)
        re = np.bincount(inv, weights=z.real, minlength=uf.size)
        im = np.bincount(inv, weights=z.imag, minlength=uf.size)

        A = 2.0 * np.hypot(re, im)          # factor 2 = half-space folding

        minord = np.full(uf.size, 99, dtype=np.int64)
        np.minimum.at(minord, inv, ok)

        good = A > 1e-10
        return uf[good], A[good], minord[good]


# ---------------------------------------------------------------------
#  Low-order Taylor reference (u - u^3/6), for teaching / cross-check
# ---------------------------------------------------------------------

def taylor3_table(freqs, amps):
    """Analytic 3rd-order table exactly as derived by hand (amplitudes only)."""
    f1, f2, f3 = freqs
    a, b, c = amps
    rows = [
        ("f1",              f1, a - a**3 / 8 - a * (b**2 + c**2) / 4),
        ("f2",              f2, b - b**3 / 8 - b * (a**2 + c**2) / 4),
        ("f3",              f3, c - c**3 / 8 - c * (a**2 + b**2) / 4),
        ("2f1-f2",   2*f1 - f2, -a**2 * b / 8),
        ("2f2-f1",   2*f2 - f1, -a * b**2 / 8),
        ("2f2-f3",   2*f2 - f3, -b**2 * c / 8),
        ("2f3-f2",   2*f3 - f2, -b * c**2 / 8),
        ("2f1-f3",   2*f1 - f3, -a**2 * c / 8),
        ("2f3-f1",   2*f3 - f1, -a * c**2 / 8),
        ("f1+f2-f3", f1+f2-f3, -a * b * c / 4),
        ("f1+f3-f2", f1+f3-f2, -a * b * c / 4),
        ("f2+f3-f1", f2+f3-f1, -a * b * c / 4),
        ("3f1",          3*f1, -a**3 / 24),
        ("3f2",          3*f2, -b**3 / 24),
        ("3f3",          3*f3, -c**3 / 24),
        ("f1+f2+f3", f1+f2+f3, -a * b * c / 4),
    ]
    return rows


# ---------------------------------------------------------------------
#  Brute-force numerical reference (FFT of sin(u(t)))
# ---------------------------------------------------------------------

def numeric_spectrum(freqs, amps, phases, fmax=350.0, npts=1 << 20, cycles=4000):
    """Direct time-domain synthesis + FFT, for verification."""
    f1, f2, f3 = freqs
    fs = 8.0 * max(fmax, max(freqs) * 3)
    T = cycles / min(freqs)
    n = int(min(npts, 2 ** int(np.ceil(np.log2(fs * T)))))
    t = np.arange(n) / fs
    u = (amps[0] * np.cos(2 * np.pi * f1 * t + phases[0])
         + amps[1] * np.cos(2 * np.pi * f2 * t + phases[1])
         + amps[2] * np.cos(2 * np.pi * f3 * t + phases[2]))
    y = np.sin(u)
    from scipy.signal.windows import blackmanharris
    w = blackmanharris(n)
    Y = np.fft.rfft(y * w)
    fr = np.fft.rfftfreq(n, 1 / fs)
    # amplitude calibration for the window
    A = 2.0 * np.abs(Y) / np.sum(w)
    return fr, A


# =====================================================================
#  PSD RENDERING (spectrum-analyser look)
# =====================================================================

def render_psd(f_lines, a_lines, fmax, rbw, floor_db, npts=4000, seed=12345):
    """
    Convolve the analytic line spectrum with a Gaussian resolution
    bandwidth and add an exponentially-distributed (i.e. realistic
    Rayleigh-amplitude) noise floor.  Returns f grid and dB.
    """
    fgrid = np.linspace(0.0, fmax, npts)
    P = np.zeros_like(fgrid)

    sel = (f_lines <= fmax * 1.02) & (a_lines > 10 ** (floor_db / 20.0) * 1e-2)
    sigma = max(rbw, 1e-4) / 2.3548
    dfg = fgrid[1] - fgrid[0]
    half = max(int(np.ceil(5.0 * sigma / dfg)), 2)   # 5 sigma is plenty

    for f0, A in zip(f_lines[sel], a_lines[sel]):
        j = int(round(f0 / dfg))
        lo, hi = max(j - half, 0), min(j + half + 1, npts)
        if lo >= hi:
            continue
        P[lo:hi] += 0.5 * A**2 * np.exp(
            -0.5 * ((fgrid[lo:hi] - f0) / sigma) ** 2)

    rng = np.random.default_rng(seed)
    P = P + 10 ** (floor_db / 10.0) * rng.exponential(1.0, size=fgrid.size)
    return fgrid, 10.0 * np.log10(P + 1e-30)


ORDER_COLORS = {
    1: "#1f6fd0",   # carriers / order-1
    3: "#d62728",   # IMD3
    5: "#ff9f1c",   # IMD5
    7: "#2ca02c",   # IMD7
    9: "#7f4fc9",   # IMD9
}
ORDER_COLOR_HI = "#8a8a8a"


def order_color(o):
    return ORDER_COLORS.get(int(o), ORDER_COLOR_HI)


# =====================================================================
#  INTERACTIVE GUI
# =====================================================================

DEFAULTS = dict(
    f=[70.0, 80.0, 90.0],
    a=[0.90, 0.90, 0.90],
    ph=[0.0, 0.0, 0.0],          # degrees
    order=9,
    fmax=320.0,
    rbw=0.9,
    floor=-95.0,
)


class ExplorerGUI:

    def __init__(self):
        self.eng = IMDEngine(DEFAULTS["order"])
        self.f = list(DEFAULTS["f"])
        self.a = list(DEFAULTS["a"])
        self.ph = list(DEFAULTS["ph"])
        self.fmax = DEFAULTS["fmax"]
        self.rbw = DEFAULTS["rbw"]
        self.floor = DEFAULTS["floor"]
        self.order = DEFAULTS["order"]
        self.show_scan = True
        self.show_labels = True
        self.show_fft = False
        self._busy = False

        self._build_figure()
        self.update()

    # ---------------- figure & widgets -------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(16.5, 10.0))
        self.fig.canvas.manager.set_window_title(
            "AOD intermodulation explorer  -  sin(u) nonlinearity")

        self.ax = self.fig.add_axes([0.055, 0.505, 0.625, 0.445])
        self.ax_scan = self.fig.add_axes([0.745, 0.755, 0.235, 0.195])
        self.ax_tf = self.fig.add_axes([0.745, 0.505, 0.235, 0.165])

        self.fig.text(0.055, 0.965,
                      "Acousto-optic deflector intermodulation  —  "
                      r"$E_{diff}/E_{in}=\sin(u)$,  "
                      r"$\eta=\frac{1}{2}[1-\cos(2u)]$,  "
                      r"$u=\sum_k a_k\cos(\omega_k t+\varphi_k)$",
                      fontsize=12, weight="bold", va="bottom")

        # ---- control geometry
        col_x = [0.092, 0.340, 0.588]
        col_w = 0.132
        tb_w = 0.048
        row_y = [0.410, 0.362, 0.314]
        h = 0.026

        self.sl = {}
        self.tb = {}
        labels = [("f", "f%d  [MHz]", 0.1, 500.0),
                  ("a", "a%d  drive", 0.0, 6.2832),
                  ("ph", u"\u03c6%d  [deg]", -720.0, 720.0)]

        for i in range(3):
            for j, (key, lab, lo, hi) in enumerate(labels):
                ax_s = self.fig.add_axes([col_x[i], row_y[j], col_w, h])
                val = getattr(self, key)[i]
                s = Slider(ax_s, lab % (i + 1), lo, hi, valinit=val,
                           color=["#1f6fd0", "#d62728", "#2ca02c"][i])
                s.valtext.set_visible(False)
                ax_t = self.fig.add_axes(
                    [col_x[i] + col_w + 0.010, row_y[j], tb_w, h])
                t = TextBox(ax_t, "", initial=self._fmt(val))
                self.sl[(key, i)] = s
                self.tb[(key, i)] = t
                s.on_changed(self._mk_slider_cb(key, i))
                t.on_submit(self._mk_text_cb(key, i))

        # tone power readout
        self.txt_pow = self.fig.text(0.092, 0.298, "", fontsize=8.0,
                                     family="monospace", va="top")

        # ---- global controls (bottom band)
        gx, gw = 0.092, 0.132
        gy = [0.205, 0.157, 0.109, 0.061]

        def mkslider(y, label, lo, hi, val, fmt="%.1f", step=None, x=gx, w=gw):
            axs = self.fig.add_axes([x, y, w, h])
            s = Slider(axs, label, lo, hi, valinit=val, valstep=step,
                       color="#666666")
            s.valtext.set_fontsize(8)
            s.valtext.set_text(fmt % val)
            return s

        self.s_fmax = mkslider(gy[0], "span [MHz]", 20.0, 1500.0, self.fmax)
        self.s_rbw = mkslider(gy[1], "RBW [MHz]", 0.05, 8.0, self.rbw, "%.2f")
        self.s_floor = mkslider(gy[2], "noise floor [dB]", -160.0, -20.0,
                                self.floor)
        self.s_order = mkslider(gy[3], "max mix order", 1, 15, self.order,
                                "%d", step=2)

        for s in (self.s_fmax, self.s_rbw, self.s_floor, self.s_order):
            s.on_changed(self._global_cb)

        # ---- checkboxes / buttons
        ax_cb = self.fig.add_axes([0.300, 0.055, 0.145, 0.172])
        ax_cb.set_facecolor("#fafafa")
        self.cb = CheckButtons(
            ax_cb,
            ["label spurs", "phase scan panel", "overlay brute-force FFT"],
            [self.show_labels, self.show_scan, self.show_fft])
        for lab in self.cb.labels:
            lab.set_fontsize(8.5)
        self.cb.on_clicked(self._check_cb)

        ax_b1 = self.fig.add_axes([0.530, 0.200, 0.070, 0.036])
        self.b_reset = Button(ax_b1, "reset")
        self.b_reset.on_clicked(self._reset)

        ax_b2 = self.fig.add_axes([0.610, 0.200, 0.100, 0.036])
        self.b_opt = Button(ax_b2, u"optimise \u03c6\u2082,\u03c6\u2083")
        self.b_opt.on_clicked(self._optimise)

        ax_b3 = self.fig.add_axes([0.530, 0.152, 0.180, 0.036])
        self.b_equal = Button(ax_b3, "pre-compensate: flatten carriers")
        self.b_equal.on_clicked(self._equalise)

        ax_b4 = self.fig.add_axes([0.530, 0.104, 0.180, 0.036])
        self.b_print = Button(ax_b4, "print spur table to console")
        self.b_print.on_clicked(self._print_table)

        # info panel
        self.txt_info = self.fig.text(0.762, 0.300, "", fontsize=9,
                                      family="monospace", va="top")

    @staticmethod
    def _fmt(v):
        return ("%.4g" % v)

    # ---------------- callbacks --------------------------------------
    def _mk_slider_cb(self, key, i):
        def cb(val):
            if self._busy:
                return
            getattr(self, key)[i] = float(val)
            self._busy = True
            self.tb[(key, i)].set_val(self._fmt(val))
            self._busy = False
            self.update()
        return cb

    def _mk_text_cb(self, key, i):
        def cb(text):
            if self._busy:
                return
            try:
                v = float(eval(text, {"__builtins__": {}},
                               {"pi": np.pi, "np": np, "sqrt": np.sqrt}))
            except Exception:
                self.tb[(key, i)].set_val(self._fmt(getattr(self, key)[i]))
                return
            getattr(self, key)[i] = v
            s = self.sl[(key, i)]
            self._busy = True
            if s.valmin <= v <= s.valmax:
                s.set_val(v)          # keeps handle in sync
            self._busy = False
            self.update()
        return cb

    def _global_cb(self, _):
        self.fmax = float(self.s_fmax.val)
        self.rbw = float(self.s_rbw.val)
        self.floor = float(self.s_floor.val)
        o = int(self.s_order.val)
        if o % 2 == 0:
            o += 1
        self.order = o
        self.eng.set_order(self.order)
        self.update()

    def _check_cb(self, label):
        st = self.cb.get_status()
        self.show_labels, self.show_scan, self.show_fft = st
        self.update()

    def _reset(self, _):
        self._busy = True
        for i in range(3):
            for key, dk in (("f", "f"), ("a", "a"), ("ph", "ph")):
                v = DEFAULTS[dk][i]
                getattr(self, key)[i] = v
                self.sl[(key, i)].set_val(v)
                self.tb[(key, i)].set_val(self._fmt(v))
        self._busy = False
        self.update()

    def _set_phase(self, i, deg):
        self._busy = True
        self.ph[i] = deg
        if -720 <= deg <= 720:
            self.sl[("ph", i)].set_val(deg)
        self.tb[("ph", i)].set_val(self._fmt(deg))
        self._busy = False

    def _optimise(self, _):
        """Brute-force scan of phi2, phi3 minimising worst spur (dBc)."""
        best, bp = 1e9, (self.ph[1], self.ph[2])
        grid = np.arange(0.0, 360.0, 5.0)
        for p2 in grid:
            for p3 in grid:
                v = self._worst_spur(self.ph[0], p2, p3)
                if v < best:
                    best, bp = v, (p2, p3)
        self._set_phase(1, bp[0])
        self._set_phase(2, bp[1])
        self.update()
        print("optimised phases: phi2=%.1f deg, phi3=%.1f deg -> "
              "worst spur %.2f dBc" % (bp[0], bp[1], best))

    def _carrier_powers(self, amps=None):
        amps = self.a if amps is None else amps
        f, A, o = self.eng.spectrum(self.f, amps, self._phases_rad())
        out = []
        for fk in self.f:
            m = np.isclose(f, fk, atol=1e-6)
            out.append(float((0.5 * A[m] ** 2).sum()))
        return np.array(out)

    def _equalise(self, _):
        """
        Pre-compensate the drive amplitudes so that the three CARRIER
        optical powers come out equal, despite compression and the
        IMD3 products that land on top of the carriers.
        Fixed-point iteration on a_k.
        """
        from scipy.optimize import least_squares

        a0 = np.array(self.a, dtype=float)
        P0 = self._carrier_powers(a0)
        target = float(np.mean(P0))

        def resid(x):
            return (self._carrier_powers(np.clip(x, 1e-4, np.pi))
                    - target) / max(target, 1e-12)

        sol = least_squares(resid, a0, bounds=(1e-4, np.pi),
                            xtol=1e-12, ftol=1e-12, max_nfev=400)
        a = np.clip(sol.x, 1e-4, np.pi)
        P = self._carrier_powers(a)

        if np.ptp(P) >= np.ptp(P0):          # solver did not help -> keep old
            print("pre-compensation: no better solution found "
                  "(spread stays %.3e); drives unchanged." % np.ptp(P0))
            return

        self._busy = True
        for i in range(3):
            self.a[i] = float(a[i])
            self.sl[("a", i)].set_val(float(a[i]))
            self.tb[("a", i)].set_val(self._fmt(a[i]))
        self._busy = False
        self.update()
        print("pre-compensated drives: a = %s  ->  carrier powers %s "
              "(spread %.2e)" % (np.round(a, 5), np.round(P, 6), np.ptp(P)))

    # ---------------- analysis helpers -------------------------------
    def _phases_rad(self, p1=None, p2=None, p3=None):
        p = [self.ph[0] if p1 is None else p1,
             self.ph[1] if p2 is None else p2,
             self.ph[2] if p3 is None else p3]
        return np.deg2rad(p)

    def _worst_spur(self, p1, p2, p3):
        f, A, o = self.eng.spectrum(self.f, self.a, np.deg2rad([p1, p2, p3]))
        if f.size == 0:
            return 0.0
        carrier = np.zeros(f.size, dtype=bool)
        for fk in self.f:
            carrier |= np.isclose(f, fk, atol=1e-6)
        if not np.any(~carrier):
            return -999.0
        ref = A[carrier].max() if np.any(carrier) else A.max()
        return 20 * np.log10(A[~carrier].max() / max(ref, 1e-30))

    def _analyse(self, f, A):
        carrier = np.zeros(f.size, dtype=bool)
        for fk in self.f:
            carrier |= np.isclose(f, fk, atol=1e-6)
        P = 0.5 * A**2
        eta = P.sum()
        Pc = P[carrier].sum()
        Pi = P[~carrier].sum()
        ref = A[carrier].max() if np.any(carrier) else (A.max() if A.size else 1)
        worst = (20 * np.log10(A[~carrier].max() / max(ref, 1e-30))
                 if np.any(~carrier) else -np.inf)
        return carrier, eta, Pc, Pi, worst, ref

    # ---------------- main redraw ------------------------------------
    def update(self, *_):
        f, A, o = self.eng.spectrum(self.f, self.a, self._phases_rad())
        carrier, eta, Pc, Pi, worst, ref = self._analyse(f, A)

        # ---- main PSD
        self.ax.clear()
        fg, db = render_psd(f, A, self.fmax, self.rbw, self.floor)
        self.ax.plot(fg, db, lw=0.7, color="#9aa4ad", zorder=1)

        sel = f <= self.fmax
        if np.any(sel):
            pdb = 10 * np.log10(0.5 * A[sel] ** 2 + 1e-30)
            vis = pdb > self.floor - 3
            self.ax.vlines(f[sel][vis], self.floor - 12, pdb[vis],
                           colors=[order_color(x) for x in o[sel][vis]],
                           lw=1.6, zorder=3)
            self.ax.scatter(f[sel][vis], pdb[vis], s=16, zorder=4,
                            c=[order_color(x) for x in o[sel][vis]])

            if self.show_labels:
                idx = np.argsort(pdb[vis])[::-1][:16]
                for k in idx:
                    ff = f[sel][vis][k]
                    yy = pdb[vis][k]
                    self.ax.annotate("%.4g" % ff, (ff, yy),
                                     textcoords="offset points",
                                     xytext=(0, 6), ha="center", fontsize=7.2,
                                     color=order_color(o[sel][vis][k]))

        if self.show_fft:
            fr, An = numeric_spectrum(self.f, self.a, self._phases_rad(),
                                      fmax=self.fmax)
            m = fr <= self.fmax
            self.ax.plot(fr[m], 10 * np.log10(0.5 * An[m] ** 2 + 1e-30),
                         lw=0.6, color="k", alpha=0.45, zorder=2,
                         label="brute-force FFT")
            self.ax.legend(loc="upper right", fontsize=8)

        for fk, cc in zip(self.f, ["#1f6fd0", "#d62728", "#2ca02c"]):
            if fk <= self.fmax:
                self.ax.axvline(fk, color=cc, lw=0.8, ls=":", alpha=0.45)

        self.ax.set_xlim(0, self.fmax)
        self.ax.set_ylim(self.floor - 12, 6)
        self.ax.set_xlabel("RF frequency  [MHz]   "
                           r"($\rightarrow$ diffraction angle)")
        self.ax.set_ylabel("diffracted optical power  [dB of $P_{in}$]")
        self.ax.grid(alpha=0.25, lw=0.5)

        handles = [plt.Line2D([], [], color=order_color(k), lw=2,
                              label=("carrier / order 1" if k == 1
                                     else "IMD order %d" % k))
                   for k in (1, 3, 5, 7, 9)]
        self.ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                       ncol=2, framealpha=0.9)

        # ---- transfer function panel
        self.ax_tf.clear()
        x = np.linspace(0, max(3.4, max(self.a) * 1.15), 400)
        self.ax_tf.plot(x, np.sin(x) ** 2, color="#333333", lw=1.4,
                        label=r"$\eta=\sin^2 u=\frac{1}{2}(1-\cos 2u)$")
        self.ax_tf.plot(x, np.sin(x), color="#999999", lw=1.0, ls="--",
                        label=r"field $\sin u$")
        for ai, cc in zip(self.a, ["#1f6fd0", "#d62728", "#2ca02c"]):
            self.ax_tf.plot([ai], [np.sin(ai) ** 2], "o", ms=6, color=cc)
        self.ax_tf.axvline(np.pi / 2, color="#bbbbbb", lw=0.8, ls=":")
        self.ax_tf.set_title("response  (operating points)", fontsize=8.5)
        self.ax_tf.set_xlabel("drive $a$", fontsize=8)
        self.ax_tf.tick_params(labelsize=7)
        self.ax_tf.legend(fontsize=6.5, loc="lower right")
        self.ax_tf.grid(alpha=0.25, lw=0.5)

        # ---- phase-scan panel
        self.ax_scan.clear()
        if self.show_scan:
            grid = np.linspace(0, 360, 73)
            w2 = [self._worst_spur(self.ph[0], g, self.ph[2]) for g in grid]
            w3 = [self._worst_spur(self.ph[0], self.ph[1], g) for g in grid]
            self.ax_scan.plot(grid, w2, color="#d62728", lw=1.3,
                              label=r"sweep $\varphi_2$")
            self.ax_scan.plot(grid, w3, color="#2ca02c", lw=1.3,
                              label=r"sweep $\varphi_3$")
            self.ax_scan.axvline(np.mod(self.ph[1], 360), color="#d62728",
                                 ls=":", lw=0.9)
            self.ax_scan.axvline(np.mod(self.ph[2], 360), color="#2ca02c",
                                 ls=":", lw=0.9)
            self.ax_scan.set_xlabel("phase [deg]", fontsize=8)
            self.ax_scan.set_ylabel("worst spur [dBc]", fontsize=8)
            self.ax_scan.legend(fontsize=7)
        else:
            self.ax_scan.text(0.5, 0.5, "phase scan off", ha="center",
                              va="center", fontsize=9, color="#999999")
            self.ax_scan.set_xticks([])
            self.ax_scan.set_yticks([])
        self.ax_scan.set_title("worst spur vs. tone phase", fontsize=8.5)
        self.ax_scan.tick_params(labelsize=7)
        self.ax_scan.grid(alpha=0.25, lw=0.5)

        # ---- numeric readouts
        lines = ["tone   f [MHz]     a      P_RF/P_pi   P_opt [%]   "
                 "A/a (gain)   dB from mean"]
        Pk_all = []
        for i in range(3):
            m = np.isclose(f, self.f[i], atol=1e-6)
            Pk_all.append((0.5 * A[m] ** 2).sum())
        Pm = max(np.mean(Pk_all), 1e-30)
        for i in range(3):
            m = np.isclose(f, self.f[i], atol=1e-6)
            Ak = float(A[m].sum()) if np.any(m) else 0.0
            lines.append("  %d  %9.3f  %6.3f    %7.3f     %7.3f      %6.3f     "
                         "%+7.3f"
                         % (i + 1, self.f[i], self.a[i],
                            (self.a[i] / (np.pi / 2)) ** 2, 100 * Pk_all[i],
                            Ak / max(self.a[i], 1e-9),
                            10 * np.log10(max(Pk_all[i], 1e-30) / Pm)))
        self.txt_pow.set_text("\n".join(lines))

        self.txt_info.set_text(
            "total diffracted   : %6.2f %% of P_in\n"
            "  in the 3 carriers: %6.2f %%\n"
            "  in IMD spurs     : %6.2f %%   (%.1f dBc)\n"
            "worst single spur  : %6.2f dBc\n"
            "spectral lines     : %d  (order <= %d)\n"
            "\ntop spurs (dBc):\n%s"
            % (100 * eta, 100 * Pc, 100 * Pi,
               10 * np.log10(max(Pi, 1e-30) / max(0.5 * ref**2, 1e-30)),
               worst, f.size, self.order,
               self._top_spurs(f, A, o, carrier, ref)))

        self.fig.canvas.draw_idle()

    @staticmethod
    def _top_spurs(f, A, o, carrier, ref, n=6):
        idx = np.where(~carrier)[0]
        if idx.size == 0:
            return "  (none)"
        idx = idx[np.argsort(A[idx])[::-1][:n]]
        return "\n".join("  %9.4g MHz  ord %-2d  %7.2f"
                         % (f[k], o[k], 20 * np.log10(A[k] / max(ref, 1e-30)))
                         for k in idx)

    def _print_table(self, _):
        f, A, o = self.eng.spectrum(self.f, self.a, self._phases_rad())
        carrier, eta, Pc, Pi, worst, ref = self._analyse(f, A)
        idx = np.argsort(A)[::-1]
        print("\n%-12s %-8s %-14s %-10s %s"
              % ("f [MHz]", "order", "amplitude", "P [dB]", "dBc"))
        print("-" * 60)
        for k in idx[:45]:
            print("%-12.5g %-8d %-14.6e %-10.2f %.2f  %s"
                  % (f[k], o[k], A[k], 10 * np.log10(0.5 * A[k] ** 2 + 1e-30),
                     20 * np.log10(A[k] / max(ref, 1e-30)),
                     "<-- carrier" if carrier[k] else ""))
        print("-" * 60)
        print("eta_total=%.4f  carriers=%.4f  IMD=%.4f  worst spur=%.2f dBc"
              % (eta, Pc, Pi, worst))


# =====================================================================
#  VERIFICATION / CLI
# =====================================================================

def verify():
    from scipy.signal.windows import blackmanharris

    freqs = (70.0, 80.0, 90.0)
    amps = (0.9, 0.7, 1.1)
    phases = np.deg2rad([0.0, 40.0, 155.0])

    eng = IMDEngine(13)
    f, A, o = eng.spectrum(freqs, amps, phases)

    # --- brute force: synthesise sin(u(t)) and FFT it -----------------
    fs = 2048.0                      # MHz
    n = 1 << 20
    t = np.arange(n) / fs            # microseconds
    u = sum(amps[i] * np.cos(2 * np.pi * freqs[i] * t + phases[i])
            for i in range(3))
    y = np.sin(u)
    w = blackmanharris(n)
    Y = np.fft.rfft(y * w)
    fr = np.fft.rfftfreq(n, 1 / fs)
    Ew = np.sum(w ** 2)
    df = fs / n

    def fft_amp(f0, half=12):
        j = int(round(f0 / df))
        lo, hi = max(j - half, 0), min(j + half + 1, Y.size)
        return 2.0 * np.sqrt(np.sum(np.abs(Y[lo:hi]) ** 2) / (n * Ew))

    print("\n=== analytic (Bessel series) vs brute-force FFT of sin(u) ===")
    print("a=%.2f b=%.2f c=%.2f   phi=(0, 40, 155) deg\n" % amps)
    print("%-10s %-6s %-16s %-16s %s"
          % ("f[MHz]", "ord", "analytic", "FFT", "rel.err"))
    idx = np.argsort(A)[::-1][:24]
    for k in sorted(idx, key=lambda i: f[i]):
        an = fft_amp(f[k])
        print("%-10.5g %-6d %-16.8e %-16.8e %8.2e"
              % (f[k], o[k], A[k], an, abs(an - A[k]) / max(A[k], 1e-15)))

    # --- Taylor (u - u^3/6) vs exact, aggregated per frequency bin ----
    print("\n=== 3rd-order Taylor table vs exact Bessel  (weak drive) ===")
    print("a=0.25 b=0.20 c=0.30, all phases 0."
          "  Degenerate products are summed per bin.\n")
    amps_w = (0.25, 0.20, 0.30)
    fw, Aw, ow = IMDEngine(9).spectrum(freqs, amps_w, np.zeros(3))

    bins = {}
    for name, ff, val in taylor3_table(freqs, amps_w):
        e = bins.setdefault(round(ff, 6), [0.0, []])
        e[0] += val
        e[1].append(name)

    print("%-10s %-26s %-15s %-15s %s"
          % ("f[MHz]", "contributions", "taylor(u-u^3/6)", "exact", "rel.err"))
    for ff in sorted(bins):
        val, names = bins[ff]
        j = np.argmin(np.abs(fw - ff))
        exact = np.sign(val) * Aw[j] if abs(fw[j] - ff) < 1e-6 else np.nan
        print("%-10.5g %-26s %+.6e   %+.6e   %8.2e"
              % (ff, "+".join(names), val, exact,
                 abs(exact - val) / max(abs(val), 1e-15)))

    # --- Parseval: line powers vs mean diffracted efficiency ----------
    N = 1 << 22
    tt = np.arange(N) / N * 1.0      # 1 us = 10 periods of the 10 MHz base
    uu = sum(amps[i] * np.cos(2 * np.pi * freqs[i] * tt + phases[i])
             for i in range(3))
    print("\n=== energy conservation ===")
    print("sum of analytic line powers  = %.8f" % (0.5 * A ** 2).sum())
    print("time average of sin^2(u)     = %.8f" % np.mean(np.sin(uu) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check analytic result against FFT / Taylor")
    ap.add_argument("--png", metavar="FILE",
                    help="render the default view to a PNG and exit")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    if args.png:
        matplotlib.use("Agg")
        gui = ExplorerGUI()
        gui.fig.savefig(args.png, dpi=110)
        print("wrote", args.png)
        return

    gui = ExplorerGUI()
    plt.show()


if __name__ == "__main__":
    main()
