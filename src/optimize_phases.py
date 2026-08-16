"""
optimize_phases.py
Crest-factor optimisation of the initial phases of an equally spaced multitone array.

Objective is exactly sorter.peak_of_sum(): the maximum of the analytic envelope
|sum_k w_k exp(i(2 pi f_k t + phi_k))| over one beat period, in units of one tone at w=1.

Key fact that makes one table enough:  for an equally spaced set f_k = f0 + k*df,
factoring out exp(i 2 pi f0 t) and substituting tau = df*t leaves
   |sum_k w_k exp(i(2 pi k tau + phi_k))|,
whose maximum over tau in [0,1) depends on {w_k} and {phi_k} ONLY.  It is independent
of f0 and of df.  So a phase set optimised here is valid for any start frequency and any
site spacing -- including after snapping to the DDS grid -- as long as the tones stay
equally spaced and the amplitude RATIOS are unchanged.

Optimisation: the true max is non-smooth, so minimise a p-norm surrogate with p ramped
80 -> 400 (softmax homotopy), many random restarts, then report the exact grid maximum.
"""

import numpy as np
from scipy.optimize import minimize

NGRID_OPT = 2048     # tau samples during optimisation
NGRID_EVAL = 131072  # tau samples for the final honest peak


def _env(phi, w, tau):
    """|sum_k w_k exp(i(2 pi k tau + phi_k))| on the reduced (f0-, df-free) grid."""
    k = np.arange(len(w))
    return np.abs((w[:, None] * np.exp(1j * (2 * np.pi * np.outer(k, tau) + phi[:, None]))).sum(0))


def peak(phi, w, n=NGRID_EVAL):
    tau = np.arange(n) / n
    return float(_env(np.asarray(phi, float), w, tau).max())


def _soft(phi, w, tau, p):
    e = _env(phi, w, tau)
    m = e.max()
    return m * (np.mean((e / m) ** p)) ** (1.0 / p)


def optimise(n, w=None, restarts=400, seed=0):
    w = np.ones(n) if w is None else np.asarray(w, float)
    tau = np.arange(NGRID_OPT) / NGRID_OPT
    rng = np.random.default_rng(seed)

    # seeds: Schroeder, Newman, and random
    seeds = [np.pi * np.arange(n) * (np.arange(n) - 1) / n,
             np.pi * np.arange(n) ** 2 / n]
    seeds += [rng.uniform(0, 2 * np.pi, n) for _ in range(restarts)]

    best_phi, best_pk = None, np.inf
    for s in seeds:
        phi = s.copy()
        for p in (80.0, 200.0, 400.0):
            r = minimize(_soft, phi, args=(w, tau, p), method="L-BFGS-B",
                         options=dict(maxiter=600, ftol=1e-14, gtol=1e-12))
            phi = r.x
        pk = peak(phi, w)
        if pk < best_pk:
            best_pk, best_phi = pk, phi
    phi = np.mod(best_phi - best_phi[0], 2 * np.pi)  # gauge-fix: phi_0 = 0
    return phi, peak(phi, w)


def report(n, w=None, **kw):
    w = np.ones(n) if w is None else np.asarray(w, float)
    phi, pk = optimise(n, w, **kw)
    sch = np.pi * np.arange(n) * (np.arange(n) - 1) / n
    pk_s = peak(sch, w)
    rms = np.sqrt(np.sum(w ** 2) / 2)          # RMS of the real summed signal
    a_opt, a_sch = 0.95 / pk, 0.95 / pk_s      # per-tone amp at STATIC_HEADROOM=0.95
    return dict(n=n, phi=phi, peak=pk, peak_sch=pk_s,
                cf=pk / rms, cf_sch=pk_s / rms, floor=np.sqrt(np.sum(w ** 2)),
                pwr_gain=(a_opt / a_sch) ** 2)


if __name__ == "__main__":
    rows = [report(n, restarts=40, seed=n) for n in range(2, 13)]
    print(f"{'n':>3} {'peak_opt':>9} {'peak_schr':>10} {'floor':>7} "
          f"{'CF_opt':>7} {'CF_schr':>8} {'a_k@0.95':>9} {'sum a^2':>8} {'gain':>6}")
    for r in rows:
        a = 0.95 / r["peak"]
        print(f"{r['n']:>3} {r['peak']:>9.4f} {r['peak_sch']:>10.4f} {r['floor']:>7.4f} "
              f"{r['cf']:>7.4f} {r['cf_sch']:>8.4f} {a:>9.4f} "
              f"{r['n'] * a ** 2:>8.4f} {r['pwr_gain']:>5.2f}x")

    print("\n\nOPTIMIZED_STATIC_PHASES = {")
    for r in rows:
        print(f"    {r['n']}: np.array([")
        for i in range(0, r["n"], 4):
            print("        " + ", ".join(f"{v:.9f}" for v in r["phi"][i:i + 4]) + ",")
        print(f"    ]),  # peak {r['peak']:.4f}  CF {r['cf']:.4f}")
    print("}")
