"""
optimize_phases.py
Find the tone phases that let a static multi-tone array carry the most RF power without
clipping the DAC. Standalone: no card, no Controller, no lab PC needed. Run it once for a
given frequency set + amplitude ratios, paste the printed phase array into your script.

    python optimize_phases.py

@author: Bjarne Schümann
"""

# =====================================================================================
# WHY MULTI-TONE ARRAYS LOSE POWER, AND WHAT THE CEILING IS
# =====================================================================================
#
# The DAC limits the PEAK voltage (it can only swing +-MAX_AMPV), but the delivered RF
# power is set by the RMS voltage. The ratio between the two is the crest factor:
#
#     CF = peak / RMS
#
# For K tones of amplitude a_k (fractions of full scale) at DISTINCT frequencies, the
# cross terms time-average to zero, so
#
#     P    = (V_fs^2 / (2R)) * sum_k a_k^2          <- power is the SUM OF SQUARES
#     RMS  = V_fs * sqrt(sum_k a_k^2 / 2)
#     peak = V_fs * max_t |sum_k a_k exp(i(2 pi f_k t + phi_k))|
#
# Scale the tones so the peak just reaches full scale (peak = V_fs). Then
#
#     sum_k a_k^2 = 2 / CF^2
#
# and that single number IS the deliverable power, in units of "one tone at a = 1".
#
# A single sine has the lowest crest factor any sinusoid can have, CF = sqrt(2), giving
# sum(a^2) = 1 -- it uses the full DAC swing efficiently. Eight tones under Schroeder
# phasing have CF = 1.894, so sum(a^2) = 2/1.894^2 = 0.5575, and you get 44 * 0.5575 =
# 24.5 mW (against 44 mW measured for one tone at a = 1, MAX_AMPV = 1.2 V).
#
# The physical picture: a single tone has a CONSTANT envelope, so it sits against the rail
# at all times. Sum tones at different frequencies and the envelope beats -- it wanders up
# and down. The rail is set by the rare high excursions while the power follows the
# average, so the DAC range spent on those peaks is range you never convert into power.
#
# K = 2 is the clean illustration: two tones always come into phase once per beat period,
# so peak = a1 + a2 = 2a no matter what phases you choose. Hence a = 0.5, sum(a^2) = 0.5,
# exactly half the single-tone power. No phase trick can help at K = 2.
#
# THE FLOOR. A perfectly flat envelope would give CF = sqrt(2) and sum(a^2) = 1, i.e. the
# full single-tone power. Exactly flat is impossible for K >= 3 at equal amplitudes:
# |A(t)|^2 = sum(a^2) + 2*sum_{j<k} a_j a_k cos((w_j - w_k) t + phi_j - phi_k), so
# flatness requires every difference-frequency term to vanish,
#
#     sum_k a_k * a_{k+m} * exp(i(phi_k - phi_{k+m})) = 0    for every lag m = 1..K-1
#
# which is 2(K-1) real equations in only K-1 free phase differences. Overdetermined, so
# you can only get close -- hence this optimizer.
#
# WHAT THIS BUYS. Schroeder (phi_n = pi*n(n-1)/K) is a convenient closed form, not the
# optimum. Numerically minimising the peak gets K = 8 from CF = 1.894 down to ~1.68,
# i.e. sum(a^2) = 0.5575 -> ~0.71, about 1.27x more power (24.5 -> 31 mW).
#
# CAVEAT. The result is optimal for the EXACT frequency set given here. It stays exact for
# a static hold array. During a sort move the tones chirp, the relative phases drift, and
# the crest factor wanders -- so do not rely on an optimised peak to justify amplitudes on
# a moving array unless you re-check the peak along the trajectory.
# =====================================================================================

import numpy as np
from scipy.optimize import minimize

# ============================ config ============================
# Either give FREQS_HZ explicitly, or leave it None and use the array geometry below.
FREQS_HZ = None

N = 9  # number of tones
F_START_HZ = 91.0e6  # frequency of site 0
SPACING_UM = 4.6  # site separation [um]
UM_PER_MHZ = 4.6 / 0.6

# Per-tone amplitude RATIOS (only the ratios matter). None -> equal weights.
WEIGHTS = None

HEADROOM = 1.0  # fraction of digital full scale the composite peak may reach
N_RESTARTS = 50  # random restarts; more = better optimum, slower
SEED = 0
# ===============================================================


def schroeder_generalized(M):
    """IMD-suppressing phase prescription: phi_n = pi*n(n-1)/M."""
    if M < 2:
        return np.zeros(max(M, 1))
    n = np.arange(M)
    return np.pi * n * (n - 1) / M


def _time_grid(freqs_hz, n):
    """One beat period of the envelope, which depends only on difference frequencies."""
    d = np.diff(np.unique(freqs_hz))
    return np.linspace(
        0.0, 1.0 / (d.min() if d.size else freqs_hz[0]), n, endpoint=False
    )


def peak_of_sum(w, freqs_hz, phases, n=8192):
    """Peak of sum_k w_k sin(2 pi f_k t + phi_k), in units of one tone at w=1.

    Uses the analytic envelope |sum_k w_k exp(i(...))|, which upper-bounds |signal| and is
    slowly varying, so it needs far fewer samples than the ~91 MHz carrier would.
    Identical to the function in hold_array.py -- keep the two in step.
    """
    w = np.asarray(w, dtype=float)
    f = np.asarray(freqs_hz, dtype=float)
    p = np.asarray(phases, dtype=float)
    t = _time_grid(f, n)
    env = (w[:, None] * np.exp(1j * (2 * np.pi * np.outer(f, t) + p[:, None]))).sum(0)
    return float(np.abs(env).max())


def _clip_project(w, B, phases, n_iter=300):
    """Iterative time-domain clip / frequency-domain projection (Gerchberg-Saxton style).

    Clip the envelope to its current mean magnitude, project back onto the fixed tone
    magnitudes keeping only the phases, repeat. Cheap, and lands close to the optimum;
    the Powell polish afterwards cleans up the rest.
    """
    M = B.shape[1]
    for _ in range(n_iter):
        a = (w * np.exp(1j * phases))[:, None] * B
        env = a.sum(0)
        mag = np.abs(env)
        target = mag.mean()
        env = env * np.minimum(1.0, target / np.maximum(mag, 1e-12))
        coef = (env[None, :] * B.conj()).sum(1) / M
        nz = np.abs(coef) > 1e-12
        phases = np.where(nz, np.angle(coef), phases)
    return phases % (2 * np.pi)


def optimize_phases(w, freqs_hz, n_restarts=N_RESTARTS, seed=SEED, n_grid=4096):
    """Minimise the composite peak over the tone phases. Returns (phases, peak)."""
    w = np.asarray(w, dtype=float)
    f = np.asarray(freqs_hz, dtype=float)
    K = len(w)
    if K < 3:
        # K=1 is trivially flat; K=2 always reaches a1+a2 whatever the phases.
        ph = np.zeros(K)
        return ph, peak_of_sum(w, f, ph)

    t = _time_grid(f, n_grid)
    B = np.exp(2j * np.pi * np.outer(f, t))

    def obj(p):
        return float(np.abs(((w * np.exp(1j * p))[:, None] * B).sum(0)).max())

    rng = np.random.default_rng(seed)
    best_p = schroeder_generalized(K)
    best = obj(best_p)
    for trial in range(n_restarts):
        p0 = best_p.copy() if trial == 0 else rng.uniform(0, 2 * np.pi, K)
        p0 = _clip_project(w, B, p0)
        r = minimize(obj, p0, method="Powell", options={"maxiter": 6000, "xtol": 1e-6})
        if float(r.fun) < best:
            best, best_p = float(r.fun), np.asarray(r.x, dtype=float) % (2 * np.pi)
    # re-evaluate on the fine grid used everywhere else, so the reported peak is the one
    # scale_amps() will compute
    return best_p, peak_of_sum(w, f, best_p)


def report(label, w, freqs_hz, phases):
    w = np.asarray(w, dtype=float)
    pk = peak_of_sum(w, freqs_hz, phases)
    amps = w * (HEADROOM / pk)
    s2 = float(np.sum(amps**2))
    cf = pk / np.sqrt(np.sum(w**2) / 2.0)
    print(
        f"  {label:22s} peak {pk:7.4f}   CF {cf:6.3f}   sum(a^2) {s2:7.4f}   "
        f"a_k {amps[0]:.4f}"
    )
    return s2


def main():
    if FREQS_HZ is not None:
        freqs = np.asarray(FREQS_HZ, dtype=float)
    else:
        freqs = F_START_HZ + np.arange(N) * (SPACING_UM / UM_PER_MHZ * 1e6)
    K = len(freqs)
    w = np.ones(K) if WEIGHTS is None else np.asarray(WEIGHTS, dtype=float)
    if w.shape != (K,):
        raise ValueError("WEIGHTS must have length %d (got %d)" % (K, w.size))

    print(
        f"\n{K} tones, {freqs[0] / 1e6:.4f} .. {freqs[-1] / 1e6:.4f} MHz, "
        f"HEADROOM = {HEADROOM}"
    )
    print(f"weights: {np.round(w, 4)}\n")

    ph_zero = np.zeros(K)
    ph_schroeder = schroeder_generalized(K)
    print("phase set               peak         CF        sum(a^2)      a_k")
    s2_zero = report("all zero", w, freqs, ph_zero)
    s2_schr = report("Schroeder", w, freqs, ph_schroeder)

    print(f"\noptimising over {K} phases ({N_RESTARTS} restarts)...")
    ph_opt, _ = optimize_phases(w, freqs)
    s2_opt = report("optimised", w, freqs, ph_opt)

    print(f"\ngain vs Schroeder: {s2_opt / s2_schr:.3f}x power")
    print(f"gain vs all-zero:  {s2_opt / s2_zero:.3f}x power")
    print(
        "\n(sum(a^2) is the deliverable power in units of one tone at a=1: multiply by"
    )
    print(" the power you measure for a single full-amplitude tone at your MAX_AMPV.)")

    print("\nPASTE THIS as phases_ch0 (radians, ascending frequency order):")
    print("PHASES = np.array([")
    for i in range(0, K, 4):
        print("    " + ", ".join(f"{v:.9f}" for v in ph_opt[i : i + 4]) + ",")
    print("])")
    print()


if __name__ == "__main__":
    main()
