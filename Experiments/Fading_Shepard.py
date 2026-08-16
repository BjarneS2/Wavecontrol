"""
Time-dependent multi-tone AOD waveform generator.

Implements, following Lu et al., arXiv:2510.11451 (3D-AODL, fading-Shepard):

  A) Generalized Schroeder phases  phi_n = pi*n*(n-1)/(M-1)   [Eq. 4]
     - static prescription valid at all times for COMMON-MODE chirps
       (all tones on one AOD share the same trajectory chirp).

  B) Fading-Shepard waveforms [Eqs. 2-3]: a ladder of tones spaced df,
     chirped in common mode; tones fade in/out at the band edges with
     cos^p envelopes, with the pair condition p_A + p_B = 1 so that
     (old tweezer power) + (new tweezer power) = const through the
     A->B diffraction chain.

  C) Crossfaded discrete frequency hops (quantum-diamond-microscope
     scanning): single-AOD power-preserving crossfade (p = 1:
     cos/sin field envelopes), phase-continuous, raised-cosine timing.

All waveforms accumulate phase by integration: theta(t) = 2*pi*cumsum(f)/fs,
so arbitrary precomputed frequency/amplitude programs are supported.

Dependencies: numpy
"""

import numpy as np

# ======================================================================
# Phase prescriptions
# ======================================================================

def schroeder_generalized(M):
    """phi_n = 2*pi*n(n-1)/(2M) = pi*n(n-1)/M, n = 0..M-1  (Lu et al. Eq. 4).
    Quadratic in n  ->  nearest-neighbor difference phases advance as
    2*pi*n/M: the M in-band tones' IM2 phasors at spacing df form the M-th
    roots of unity and cancel; IM3 phasors form Gauss sums (sqrt suppression)."""
    if M < 2:
        return np.zeros(max(M, 1))
    n = np.arange(M)
    return np.pi * n * (n - 1) / M

# ======================================================================
# Core synthesis: phase-continuous, arbitrary f(t) and A(t) per tone
# ======================================================================

def synthesize_tones(freq_traj, amp_traj, phi0, fs):
    """
    freq_traj: (n_tones, n_samples) instantaneous frequency per tone [Hz]
    amp_traj : (n_tones, n_samples) field-amplitude envelope per tone
    phi0     : (n_tones,) initial (design) phases [rad]
    Returns the summed waveform, with per-tone phase
        theta_n(t) = 2*pi * cumsum(f_n)/fs + phi0_n      (Eq. 2)
    """
    dphase = 2 * np.pi * freq_traj / fs
    theta = np.cumsum(dphase, axis=1) + phi0[:, None]
    return np.sum(amp_traj * np.cos(theta), axis=0)


def render_segments(freqs_kt, amps_kt, phases, time_arr, fs, window=None):
    """
    Reconstruct the time-domain RF signal the DDS would emit, tone by tone, from the
    per-segment frequency/amplitude waypoints (freqs_kt, amps_kt = (K, T), time_arr = (T,)).
    Each tone is a piecewise-linear chirp with continuous phase across segment boundaries,
    starting from initial phase phases[k]. Renders only over `window`=(t0, t1) [s] (default:
    first 10 us) so the sample count stays tractable at RF rates.
    Returns (t, waves) with waves shape (K, len(t)); waves.sum(axis=0) is the real output.
    """
    freqs_kt = np.atleast_2d(np.asarray(freqs_kt, dtype=float))
    amps_kt  = np.atleast_2d(np.asarray(amps_kt,  dtype=float))
    phases   = np.atleast_1d(np.asarray(phases,   dtype=float))
    K, T = freqs_kt.shape
    tstart = float(time_arr[0])
    if window is None:
        t0, t1 = tstart, min(tstart + 10e-6, float(time_arr[-1]))
    else:
        t0 = max(float(window[0]), tstart)
        t1 = min(float(window[1]), float(time_arr[-1]))
    n = max(1, int(round((t1 - t0) * fs)))
    t = t0 + np.arange(n) / fs
    waves = np.zeros((K, n))
    seg_phase = phases.copy()
    for i in range(T - 1):
        ta, tb = float(time_arr[i]), float(time_arr[i + 1])
        dt = tb - ta
        mask = (t >= ta) & (t < tb)
        if np.any(mask):
            tau = t[mask] - ta
            for k in range(K):
                f_a, f_b = freqs_kt[k, i], freqs_kt[k, i + 1]
                a_a, a_b = amps_kt[k, i],  amps_kt[k, i + 1]
                phi = seg_phase[k] + 2 * np.pi * (f_a * tau + 0.5 * (f_b - f_a) * tau**2 / dt)
                waves[k, mask] = (a_a + (a_b - a_a) * tau / dt) * np.sin(phi)
        for k in range(K):
            f_a, f_b = freqs_kt[k, i], freqs_kt[k, i + 1]
            seg_phase[k] += 2 * np.pi * 0.5 * (f_a + f_b) * dt
    return t, waves


def crest_factor(w):
    """Crest factor = peak / RMS of a waveform. Lower is better: less coherent spiking,
    so more headroom before clipping and less intermodulation-distortion power."""
    w = np.asarray(w, dtype=float)
    rms = np.sqrt(np.mean(w**2))
    return float(np.max(np.abs(w)) / rms) if rms > 0 else float("inf")

# ======================================================================
# B) Fading-Shepard waveform for one AOD of a counter-propagating pair
# ======================================================================

def fading_envelope(fz_over_df, M, p, eta=0.5):
    """Eq. (3): amplitude vs the tone's instantaneous ladder coordinate
    x = f_Z^(n)/df (frequency offset from band center in units of df).
    Full amplitude for |x| <= (M-eta)/2, zero for |x| >= (M+eta)/2,
    cos^p crossfade in between."""
    x = np.abs(fz_over_df)
    A = np.zeros_like(x)
    inside = x <= (M - eta) / 2
    fading = (~inside) & (x < (M + eta) / 2)
    A[inside] = 1.0
    arg = (np.pi / (2 * eta)) * (x[fading] - M / 2) + np.pi / 4
    A[fading] = np.cos(arg) ** p if p > 0 else 1.0   # p=0: rectangular gate
    return A

def fading_shepard_channel(t, f0, df, M, p, xi,
                           f_lat,            # common lateral term  [Hz](t)
                           f_chirp_int,      # common Z-chirp term  [Hz](t)
                           n_ladder=None, eta=0.5):
    """
    Build (freq_traj, amp_traj, phi0) for ONE AOD channel (Eq. 2).
      f_n(t) = f0 + f_lat(t) + f_chirp_int(t) + (n + xi)*df
      A_n(t) = fading_envelope( (f_chirp_int(t) + (n+xi)*df) / df )
    The ladder index range must cover every rung that ever enters the
    band during the trajectory.
    """
    if n_ladder is None:
        span = (np.max(f_chirp_int) - np.min(f_chirp_int)) / df
        half = int(np.ceil(M / 2 + span / 2 + 2))
        n_ladder = np.arange(-half, half + 1)
    n = n_ladder[:, None]
    fz = f_chirp_int[None, :] + (n + xi) * df            # f_Z^(n)(t)
    freq = f0 + f_lat[None, :] + fz
    amp = fading_envelope(fz / df, M, p, eta)
    # Generalized Schroeder on the ladder index (quadratic sequences
    # extend naturally to newly fading-in rungs):
    phi0 = np.pi * n_ladder * (n_ladder - 1) / max(M, 1)
    return freq, amp, phi0

def transport_pair(t, X, Z, f0, df, M, fs,
                   lamF_v, lamF2_v2, pA=1.0, pB=0.0, xi=0.0):
    """
    Waveforms for a counter-propagating AOD pair (Ax, Bx) realizing the
    precomputed trajectory X(t), Z(t) (set Z=0 for pure 2D transport).
      lateral:  f_lat = -/+ (v / 2 lambda F) X(t)        [Table I]
      axial  :  common chirp with  d/dt(f_A+f_B) = (v^2/lambda F^2) Z
    pA + pB = 1 keeps total tweezer power constant during fading;
    pB = 0 (rectangular) keeps multi-tone array amplitudes uniform,
    pA = 1 puts the smooth fade on the single-tone A AOD (Table II).
    """
    f_lat = X / (2 * lamF_v)                  # v/(2 lambda F) * X(t)
    # integral of Z gives the common chirp frequency offset:
    f_chirp = np.cumsum(Z) / fs / (2 * lamF2_v2)
    chA = fading_shepard_channel(t, f0, df, 1,  pA, xi, -f_lat, f_chirp)
    chB = fading_shepard_channel(t, f0, df, M,  pB, xi, +f_lat, f_chirp)
    wA = synthesize_tones(*chA, fs)
    wB = synthesize_tones(*chB, fs)
    return wA, wB

# ======================================================================
# C) Crossfaded discrete hops (microscope scanning, single AOD per axis)
# ======================================================================

def raised_cosine_crossfade(n_fade):
    """theta ramps 0 -> pi/2 with raised-cosine timing; field envelopes
    cos(theta), sin(theta) => power cos^2 + sin^2 = 1 (the p=1 case)."""
    s = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_fade)))  # smooth 0->1
    th = (np.pi / 2) * s
    return np.cos(th), np.sin(th)

def hop_scan_waveform(frames, dwell_s, fade_s, fs):
    """
    frames : list of 1D arrays of tone frequencies [Hz]; one per scan frame
             (each frame = static multi-tone pattern, e.g. uniform amps).
    dwell_s: time at full amplitude per frame; fade_s: crossfade duration
             (choose fade_s >= AOD fill time D/v to avoid shock fronts).
    Per-frame phases: generalized Schroeder, re-anchored phase-continuously:
    the oscillator phase of every tone is accumulated through the whole
    buffer (never reset), and frame phases are imposed as offsets at the
    fade midpoint so each settled frame realizes the Schroeder pattern.
    """
    n_dwell, n_fade = int(dwell_s * fs), int(fade_s * fs)
    n_frame = n_dwell + n_fade
    n_total = n_frame * len(frames)
    # global tone bookkeeping: one oscillator per (frame, tone)
    out = np.zeros(n_total)
    t0 = 0
    for k, fk in enumerate(frames):
        Mk = len(fk)
        phik = schroeder_generalized(Mk)
        amps = np.ones(Mk) / np.sqrt(Mk)            # uniform power budget
        seg = slice(t0, t0 + n_frame)
        tt = np.arange(n_frame)
        # dwell + fade-out of this frame
        env = np.ones(n_frame)
        cf_out, _ = raised_cosine_crossfade(n_fade)
        env[n_dwell:] = cf_out
        # phase-continuous: anchor each tone's phase at segment start
        for m in range(Mk):
            theta = 2*np.pi*fk[m]*(tt/fs) + phik[m] \
                    + 2*np.pi*fk[m]*(t0/fs)          # global-time anchor
            out[seg] += amps[m] * env * np.cos(theta)
        # fade-in of NEXT frame overlapping the fade-out
        if k + 1 < len(frames):
            fn = frames[k + 1]
            phin = schroeder_generalized(len(fn))
            an = np.ones(len(fn)) / np.sqrt(len(fn))
            _, cf_in = raised_cosine_crossfade(n_fade)
            segf = slice(t0 + n_dwell, t0 + n_frame)
            tf = np.arange(n_fade) + n_dwell
            for m in range(len(fn)):
                theta = 2*np.pi*fn[m]*(tf/fs) + phin[m] \
                        + 2*np.pi*fn[m]*(t0/fs)
                out[segf] += an[m] * cf_in * np.cos(theta)
        t0 += n_frame
    return out

# ======================================================================
# Diagnostics: time-resolved IM3 (does suppression hold during motion?)
# ======================================================================

def im3_spectrogram(w, fs, k3=0.05, n_win=8192, hop=4096):
    """Spectrogram of the cubic-distortion-only signal -k3*w^3.
    Inspect: distortion ridges should track the (chirping) tone grid at
    a suppressed level, with no bursts during fades/hops."""
    d = -k3 * w**3
    n = (len(d) - n_win) // hop
    S = np.empty((n, n_win // 2 + 1))
    win = np.hanning(n_win)
    for i in range(n):
        seg = d[i*hop : i*hop + n_win] * win
        S[i] = np.abs(np.fft.rfft(seg))**2
    f_axis = np.fft.rfftfreq(n_win, 1/fs)
    t_axis = (np.arange(n) * hop + n_win/2) / fs
    return t_axis, f_axis, 10*np.log10(S + 1e-30)

# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    fs = 625e6
    # --- A: transport of an M=8 array, minimum-jerk lateral move ------
    T = 200e-6
    t = np.arange(int(T*fs)) / fs
    s = t / T
    Xtraj = 50e-6 * (10*s**3 - 15*s**4 + 6*s**5)      # minimum jerk, 50 um
    lamF_v = 808e-9 * 0.1 / 650.0                      # lambda*F/v  [m/Hz]
    lamF2_v2 = 808e-9 * 0.1**2 / 650.0**2
    wA, wB = transport_pair(t, Xtraj, np.zeros_like(t),
                            f0=100e6, df=2.5e6, M=8, fs=fs,
                            lamF_v=lamF_v, lamF2_v2=lamF2_v2)
    print("transport buffers:", wA.shape, wB.shape)

    # --- C: microscope scan, 4 frames of 6 beams, crossfaded hops -----
    rng = np.random.default_rng(1)
    frames = [100e6 + 2.5e6*np.sort(rng.choice(20, 6, replace=False))
              for _ in range(4)]
    w = hop_scan_waveform(frames, dwell_s=20e-6, fade_s=5e-6, fs=fs)
    ts, fa, S = im3_spectrogram(w, fs)
    print("scan buffer:", w.shape, "| spectrogram:", S.shape)