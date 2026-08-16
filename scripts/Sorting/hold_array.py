"""
hold_array.py
Standalone "just hold the tones" script for calibration / trap-power tuning.

Arms the static 1xN tweezer array on CH0 (the sortable axis) plus the single static tone on
CH1 (the perpendicular axis) and holds them until you press Enter. No camera, no sorting, no
data-taking: run this while taking calibration shots, or while tuning AMPLITUDES_CH0 so every
trap comes out at roughly the same depth on the camera.

CHANGED vs the previous version:
  * phases come from sorter.static_phases(N), which reads optimized_phases.py and falls back
    to Schroeder. The old file computed Schroeder and then overwrote it with a hard-coded
    10-element array, so changing N silently broke it. There is now ONE definition of the
    phases and sorter.py and this script cannot disagree about what is on the card.
  * peak_of_sum() is imported from sorter.py rather than duplicated.
  * PHASE_SET lets you A/B optimised vs Schroeder on the photodiode by editing one line.

Requires the spcm SDK + the AWG card (run it on the lab PC).

@author: Bjarne Schuemann
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # parent: Controller
from Controller27072026 import AWGController  # type: ignore  # noqa: E402
from sorter import (  # type: ignore  # noqa: E402
    OPTIMIZED_STATIC_PHASES,
    peak_of_sum,
    schroeder_generalized,
    static_phases,
)

# ============================ config ============================
SERIAL = 24909
CORE_MAPPING = "20/1"  # CH0 = up to 20 tweezer tones, CH1 = 1 tone

N = 3  # number of tweezers on CH0
SPACING_UM = 4.6  # site separation [um]
CENTER_ARRAY = (
    False  # False -> f_start is the first site; True -> symmetric about f_start
)

F_START_HZ = 91.0e6  # frequency of site 0 (CH0)
F_MIN_HZ = 70.0e6
F_MAX_HZ = 130.0e6
X_FREQ_HZ = 91.0e6  # the single CH1 tone (perpendicular axis)
UM_PER_MHZ = 4.6 / 0.6

# Which phase prescription to put on CH0. The measurement worth doing is to flip between
# "optimized" and "schroeder" with everything else fixed and watch the photodiode: the
# optimised set should let you run ~1.6x the RF power (n=9) at the same peak.
#   "optimized" -> optimized_phases.py table, Schroeder for any N not in it
#   "schroeder" -> always the analytic pi*n(n-1)/N
#   list/array  -> exactly these N phases [rad]
PHASE_SET = "schroeder"  #"optimized"

# Per-channel output full-scale [V] (each channel drives a different AOD). SET FOR YOUR SETUP.
MAX_AMPV = 1.1
MAX_AMPV_CH1 = 1.2

# Power budget per channel, in units of "one tone at a=1": the constraint is sum_k a_k^2
# <= this. Delivered RF power is P = MAX_POWER_CH0 * V_fs^2 / (2*50 ohm), so 1.0 means
# "as much power as a single full-amplitude tone". Set it from your AOD damage threshold:
# measure P for one tone at a=1 and MAX_AMPV, then MAX_POWER_CH0 = P_damage / P_measured.
MAX_POWER_CH0 = 1.0
MAX_POWER_CH1 = 1.0
# Fraction of digital full scale the summed waveform's peak may reach (anti-clipping).
HEADROOM = 1

# Per-tone CH0 amplitude RATIOS. Only the relative values matter: the absolute scale is set
# by scale_amps() to the largest value that neither clips nor exceeds MAX_POWER_CH0. None ->
# equal weights. Otherwise a length-N list you tune by hand to equalize trap depths, e.g.
#   AMPLITUDES_CH0 = [1.00, 0.95, 1.05, 1.02, 0.97, 1.00, 1.07, 0.95, 1.01]
# NOTE: the optimised phase table assumes EQUAL weights. Once you hand-tune these, re-run
# optimize_phases.py with WEIGHTS set to them and regenerate optimized_phases.py.
AMPLITUDES_CH0 = None
AMPLITUDE_CH1 = None  # single CH1 tone weight; None -> 1.0 (scale is set by MAX_POWER_CH1)

# Snap f_start / spacing / x-tone onto the card's DDS frequency grid so all tones are
# EXACTLY equally spaced. Off-grid spacing makes each tone round to a slightly different
# grid point -> unequal spacings -> slow power beating; snapping removes it.
# This does NOT invalidate the phase table: for an equally spaced set the envelope peak is
# independent of both f_start and the spacing.
SNAP_TO_GRID = True
# ===============================================================


def resolve_phases(n, phase_set=PHASE_SET):
    """The N phases to program, and a label for the printout."""
    if isinstance(phase_set, str):
        key = phase_set.lower()
        if key == "schroeder":
            return schroeder_generalized(n), "Schroeder"
        if key == "optimized":
            in_table = n in OPTIMIZED_STATIC_PHASES
            return static_phases(n), "optimised" if in_table else "Schroeder (no table entry)"
        raise ValueError("PHASE_SET must be 'optimized', 'schroeder', or a length-N array")
    ph = np.asarray(phase_set, dtype=float)
    if ph.shape != (n,):
        raise ValueError("PHASE_SET array must have length N=%d (got %d)" % (n, ph.size))
    return ph, "explicit"


def site_positions(n, spacing_um, center):
    x = np.arange(n) * float(spacing_um)
    return x - x.mean() if center else x


def grid_spacing(f_step_hz):
    n_steps = max(1, round(SPACING_UM / UM_PER_MHZ * 1e6 / f_step_hz))
    return n_steps, n_steps * f_step_hz / 1e6 * UM_PER_MHZ


def scale_amps(w, freqs_hz, phases, power_budget, label):
    """Scale tone weights `w` (ratios kept) to the largest amplitudes that neither clip the
    DAC nor exceed `power_budget` = sum_k a_k^2, in units of one tone at a=1.

    Power goes as sum(a_k^2), not (sum a_k)^2, so the old "divide the amplitude budget by N"
    rule threw away a factor N in power. Under a good phase prescription the summed waveform
    peaks at ~1.1-1.2*sqrt(K)*a rather than K*a, so the anti-clipping limit permits far more
    than a_k = 1/K -- roughly half the single-tone power regardless of K.
    """
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise ValueError(f"{label}: amplitude weights must be non-negative.")
    pk = peak_of_sum(w, freqs_hz, phases)
    s_peak = HEADROOM / pk
    s_power = np.sqrt(power_budget / np.sum(w**2))
    s = min(s_peak, s_power)
    amps = w * s
    print(
        f"[hold_array] {label}: peak {s * pk:.3f} of full scale, "
        f"power {float(np.sum(amps**2)):.4f} x one full tone "
        f"({'peak' if s_peak < s_power else 'power'}-limited)"
    )
    return amps


def main():
    ctrl = AWGController(
        serial_number=SERIAL,
        f_start_hz=F_START_HZ,
        f_min_hz=F_MIN_HZ,
        f_max_hz=F_MAX_HZ,
        max_channel_amp_v=MAX_AMPV,
        max_channel_amp_v_ch1=MAX_AMPV_CH1,
        max_total_power=max(MAX_POWER_CH0, MAX_POWER_CH1),
        # sqrt(N) is the summed amplitude of a perfectly flat-envelope (ideal) N-tone
        # array and the theoretical floor of the envelope peak. scale_amps() enforces the
        # real peak directly, so this only fires if the weights are so lopsided that the
        # crest factor is worse than ideal.
        max_total_amplitude=float(np.sqrt(max(N, 1))),
        um_per_MHz=UM_PER_MHZ,
        core_mapping=CORE_MAPPING,
    )
    ctrl.connect()
    try:
        f_start_hz, spacing_um, x_freq_hz = F_START_HZ, SPACING_UM, X_FREQ_HZ
        f_step = ctrl.dds_freq_step_hz()
        print("[hold_array] DDS frequency grid step = %.6f Hz" % f_step)
        print("             (put this in the config so the offline dry run can snap too)")
        if SNAP_TO_GRID:
            f_start_hz = round(F_START_HZ / f_step) * f_step
            n_steps, spacing_um = grid_spacing(f_step)
            x_freq_hz = round(X_FREQ_HZ / f_step) * f_step
            ctrl.f_start_hz = f_start_hz
            print(
                "[hold_array] snapped to DDS grid (spacing %d steps -> %.6f um)"
                % (n_steps, spacing_um)
            )

        positions = site_positions(N, spacing_um, CENTER_ARRAY)
        x_pos_um = (x_freq_hz - f_start_hz) / 1e6 * UM_PER_MHZ

        if AMPLITUDES_CH0 is None:
            w0 = np.ones(N)
        else:
            w0 = np.asarray(AMPLITUDES_CH0, dtype=float)
            if w0.shape != (N,):
                raise ValueError(
                    "AMPLITUDES_CH0 must have length N=%d (got %d)" % (N, w0.size)
                )
        freqs0 = f_start_hz + positions / UM_PER_MHZ * 1e6

        phases0, label = resolve_phases(N, PHASE_SET)
        pk_used = peak_of_sum(w0, freqs0, phases0)
        pk_schr = peak_of_sum(w0, freqs0, schroeder_generalized(N))
        print(
            "[hold_array] phases: %s | envelope peak %.4f vs Schroeder %.4f "
            "vs floor %.4f -> %.2fx the RF power of Schroeder"
            % (label, pk_used, pk_schr, np.sqrt(np.sum(w0**2)), (pk_schr / pk_used) ** 2)
        )

        amps0 = scale_amps(w0, freqs0, phases0, MAX_POWER_CH0, "CH0")
        w1 = np.array([1.0 if AMPLITUDE_CH1 is None else float(AMPLITUDE_CH1)])
        amps1 = scale_amps(w1, [x_freq_hz], [0.0], MAX_POWER_CH1, "CH1")

        ctrl.program_static_2d(
            positions,
            np.ones(N, dtype=bool),
            [x_pos_um],
            [True],
            amplitudes_ch0=amps0,
            amplitudes_ch1=amps1,
            phases_ch0=phases0,
            phases_ch1=[0.0],
        )

        print("\nHolding %d CH0 tones + 1 CH1 tone." % N)
        print("  CH0 site freqs [MHz]: %s" % np.round(freqs0 / 1e6, 4))
        print("  CH0 phases [rad]:     %s" % np.round(phases0, 4))
        print("  CH0 amps:             %s" % np.round(amps0, 4))
        print("  CH1 tone: %.4f MHz  amp %.4f" % (x_freq_hz / 1e6, float(amps1[0])))
        input("\nPress Enter to stop and disconnect...")
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    main()
