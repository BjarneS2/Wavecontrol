# For transfer between two stationary traps (mephisto st. and tisaph aux.):

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from Controller import AWGController, arc_length_spacing
# from Fading_Shepard import schroeder_generalized, render_segments, crest_factor


def STA(p0, p1, arr):
    return (p1 - p0) * (10 * arr**3 - 15 * arr**4 + 6 * arr**5) + p0


def schroeder_generalized(M):
    """phi_n = 2*pi*n(n-1)/(2M) = pi*n(n-1)/M, n = 0..M-1  (Lu et al. Eq. 4).
    Quadratic in n  ->  nearest-neighbor difference phases advance as
    2*pi*n/M: the M in-band tones' IM2 phasors at spacing df form the M-th
    roots of unity and cancel; IM3 phasors form Gauss sums (sqrt suppression)."""
    if M < 2:
        return np.zeros(max(M, 1))
    n = np.arange(M)
    return np.pi * n * (n - 1) / M


MAX_AMPV = 1.2
MAX_TOL = 0.8

ctrl = AWGController(
    serial_number=24909,
    realtime_priority=False,
    f_start_hz=91.0e6,
    max_channel_amp_v=MAX_AMPV,
)


"""
The Channel 0 and 1 will be 1064, so we need to convert these to the 1064 frequency range.
    | 933nm (frequency in MHz needed)       | 1064nm (frequency in MHz needed)  | Corresponding position in um
0   | 91.0                                  | 79.79                             | 0um
1   | 91.6                                  | 80.39                             | 4.6um
...
N   | 91.0 + N * 0.6                        | 79.79 + N * 0.6                   | N * 4.6um

"""

N_channels = 3
N = 3  # NUMBER OF SITES THAT WE WANNA MOVE ACROSS (1, 2, 3, 4, 5, 6, 7)
stationary_ch0 = (79.79 - 91.0) * 4.6 / 0.6
stationary_ch1 = ((79.79 + N * 0.6) - 91.0) * 4.6 / 0.6
auxiliary_start = 0
auxiliary_end = 4.6

# Use Ch0 + Ch1 for the 2 stationary traps
# Use Ch2 for the auxiliary trap to steal and deliver

# Phases in the protocol:
# ==================================================
# first load into ch0 where ch1 is off
# ==================================================

resolution_1 = 2  # number of points for pos/amp modifications
time_step1 = 2  # [s]
amp_ch0_step1 = 1
amp_ch1_step1 = 0
amp_ch2_step1 = 0

pos_ch0_step1 = stationary_ch0  # [um]
pos_ch1_step1 = stationary_ch1
pos_ch2_step1 = auxiliary_start  # [um] translate since this is likely offset

# ==================================================
# then turn on ch1 and ch2
# ==================================================

resolution_2 = 20  # number of points for pos/amp modifications
time_step2 = 5e-6  # [s]
amp_ch0_step2 = 1
amp_ch1_step2 = 1
amp_ch2_step2 = 1

offset_from_ch0 = 0  # SETS THE VALUE HOW MUCH OFFSET FROM THE INITIAL STATIONARY TWEEZER WE WANNA HAVE
pos_ch0_step2 = stationary_ch0  # [um]
pos_ch1_step2 = stationary_ch1
pos_ch2_step2 = auxiliary_start + offset_from_ch0  # [um] overlap ch0 +/- small offset

# ==================================================
# then move ch2 ontop of ch1
# ==================================================

resolution_3 = 50  # number of points for pos/amp modifications
time_step3 = 1  # [s]
amp_ch0_step3 = 1
amp_ch1_step3 = 1
amp_ch2_step3 = 1

offset_from_ch1 = (
    0  # SETS THE VALUE HOW MUCH OFFSET FROM THE FINAL STATIONARY TWEEZER WE WANNA HAVE
)
pos_ch0_step3 = stationary_ch0
pos_ch1_step3 = stationary_ch1
pos_ch2_step3 = auxiliary_end + offset_from_ch1  # [um] overlap ch0 +/- small offset

# ==================================================
# finally turn off ch2 (you can also turn off ch0 but it might be
# interesting to see if stealing fails if the ato remains or is ejected)
# ==================================================

resolution_4 = 20  # number of points for pos/amp modifications
time_step4 = 5e-6  # [s]
amp_ch0_step4 = 1  # see comment above about turning it off
amp_ch1_step4 = 1
amp_ch2_step4 = 0

# should stay the same
pos_ch0_step4 = pos_ch0_step3  # [um]
pos_ch1_step4 = pos_ch1_step3
pos_ch2_step4 = pos_ch2_step3

# ==================================================
# do not forget to keep the final position for the image!
# ==================================================

resolution_5 = 2  # number of points for pos/amp modifications
time_step5 = 1.5  # [s]
amp_ch0_step5 = 1  # see comment above about turning it off
amp_ch1_step5 = 1
amp_ch2_step5 = 0

# should stay the same
pos_ch0_step5 = pos_ch0_step3  # [um]
pos_ch1_step5 = pos_ch1_step3
pos_ch2_step5 = pos_ch2_step3

# ==================================================
# Now we create the time arrays given the resolutions
# ==================================================
time1 = np.linspace(0, time_step1, resolution_1)
time2 = np.linspace(time1[-1] + 0.5e-6, time1[-1] + 0.5e-6 + time_step2, resolution_2)
time3 = np.linspace(time2[-1] + 0.5e-6, time2[-1] + 0.5e-6 + time_step3, resolution_3)
time4 = np.linspace(time3[-1] + 0.5e-6, time3[-1] + 0.5e-6 + time_step4, resolution_4)
time5 = np.linspace(time4[-1] + 0.5e-6, time4[-1] + 0.5e-6 + time_step5, resolution_5)


# ==================================================
# Now we create the amplitude arrays given the resolutions
# ==================================================
# Step 1 == LOAD INTO CH0
amp_ch0_s1 = np.ones(resolution_1) * amp_ch0_step1
amp_ch1_s1 = np.ones(resolution_1) * amp_ch1_step1
amp_ch2_s1 = np.ones(resolution_1) * amp_ch2_step1
# Step 2 == TURN ON CH1 and CH2
amp_ch0_s2 = STA(amp_ch0_step1, amp_ch0_step2, np.linspace(0, 1, resolution_2))
amp_ch1_s2 = STA(amp_ch1_step1, amp_ch1_step2, np.linspace(0, 1, resolution_2))
amp_ch2_s2 = STA(amp_ch2_step1, amp_ch2_step2, np.linspace(0, 1, resolution_2))
# Step 3 == MOVE CH2 FROM CH0 TO CH1
if amp_ch0_step2 == amp_ch0_step3:
    amp_ch0_s3 = np.ones(resolution_3) * amp_ch0_step3
else:
    amp_ch0_s3 = STA(amp_ch0_step2, amp_ch0_step3, np.linspace(0, 1, resolution_3))

if amp_ch1_step2 == amp_ch1_step3:
    amp_ch1_s3 = np.ones(resolution_3) * amp_ch1_step3
else:
    amp_ch1_s3 = STA(amp_ch1_step2, amp_ch1_step3, np.linspace(0, 1, resolution_3))

if amp_ch2_step2 == amp_ch2_step3:
    amp_ch2_s3 = np.ones(resolution_3) * amp_ch2_step3
else:
    amp_ch2_s3 = STA(amp_ch2_step2, amp_ch2_step3, np.linspace(0, 1, resolution_3))
# Step 4 == TURN OFF CH2
amp_ch0_s4 = STA(amp_ch0_step3, amp_ch0_step4, np.linspace(0, 1, resolution_4))
amp_ch1_s4 = STA(amp_ch1_step3, amp_ch1_step4, np.linspace(0, 1, resolution_4))
amp_ch2_s4 = STA(amp_ch2_step3, amp_ch2_step4, np.linspace(0, 1, resolution_4))
# Step 5 == HOLDING
amp_ch0_s5 = np.ones(resolution_5) * amp_ch0_step5
amp_ch1_s5 = np.ones(resolution_5) * amp_ch1_step5
amp_ch2_s5 = np.ones(resolution_5) * amp_ch2_step5


# ==================================================
# Now we create the positon arrays given the resolutions
# ==================================================
# Ch0 / Ch1 == STATIONARY the entire time (constant per step, same value across all steps)
pos_ch0_s1 = np.ones(resolution_1) * pos_ch0_step1
pos_ch1_s1 = np.ones(resolution_1) * pos_ch1_step1
pos_ch0_s2 = np.ones(resolution_2) * pos_ch0_step2
pos_ch1_s2 = np.ones(resolution_2) * pos_ch1_step2
pos_ch0_s3 = np.ones(resolution_3) * pos_ch0_step3
pos_ch1_s3 = np.ones(resolution_3) * pos_ch1_step3
pos_ch0_s4 = np.ones(resolution_4) * pos_ch0_step4
pos_ch1_s4 = np.ones(resolution_4) * pos_ch1_step4
pos_ch0_s5 = np.ones(resolution_5) * pos_ch0_step5
pos_ch1_s5 = np.ones(resolution_5) * pos_ch1_step5
# Ch2 == AUXILIARY: held for steps 1,2,4,5; STA move only in step 3
pos_ch2_s1 = np.ones(resolution_1) * pos_ch2_step1  # at auxiliary_start
pos_ch2_s2 = np.ones(resolution_2) * pos_ch2_step2  # held during turn-on
""" We could think about making this arclength spaced """
# pos_ch0_s3 = arc_length_spacing(STA(pos_ch2_step2, pos_ch2_step3, np.linspace(0, 1, 1000)), np.linspace(0, 1, 1000), resolution_3)
pos_ch2_s3 = STA(
    pos_ch2_step2, pos_ch2_step3, np.linspace(0, 1, resolution_3)
)  # THE motion
pos_ch2_s4 = np.ones(resolution_4) * pos_ch2_step4  # held during turn-off
pos_ch2_s5 = np.ones(resolution_5) * pos_ch2_step5  # held for imaging


# ==================================================
# Concatenate the per-step segments and stack into (K, T) arrays
# ==================================================
t = np.concatenate((time1, time2, time3, time4, time5))

amp_ch0 = np.concatenate((amp_ch0_s1, amp_ch0_s2, amp_ch0_s3, amp_ch0_s4, amp_ch0_s5))
amp_ch1 = np.concatenate((amp_ch1_s1, amp_ch1_s2, amp_ch1_s3, amp_ch1_s4, amp_ch1_s5))
amp_ch2 = np.concatenate((amp_ch2_s1, amp_ch2_s2, amp_ch2_s3, amp_ch2_s4, amp_ch2_s5))

pos_ch0 = np.concatenate((pos_ch0_s1, pos_ch0_s2, pos_ch0_s3, pos_ch0_s4, pos_ch0_s5))
pos_ch1 = np.concatenate((pos_ch1_s1, pos_ch1_s2, pos_ch1_s3, pos_ch1_s4, pos_ch1_s5))
pos_ch2 = np.concatenate((pos_ch2_s1, pos_ch2_s2, pos_ch2_s3, pos_ch2_s4, pos_ch2_s5))

pos = np.stack((pos_ch0, pos_ch1, pos_ch2))  # (3, T)
amp = np.stack((amp_ch0, amp_ch1, amp_ch2))  # (3, T)


# ==================================================
# Dry-run plan + plot (uncomment move() to stream to hardware)
# ==================================================
# Schroeder phases (per tone) suppress intermodulation distortion between the 3 tones.
phases = schroeder_generalized(N_channels)
result = ctrl.plan(t, pos, amplitudes=amp)  # ,phases=phases)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for k, lbl in enumerate(("Ch0 (stat)", "Ch1 (stat)", "Ch2 (aux)")):
    ax1.plot(result.time_arr, result.position_arr[k], label=lbl)
    ax2.plot(result.time_arr, result.amps_kt[k], label=lbl)
ax1.set_ylabel("position [um]")
ax2.set_ylabel("amp")
ax2.set_xlabel("t [s]")
ax1.legend()
plt.show()

# ==================================================
# Real RF signal: individual tones + summed output, at the instant most tones are on
# ==================================================
# fs_rf = 1.25e9                                        # DAC sample rate
# ic    = int(np.argmax(result.amps_kt.sum(axis=0)))    # index where combined amplitude peaks
# tc    = float(result.time_arr[ic])
# win   = (tc, min(tc + 1e-6, float(result.time_arr[-1])))

# tw, waves = render_segments(result.freqs_kt, result.amps_kt, result.phases, result.time_arr, fs_rf, win)
# summed    = waves.sum(axis=0)
# cf        = crest_factor(summed)
# _, waves0 = render_segments(result.freqs_kt, result.amps_kt, np.zeros_like(result.phases), result.time_arr, fs_rf, win)
# cf0       = crest_factor(waves0.sum(axis=0))
# print(f"crest factor: Schroeder phases = {cf:.3f}   vs   zero-phase = {cf0:.3f}   "
#       f"({100*(1-cf/cf0):.1f}% lower)")

# fig2, (bx1, bx2) = plt.subplots(2, 1, sharex=True)
# for k in range(waves.shape[0]):
#     bx1.plot(tw*1e6, waves[k], lw=0.8, label=f"tone {k}")
# bx2.plot(tw*1e6, summed, color="k", lw=0.8)
# bx1.set_ylabel("individual tones"); bx2.set_ylabel("summed output")
# bx2.set_xlabel("t [us]"); bx1.legend(loc="upper right", fontsize=8)
# bx1.set_title(f"real DDS signal @ t={tc:.3g}s  |  crest factor {cf:.2f} (zero-phase {cf0:.2f})")
# plt.show()

# --- Uncomment to stream to hardware (test after verifying the plot) ---
# ctrl.connect()
# n_seg = ctrl.move(t, pos, channel=0, amplitudes=amp, phases=phases, continuous=True, force_trigger=True)
# input("Press Enter to disconnect...")
# ctrl.disconnect()
