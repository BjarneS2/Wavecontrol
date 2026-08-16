#For inversion, first load into Mephisto/1064 to then steal with TiSaph


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from Controller import AWGController, arc_length_spacing

MAX_AMPV = 1.2
MAX_TOL = 0.8

ctrl = AWGController(serial_number=24909, realtime_priority=False, f_start_hz = 91.0e6, max_channel_amp_v = MAX_AMPV)
#ctrl.DDS_TIMER_MIN_NS = 6.4  # override for offline testing

N_channels = 2
T_amp = 20

amp_in_ch_0_val = 0.0
amp_fin_ch_0_val = 0.45
# amp_fin_ch_0_val = amp_in_ch_0_val
amp_in_ch_1_val = 0.45
amp_fin_ch_1_val = 0.45
# amp_fin_ch_1_val = amp_in_ch_1_val

T   = 50

transport_time = 1.5 #3e-4
swap = 5e-6
freeze_before_move = 2e-6
hold_time_for_image = 1.5 


transport_forth = np.linspace(0.0, transport_time, T)
amplitude_modulation = np.linspace(transport_time+1e-6, transport_time+swap, T_amp)
freeze_motion_n_amp = np.linspace(1e-6 + (transport_time+swap), (transport_time+swap)+freeze_before_move, 3)
transport_back = np.linspace(1e-6+(transport_time+swap+freeze_before_move), 2*transport_time+swap+freeze_before_move , T)

hold_array = np.linspace(0.001 + transport_back[-1], 0.001 + transport_back[-1] + hold_time_for_image, 10)
freeze_before_move_array = np.ones(len(freeze_motion_n_amp))

t = np.concatenate((transport_forth, amplitude_modulation, freeze_motion_n_amp, transport_back, hold_array))
print(t)
amps_in_ch_0 = amp_in_ch_0_val*np.ones(T)
amps_fin_ch_0 = amp_fin_ch_0_val*np.ones(T)
amps_ch_0 = np.linspace(0, 1, T_amp)
amps_ch_0_freeze = amp_fin_ch_0_val*np.ones(len(freeze_motion_n_amp))
amps_ch_0_hold = amp_fin_ch_0_val*np.ones(len(hold_array))

amps_in_ch_1 = amp_in_ch_1_val*np.ones(T)
amps_fin_ch_1 = amp_fin_ch_1_val*np.ones(T)
amps_ch_1 = np.linspace(0, 1, T_amp)
amps_ch_1_freeze = amp_fin_ch_1_val*np.ones(len(freeze_motion_n_amp))
amps_ch_1_hold = amp_fin_ch_1_val*np.ones(len(hold_array))


amp_arr_ch_0 = np.concatenate((amps_in_ch_0, amps_ch_0, amps_ch_0_freeze, amps_fin_ch_0))
amp_arr_ch_1 = np.concatenate((amps_in_ch_1, amps_ch_1, amps_ch_1_freeze, amps_fin_ch_1))

print('len', len(amp_arr_ch_0))

# amp_arr = np.concatenate((amp_arr_ch_0, amp_arr_ch_1))
# amp_arr = np.reshape(amp_arr, (N_channels, len(amp_arr)//N_channels))



s_ch_0   = np.linspace(0.0, 1.0, T)
s_fin_ch_0 = np.ones(T_amp + T )
s_freeze_ch_0 = np.ones(len(freeze_motion_n_amp)) 
s_hold_ch_0 = np.ones(len(hold_array))


s_ch_1 = s_ch_0[::-1] # np.ones(T) # 
s_in_ch_1 = np.ones(T_amp + T)
s_freeze_ch_1 = np.ones(len(freeze_motion_n_amp))
s_hold_ch_1 = np.ones(len(hold_array))

s_arr_ch_1 = np.concatenate((s_ch_0, s_freeze_ch_0, s_fin_ch_0))
s_arr_ch_0 = np.concatenate((s_in_ch_1, s_freeze_ch_1, s_ch_1))


#start_offset = (81.899 - 91.0) * 4.6/0.6  # distance * MHz_to_um
#end_offset = (79.82 - 91.0) * 4.6/0.6

#start_forth = 4.6/0.6 * 2.4
#end_forth = 0

#end_back = start_offset 
#start_back = end_offset
# WE NOW FLIP IT FOR THE REVERSE

#start_back = start_offset 
#end_back = end_offset

N = 8

start_pos_ch0 = 0 * 4.6/0.6 # (since 91MHz is set to be the 0 value here)
final_pos_ch0 = N* 4.6 # ((N*0.6 + 91) - 91) * 4.6/0.6 <-- (set N to how many spaces you wanna move)
start_forth = start_pos_ch0
end_forth = final_pos_ch0

start_pos_ch1 = ((79.79+N*0.6)-91) * 4.6/0.6 # (select 79.79+... to be the position of Mephisto/1064 at N spaces from origin)
final_pos_ch1 = (79.79 - 91) * 4.6/0.6 # since 79.79 is the same as 91 but for 1064.
start_back = start_pos_ch1
end_back = final_pos_ch1

pos_ch_0 = (end_forth - start_forth)  * (10*s_arr_ch_0**3 - 15*s_arr_ch_0**4 + 6*s_arr_ch_0**5) + start_forth # min-jerk STA 0 → 6 µm
pos_ch_0_hold = start_forth*s_hold_ch_0
pos_ch_1 = (end_back - start_back)  * (10*s_arr_ch_1**3 - 15*s_arr_ch_1**4 + 6*s_arr_ch_1**5) + start_back
pos_ch_1_hold = s_hold_ch_1*end_back

pos_ch_0_conc = np.concatenate((pos_ch_0, pos_ch_0_hold))
pos_ch_1_conc = np.concatenate((pos_ch_1, pos_ch_1_hold))

# pos = 35.2*s
STA_amplit_ch_0 = (amp_fin_ch_0_val - amp_in_ch_0_val)*(10*amps_ch_0**3 - 15*amps_ch_0**4 + 6*amps_ch_0**5) + amp_in_ch_0_val
STA_amplit_ch_1 = (amp_fin_ch_1_val - amp_in_ch_1_val)*(10*amps_ch_1**3 - 15*amps_ch_1**4 + 6*amps_ch_1**5) + amp_in_ch_1_val
#STA_amplit_ch_0 = (amp_fin_ch_0_val - amp_in_ch_0_val)*amps_ch_0 + amp_in_ch_0_val
#STA_amplit_ch_1 = (amp_fin_ch_1_val - amp_in_ch_1_val)*amps_ch_1 + amp_in_ch_1_val


amplit_ch_0 = np.concatenate((amps_in_ch_0, STA_amplit_ch_0, amps_ch_0_freeze, amps_fin_ch_0, amps_ch_0_hold))
amplit_ch_1 = np.concatenate((amps_in_ch_1, STA_amplit_ch_1, amps_ch_1_freeze, amps_fin_ch_1, amps_ch_1_hold))


amplit = np.stack((amplit_ch_0, amplit_ch_1))

pos = np.stack((pos_ch_0_conc, pos_ch_1_conc))
 
# First plan the move to see the quantized time/freq/...
result = ctrl.plan(t, pos, amplitudes = amplit)
newamp = amplit
newp = pos 
newt = t

amp_add = amplit[0]**2+amplit[1]**2
print(amplit[0])
print(amplit[1])
print(amp_add * MAX_AMPV**2)
if np.any(amp_add * MAX_AMPV**2 > MAX_TOL**2):
    print("ERROR: the amplitude of the cores exceeds the limit value...")
    exit()
# newp, newt = arc_length_spacing(result.position_arr[0], result.time_arr, 20)
# newp = np.reshape(newp, (1, len(newp)))
# print(result.time_arr*1e3)
# print(np.diff(result.freqs_kt)[0])
print("peak velocity :   {:.2f} $\\mu m/ms$ ".format(np.max(np.diff(result.position_arr)[0][:T]/(np.diff(result.time_arr)[:T]*1e3))))
print("average velocity :   {:.2f} $\\mu m/ms$ ".format(np.mean(np.diff(result.position_arr)[0][:T]/(np.diff(result.time_arr)[:T]*1e3))))


fig, ax = plt.subplots()
# ax.plot(result.time_arr[1:]*1e3, np.diff(result.position_arr)[0]/(np.diff(result.time_arr)*1e3))
ax.plot(result.time_arr*1e3, pos[0], label = "position CH0")
ax.plot(result.time_arr*1e3, pos[1], label = "position CH1")

# ax.plot(result.time_arr*1e3, result.position_arr[0], label="Original Trajectory")
# ax.plot(newt*1e3, newp, "x", label="Arc Length Trajectory")
# ax.axhline(21)
ax.set_xlabel('time [ms]')
ax.set_ylabel('position [$\\mu m$]')

ax.legend()
plt.show()

fig, ax= plt.subplots()
ax.plot(result.time_arr*1e3, amplit[0]*MAX_AMPV, "x-", label = "Amplit0", color="red")
ax.plot(result.time_arr*1e3, amplit[1]*MAX_AMPV, "--", label = "Amplit1", color="red")
ax.plot(result.time_arr*1e3, (amplit[1]+amplit[0])*MAX_AMPV, label = "Amplit Combined", color="orange")

#ax.plot(result.time_arr*1e3, (result.amps_kt)[0], label = "Amp CH0")
#ax.plot(result.time_arr*1e3, (result.amps_kt)[1], label = "Amp CH1")
ax.set_xlabel('time [ms]')
ax.set_ylabel('Amplitude')

ax.legend()
plt.show()
exit()
#ctrl.connect()
#n_segments = ctrl.move(newt, newp, channel=0, amplitudes= newamp, hold = 0, back_and_forth=False, continuous=True, force_trigger=True)
#input("press enter to exit...\n")
#ctrl.disconnect()