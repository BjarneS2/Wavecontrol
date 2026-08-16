"""
run_sort_live.py
Launch the Orca camera GUI and wire the 1D sorter to it WITHOUT editing the GUI.

The GUI (main_program_v3_3frames_series_params.py) emits `sorterhook` with the loading
frame (image 2) the moment it is read; we compute the occupancy, plan the moves and fire a
self-triggered AWG move (non-blocking) so the sort happens before image 3 (survival). After
image 3 the GUI emits `imagesignal3`, on which we revert to the original 1xN array.

Point CALIB_FOLDER at the folder produced by a calibration run (Sorter1D(None)); it holds
sorter_config.json + sorter_calibration.npz, so the sorter reloads instantly.

@author: Bjarne SchÃ¼mann
"""

import sys
import argparse
import pyqtgraph as pg
from PyQt5 import QtCore
from datetime import datetime
from zoneinfo import ZoneInfo

import orca_gui_bjarne as gui
from sorter import Sorter1D
import dcam_bjarne

cet_time = datetime.now(ZoneInfo("Europe/Paris"))
timestamp = cet_time.strftime("%Y-%m-%d_%H-%M-%S")

CALIB_FOLDER = r"N:\\SCI-NBI-quantop-data\\data\\LAQS\\2026\\20260729\\tweezerImages\\calibration"  # <-- set me
REPORT_PATH = rf"N:\\SCI-NBI-quantop-data\\data\\LAQS\\2026\\20260729\\report\\report_{timestamp}.json"
# Per-tone CH0 amplitude fractions (hand-tuned to equalize trap depths across the array).
# None -> equal share of the CH0 budget. Length must equal the calibrated site count n.
#   e.g. AMPLITUDES_CH0 = [0.040, 0.038, 0.042, 0.041, 0.039, 0.040, 0.043, 0.038]
AMPLITUDES_CH0 = None
AMPLITUDE_CH1 = None  # single CH1 tone fraction; None -> max_total_ch1

# Snap f_start / spacing / x-tone onto the card's DDS frequency grid so all array tones are
# exactly equally spaced (removes the slow power beating from off-grid rounding residuals).
SNAP_TO_GRID = True

# Trap-depth handling across a move. These are runtime knobs: they are NOT stored in
# sorter_config.json / sorter_calibration.npz, so what you set here is what runs.
#   AMP_RAMP_S : duration [s] of the hardware amplitude ramp at EACH end of a move. 0.0 =
#                the depth steps instantly from the static level to the move level and back
#                (for an 11-site array sorted down to 2 tones in "sum_amp" that is a 5.1x
#                jump). 20e-6 bridges it in 2 extra segments using the card's amp_slope.
#   AMP_RAMP_TOP: per-tone amplitude at the ramp ends; None = the static-arm level, i.e.
#                the depth is continuous from loading -> transport -> survival image.
#   AMP_RAMP_OUT: True  -> ramp back down at the end; the survival image is taken at the
#                          loading depth (AMP_RAMP_TOP).
#                 False -> no closing ramp; the survival image is taken in the deeper
#                          sorted traps and revert() brings the depth back afterwards.
AMP_RAMP_S = 15e-6
AMP_RAMP_TOP = None
AMP_RAMP_OUT = False
AMP_RAMP_WAYPOINTS = 20
MOVE_AMP_MODE = "sum_amp"  # sum_amp | match_static | equal_power | explicit

def main():
    dcam_bjarne.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
 
    srt = Sorter1D(
        CALIB_FOLDER,
        amplitudes_ch0=AMPLITUDES_CH0,
        amplitude_ch1=AMPLITUDE_CH1,
        snap_to_grid=SNAP_TO_GRID,
        amp_ramp_s=AMP_RAMP_S,
        amp_ramp_top=AMP_RAMP_TOP,
        amp_ramp_out=AMP_RAMP_OUT,
        amp_ramp_waypoints=AMP_RAMP_WAYPOINTS,
        move_amp_mode=MOVE_AMP_MODE,
        record_path=REPORT_PATH
    )
    if not args.no_plots:
        srt.cal.plot_sites()
        srt.cal.plot_mask()
 
    gui.Dcamapi.init()
    app = pg.mkQApp()
    win = gui.MainWindow(dcamapistat=True)
 
    # DirectConnection: sorterhook/imagesignal3 are emitted from the camera QThread, and a
    # plain Python callable would otherwise be delivered through the MAIN thread's event
    # loop (visible in the logs as 'LOOPN: n+1' printing before 'loaded : ...'). Direct
    # means sort_live() runs inline in the acquisition thread the moment the loading frame
    # is read, and revert() is ordered strictly after it.
    if args.dry:
        win.thread.sorterhook.connect(srt.dry_run, QtCore.Qt.DirectConnection)
    else:
        srt.connect()
        win.thread.sorterhook.connect(srt.sort_live, QtCore.Qt.DirectConnection)
        win.thread.imagesignal3.connect(
            lambda *_: srt.revert(), QtCore.Qt.DirectConnection
        )
 
    win.show()
    try:
        sys.exit(app.exec_())
    finally:
        if not args.dry:
            srt.close()

if __name__ == "__main__":
    main()



