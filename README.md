# AWGController

A Python wrapper around the Spectrum Instrumentation AWG card (`spcm` SDK) for controlling atoms in optical tweezer arrays — DDS multi-tone waveform generation, live camera-based atom sorting/rearrangement, and analysis of the resulting experimental data. Written specifically for the M4i.6631-x8 with two output channels. Developed alongside a physics thesis on optimal control of atomic motion in optical tweezer arrays.

## Folder structure

- **`Controller.py`** — core AWG/DDS driver: frequency table management, per-channel core mapping, waveform preview (`plan()`).
- **`Fading_Shepard.py`** — Schroeder-phase / fading-Shepard multi-tone waveform generation, used to reduce intermodulation distortion.
- **`run_sort_live.py`** — glue script wiring the live camera feed into the sorter.
- **`main_program_v3_3frames_series_params.py`** — PyQt5 GUI for camera-based experiment control.
- **`Sorting/`** — core atom-rearrangement pipeline: HCA sorting, 1D sliding-window sorter, image analysis/calibration, the live `sorter.py` class, and its tests/offline verification.
- **`Own Data Analysis/`** — the canonical thesis-figure generation pipeline: one script per figure, reading experiment output and producing the plots used in the thesis. `run_all.py` regenerates every figure for every run.
- **`Miscellaneous/`** — standalone illustrative figures with no experimental-data dependency (e.g. Gaussian-beam/tweezer-potential and Doppler-shift diagrams).
- **`Experiments/`** — one-off or dated experiment/analysis scripts kept for reference; not part of the maintained pipeline.
- **`Data/`** *(gitignored)* — raw experimental data (camera frames, sorting reports).
- **`Figures/`** *(gitignored)* — generated figures from `Own Data Analysis/`.

## Setup

```
pip install -r requirements.txt
```

The `spcm` SDK (Spectrum Instrumentation) is not on PyPI and must be installed separately from the vendor driver package. 

## Related repository

Some scripts in `Experiments/` (e.g. `tryOptMove.py`) load control-protocol output (`.h5`) from the sibling thesis repo [`Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays`](../Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays), which contains the Julia optimal-control simulation code that produces those protocols.
