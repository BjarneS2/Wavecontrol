# Wavecontrol

A Python wrapper around the Spectrum Instrumentation AWG card (`spcm` SDK) for controlling atoms in optical tweezer arrays — DDS multi-tone waveform generation, live camera-based atom sorting/rearrangement, and analysis of the resulting experimental data. Written for the M4i.6631-x8 with two output channels. Development began as part of a physics master's thesis on optimal control of atomic motion in optical tweezer arrays, and has continued beyond it.

## Layout

- `src/` — the AWG/DDS driver (`Controller.py`), camera/GUI glue (`orca_gui_bjarne.py`, `dcam_bjarne.py`, `run_sort_live.py`), and phase-optimization helpers.
- `scripts/Sorting/` — the atom-rearrangement pipeline: HCA sorting, the 1D sliding-window sorter, image analysis/calibration, and the live `sorter.py` class.
- `scripts/DataAnalysis/` — the figure-generation pipeline; `run_all.py` regenerates every figure for every run.
- `scripts/Miscellaneous/` — standalone illustrative figures with no experimental-data dependency.
- `scripts/Experiments/` — one-off or dated experiment/analysis scripts kept for reference.
- `scripts/Data/` *(gitignored)* — raw experimental data.
- `scripts/Figures/` — generated figures from `scripts/DataAnalysis/`.

## Setup

```
pip install -r requirements.txt
```

The `spcm` SDK (Spectrum Instrumentation) is not on PyPI and must be installed separately from the vendor driver package.

## Related repository

Some scripts in `scripts/Experiments/` load control-protocol output (`.h5`) from the sibling thesis repo [`Atomove`](../Atomove), which contains the Julia optimal-control simulation code that produces those protocols.

## AI declaration

Generative AI assistance (Claude Sonnet) was used during the development of this framework, limited to plotting and visualization scripts, code review, and bug finding/fixing to work more efficiently. All design and implementation choices were made by the author; the AI served only as a tool to increase efficiency, and code changes have all been manually reviewed and approved.
