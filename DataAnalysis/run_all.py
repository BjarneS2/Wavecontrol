"""
Runs every plotting script's `plot()` entry point once, saving every figure under
Figures/<Sorting|Transportation>/<run>/<title>.png. Each script is also runnable on its
own to recreate a single figure for a chosen run -- see that script's --help.

Besides the existing pooled/all-runs figure, this also produces one figure per
individual run discovered in the relevant data folder. tweezerLoad1x11-bestsort-80us and
tweezerLoad1x11-sortbest-80usAGAIN are always treated as a single pooled run named
tweezerLoad1x11-80us_pooled (see CommonThings.MERGE_GROUPS) -- everywhere runs are
listed or plotted individually, that pair shows up once, not twice.

TunerForCalibration.py is excluded: it's an interactive lab dashboard, not a batch figure.
SingleAtomTransport.py has no per-run concept (it plots a hardcoded set of tweezerLoad1x2
runs into one 3-panel figure) so it only gets the single pooled call.
FidelityOverDistance.py already saves a figure per run inside its own pooled plot(save=True)
call (per_run=True by default), so it is not looped again here.

@author: Bjarne Schümann
"""

import runpy
import sys
import time
from pathlib import Path

import CommonThings as C

# scripts whose plot(runs=..., save=..., name=...) pools over Data/tweezerImagesSorting1D
SORTING_SCRIPTS = [
    "PerRunComparison.py",
    "ExpectationsOfDelivery.py",
    "SortedShotsByK_SourceOverTarget.py",
    "HeatmapMeanLostAtoms.py",
    "HeatmapSurvivalHopping.py",
    "SurvivalBySourceTargerSite.py",
    "OptimizationPotentialForSortingBasedOnCalibration.py",
    "PerSiteThresholds.py",
    "LossDistributions.py",
    "SurvivalStationaryVsMoved.py",
]

# scripts whose plot(runs=...) pools over Data/tweezerImagesSingleAtomTransport
TRANSPORT_SCRIPTS = [
    "SpeedOfTrajectories.py",
]

# scripts already run once, no per-run loop (see module docstring)
POOLED_ONLY_SCRIPTS = [
    "FidelityOverDistance.py",
]

NO_RUN_CONCEPT_SCRIPTS = [
    "SingleAtomTransport.py",
]

SCRIPTS = (
    SORTING_SCRIPTS
    + TRANSPORT_SCRIPTS
    + POOLED_ONLY_SCRIPTS
    + ["LoadingDrift.py"]
    + NO_RUN_CONCEPT_SCRIPTS
)

HERE = Path(__file__).resolve().parent

# scripts without a "runs=None supports substring accepting name=" signature that instead
# derive the saved name straight from the ds.name (already correct after the merge)
NO_NAME_KWARG = {"SortedShotsByK_SourceOverTarget.py", "SpeedOfTrajectories.py"}


def _substrings_for(name, groups=C.MERGE_GROUPS):
    """Merged pooled name -> the original substrings that must both be matched;
    any other run name -> itself, unchanged."""
    for *subs, merged in groups:
        if name == merged:
            return list(subs)
    return [name]


def _sorting_run_names():
    groups = C.discover_runs(str(C.SORTING1D_IMAGES), min_shots=20, verbose=False)
    return C.merge_run_names(sorted(groups))


def _transport_run_names():
    names = set()
    for cfg in C.TRANSPORT_CONFIGS:
        _, _, cfg_runs = C.load_transport_config(str(C.TRANSPORT_IMAGES), cfg, 1)
        names.update(cfg_runs)
    return C.merge_run_names(sorted(names))


def _run_one(name):
    """Executes one script's module body, calls its pooled plot(save=True), then --
    unless the script is in POOLED_ONLY_SCRIPTS/NO_RUN_CONCEPT_SCRIPTS -- one plot() call
    per individual run (with the 80us pair collapsed to one pooled run)."""
    ns = runpy.run_path(str(HERE / name), run_name=Path(name).stem)
    ns["plot"](save=True)

    if name == "LoadingDrift.py":
        ns["plot"](dataset="transport", save=True)
        for run_name in _sorting_run_names():
            ns["plot"](
                dataset="sorting",
                runs=_substrings_for(run_name),
                name=run_name,
                save=True,
            )
        for run_name in _transport_run_names():
            ns["plot"](
                dataset="transport",
                runs=_substrings_for(run_name),
                name=run_name,
                save=True,
            )
        return

    if name in POOLED_ONLY_SCRIPTS or name in NO_RUN_CONCEPT_SCRIPTS:
        return

    run_names = (
        _sorting_run_names() if name in SORTING_SCRIPTS else _transport_run_names()
    )
    for run_name in run_names:
        kwargs = {"runs": _substrings_for(run_name), "save": True}
        if name not in NO_NAME_KWARG:
            kwargs["name"] = run_name
        ns["plot"](**kwargs)


def main():
    failed = []
    for name in SCRIPTS:
        print(f"-- {name} ...", flush=True)
        t0 = time.time()
        try:
            _run_one(name)
            print(f"   ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed.append(name)
            print(f"   FAILED: {e!r}")
    print()
    if failed:
        print(f"{len(failed)} of {len(SCRIPTS)} scripts failed: {failed}")
        return 1
    print(f"all {len(SCRIPTS)} scripts completed -- see Figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
