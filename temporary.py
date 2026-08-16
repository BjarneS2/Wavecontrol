"""
Quick lab-side inspection of experimental run .npy files.

Each shot is a pickled dict: {"Images": [...], "globalvariables": {...}}.
Point this at a folder (or a single file) and it prints, per run-prefix:
  - how many shots
  - image shape/dtype
  - the global variables (param1..param4 + descriptions) for the first
    and last shot, and whether they change within the run

Usage:
    python temporary.py "C:\\path\\to\\run_folder"
    python temporary.py "C:\\path\\to\\one_shot.npy"
"""

import glob
import os
import sys
import re

import numpy as np

SHOT_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)_(?P<ts>\d{8}-\d{6})\.npy$")


def load_shot(path):
    return np.load(path, allow_pickle=True)[()]


def group_runs(folder):
    groups = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        m = SHOT_RE.match(os.path.basename(p))
        prefix = m["prefix"] if m else os.path.basename(folder)
        groups.setdefault(prefix, []).append(p)
    return groups


def describe_shot(d, path):
    if not isinstance(d, dict):
        print(f"  ! {path} is not a dict, it's {type(d)}")
        return
    keys = list(d.keys())
    print(f"  keys: {keys}")
    if "Images" in d:
        imgs = np.asarray(d["Images"])
        print(f"  Images: shape={imgs.shape} dtype={imgs.dtype}")
    gv = d.get("globalvariables")
    if gv is not None:
        print("  globalvariables:")
        for k, v in gv.items():
            print(f"    {k}: value={v[0]!r}  desc={v[1]!r}")


def describe_run(prefix, paths):
    print(f"\n=== {prefix} ({len(paths)} shots) ===")
    first = load_shot(paths[0])
    describe_shot(first, paths[0])

    if len(paths) > 1:
        last = load_shot(paths[-1])
        gv_first = first.get("globalvariables") if isinstance(first, dict) else None
        gv_last = last.get("globalvariables") if isinstance(last, dict) else None
        if gv_first and gv_last:
            changed = [
                k for k in gv_first if k in gv_last and gv_first[k][0] != gv_last[k][0]
            ]
            if changed:
                print(f"  ! globalvariables changed within the run: {changed}")
            else:
                print("  globalvariables constant across first/last shot")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1]

    if os.path.isfile(target):
        d = load_shot(target)
        print(f"=== {target} ===")
        describe_shot(d, target)
        return

    groups = group_runs(target)
    if not groups:
        print(f"no .npy files found in {target}")
        return

    for prefix, paths in groups.items():
        try:
            describe_run(prefix, paths)
        except Exception as e:
            print(f"  ! failed to read {prefix}: {e}")


if __name__ == "__main__":
    main()
