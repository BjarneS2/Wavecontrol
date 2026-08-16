"""
Tests the independent-per-atom loss model (Eq. 6.3: each atom survives as its own
Bernoulli draw, p_stat for stationary atoms / p_move for movers) two ways:

1. On a baseline run, per (K, M) cell: does the observed shot-to-shot variance in
   atoms-lost match the variance the independence model predicts? If observed
   variance is bigger, something fails whole shots at once instead of one atom
   at a time (both stories give the same MEAN loss, only the VARIANCE tells them
   apart).
2. Once the model is trusted on the baseline, its p_stat/p_move are plugged into a
   test run's own (K, M) pairs to predict that run's total atoms lost. If the
   test run lost far more than predicted, its bad numbers aren't explained by its
   (K, M) mix alone -- evidence for a real per-shot cost (e.g. long sequences
   failing outright) that the baseline alone doesn't capture.

@author: Bjarne Schümann
"""

import argparse
import sys
from collections import Counter

import CommonThings as C
import numpy as np
from scipy import stats

CATEGORY = C.CAT_SORTING
DEFAULT_BASELINE = ["80us_pooled"]
DEFAULT_TEST = ["sortto0"]
MIN_CELL_N = 5  # cells with fewer shots than this are too noisy to judge


def _load_all(images=None, min_shots=20):
    """Every run in the images folder, merged (e.g. the two 80us halves -> 80us_pooled)."""
    images = images or str(C.SORTING1D_IMAGES)
    cal = C.load_calibration(str(C.SORTING1D_CAL), verbose=False)
    groups = C.discover_runs(images, min_shots, verbose=False)
    run_dss = []
    for name in sorted(groups):
        ds = C.load_dataset(name, groups[name], cal, verbose=False)
        C.build_plans(ds, verbose=False)
        run_dss.append(ds)
    return C.merge_run_datasets(run_dss)


def _select(run_dss, substrings):
    """merged Dataset list -> those whose .name matches any of the given substrings."""
    return [ds for ds in run_dss if any(s in ds.name for s in substrings)]


def dispersion_table(loss, min_cell_n=MIN_CELL_N):
    """Per-(K,M) cell: n, mean/var observed vs. the independence-model prediction.
    Cells with var_obs == 0 despite n >= 20 are flagged separately (they signal a
    deterministic, not merely low-variance, failure mode -- the opposite of the
    catastrophic-common-mode story, but still a break from independence)."""
    rows, anomalies = [], []
    for (k, m), c in sorted(loss["cells"].items()):
        if c["n"] < min_cell_n:
            continue
        mean_obs = float((c["obs"] * np.arange(len(c["obs"]))).sum() / c["n"])
        ratio = c["var_obs"] / c["var_exp"] if c["var_exp"] > 0 else np.nan
        row = {
            "k": k,
            "m": m,
            "n": c["n"],
            "mean_obs": mean_obs,
            "var_obs": c["var_obs"],
            "var_exp": c["var_exp"],
            "ratio": ratio,
        }
        if c["var_obs"] == 0 and c["n"] >= 20:
            anomalies.append(row)
        else:
            rows.append(row)
    return rows, anomalies


def pooled_dispersion_test(rows):
    """T = sum (n_i - 1) * var_obs_i / var_exp_i ~ chi2(df) under H0 (independence).
    Excludes rows already pulled out as anomalies by the caller."""
    T = sum((r["n"] - 1) * r["ratio"] for r in rows if np.isfinite(r["ratio"]))
    df = sum(r["n"] - 1 for r in rows if np.isfinite(r["ratio"]))
    pval = stats.chi2.sf(T, df) if df else np.nan
    return T, df, pval


def predict_test_run(test_rows, p_stat, p_move):
    """Plug the test run's own (K, M) per shot into Eq. 6.3 using the BASELINE's
    p_stat/p_move, and compare the predicted total loss to what was actually lost."""
    k = test_rows[:, 0].astype(float)
    m = test_rows[:, 1].astype(float)
    lost = test_rows[:, 2].astype(float)
    pred_mean = (k - m) * (1 - p_stat) + m * (1 - p_move)
    pred_var = (k - m) * p_stat * (1 - p_stat) + m * p_move * (1 - p_move)
    pred_total, pred_var_total = pred_mean.sum(), pred_var.sum()
    actual_total = lost.sum()
    z = (actual_total - pred_total) / np.sqrt(pred_var_total)
    return {
        "n_shots": len(k),
        "pred_total": pred_total,
        "actual_total": actual_total,
        "excess": actual_total - pred_total,
        "excess_pct": 100 * (actual_total - pred_total) / pred_total,
        "z": z,
    }


def run_analysis(baseline_runs, test_runs, images=None, min_shots=20):
    all_ds = _load_all(images, min_shots)
    base_ds = _select(all_ds, baseline_runs)
    test_ds = _select(all_ds, test_runs)
    if not base_ds or not test_ds:
        return None

    base_loss = C.loss_distributions(base_ds)
    rows, anomalies = dispersion_table(base_loss)
    T, df, pval = pooled_dispersion_test(rows)

    test_loss = C.loss_distributions(test_ds)
    pred = predict_test_run(test_loss["rows"], base_loss["p_stat"], base_loss["p_move"])

    km_counts = Counter(
        zip(test_loss["rows"][:, 0].tolist(), test_loss["rows"][:, 1].tolist())
    )

    return {
        "baseline_name": "+".join(ds.name for ds in base_ds),
        "baseline_shots": sum(ds.n_shots for ds in base_ds),
        "p_stat": base_loss["p_stat"],
        "p_move": base_loss["p_move"],
        "cells": rows,
        "anomalies": anomalies,
        "T": T,
        "df": df,
        "pval": pval,
        "test_name": "+".join(ds.name for ds in test_ds),
        "test_shots": sum(ds.n_shots for ds in test_ds),
        "test_p_stat": test_loss["p_stat"],
        "test_p_move": test_loss["p_move"],
        "prediction": pred,
        "test_km_counts": km_counts,
    }


def print_report(r):
    print(
        f"BASELINE ({r['baseline_name']}, n={r['baseline_shots']} shots): "
        f"p_stat={r['p_stat']:.4f}  p_move={r['p_move']:.4f}"
    )
    print()
    print(f"{'K':>3} {'M':>3} {'n':>5} {'mean_obs':>9} {'var_obs':>9} {'var_exp':>9} {'ratio':>7}")
    for c in r["cells"]:
        print(
            f"{c['k']:>3} {c['m']:>3} {c['n']:>5} {c['mean_obs']:>9.3f} "
            f"{c['var_obs']:>9.3f} {c['var_exp']:>9.3f} {c['ratio']:>7.2f}"
        )
    if r["anomalies"]:
        print()
        print("Anomalous cells excluded from the pooled test (var_obs == 0 despite n >= 20):")
        for c in r["anomalies"]:
            print(f"  K={c['k']} M={c['m']} n={c['n']} mean_obs={c['mean_obs']:.3f}")
    print()
    print(
        f"Pooled dispersion test: T={r['T']:.1f}  df={r['df']}  "
        f"mean(var_obs/var_exp)={r['T'] / r['df']:.3f}  chi2 p-value={r['pval']:.4f}"
    )
    print()
    p = r["prediction"]
    print(f"TEST RUN ({r['test_name']}, n={r['test_shots']} shots): "
          f"own p_stat={r['test_p_stat']:.4f}  own p_move={r['test_p_move']:.4f}")
    print(f"  predicted total lost (baseline model, test's own K,M): {p['pred_total']:.1f}")
    print(f"  actual total lost:                                     {p['actual_total']:.0f}")
    print(f"  excess: {p['excess']:.1f} atoms ({p['excess_pct']:.1f}% of predicted)")
    print(f"  z-score: {p['z']:.2f}")


def write_markdown(r, path):
    lines = []
    lines.append("# Independence-Model Test: Baseline Dispersion & sortto0 Prediction\n")
    lines.append(
        f"Source: `IndependenceModelTest.py`, baseline=`{r['baseline_name']}` "
        f"({r['baseline_shots']} shots), test=`{r['test_name']}` ({r['test_shots']} shots).\n"
    )
    lines.append(
        "Eq. (6.3) independence model: each atom survives as an independent Bernoulli "
        "draw, `p_stat` for stationary atoms and `p_move` for movers. Both stories "
        "(independent per-atom loss vs. occasional whole-shot wipeout) give the same "
        "*mean* loss -- only the *variance* tells them apart.\n"
    )

    lines.append("## 1. Baseline dispersion test\n")
    lines.append(f"Fit: p_stat = {r['p_stat']:.4f}, p_move = {r['p_move']:.4f}\n")
    lines.append("| K | M | n | mean lost (obs) | var (obs) | var (model) | ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in r["cells"]:
        lines.append(
            f"| {c['k']} | {c['m']} | {c['n']} | {c['mean_obs']:.3f} | "
            f"{c['var_obs']:.3f} | {c['var_exp']:.3f} | {c['ratio']:.2f} |"
        )
    for c in r["anomalies"]:
        lines.append(
            f"| {c['k']} | {c['m']} | {c['n']} | {c['mean_obs']:.3f} | "
            f"{c['var_obs']:.3f} | {c['var_exp']:.3f} | 0.00 ⚠ (deterministic) |"
        )
    lines.append("")
    if r["anomalies"]:
        lines.append(
            "**Anomaly:** cell(s) marked ⚠ lost every atom in every shot (var_obs = 0 "
            "despite n >= 20) -- a *deterministic* failure, not extra scatter. These are "
            "excluded from the pooled test below and worth a separate look.\n"
        )
    lines.append(
        f"**Pooled dispersion test** (excluding anomalies): "
        f"T = {r['T']:.1f}, df = {r['df']}, mean ratio = {r['T'] / r['df']:.3f}, "
        f"chi2 p-value = {r['pval']:.4f}.\n"
    )
    verdict = (
        "consistent with the independence model (ratio close to 1, not significant)."
        if r["pval"] > 0.05
        else "inconsistent with the independence model (significant excess variance)."
    )
    lines.append(f"**Verdict:** the baseline's observed variance is {verdict}\n")

    lines.append("## 2. Does the model explain the test run's losses?\n")
    p = r["prediction"]
    lines.append(
        f"Test run's own fit: p_stat = {r['test_p_stat']:.4f}, p_move = {r['test_p_move']:.4f} "
        f"(for reference only -- the prediction below uses the *baseline's* p_stat/p_move, "
        f"applied to the test run's own per-shot K,M pairs).\n"
    )
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| predicted total atoms lost | {p['pred_total']:.1f} |")
    lines.append(f"| actual total atoms lost | {p['actual_total']:.0f} |")
    lines.append(f"| excess | {p['excess']:.1f} atoms ({p['excess_pct']:.1f}%) |")
    lines.append(f"| z-score | {p['z']:.2f} |")
    lines.append("")
    if abs(p["z"]) > 3:
        lines.append(
            "**Verdict:** the test run lost significantly more than the independence "
            "model predicts from its (K, M) mix alone -- evidence for a real per-shot "
            "cost (e.g. long/complex sequences failing outright) on top of ordinary "
            "per-atom loss.\n"
        )
    else:
        lines.append(
            "**Verdict:** the test run's losses are consistent with the independence "
            "model -- its (K, M) mix alone explains the extra loss, no new mechanism "
            "needed.\n"
        )

    lines.append("## 3. Test run's (K, M) mix\n")
    lines.append("| K | M | n |")
    lines.append("|---|---|---|")
    for (k, m), n in sorted(r["test_km_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {int(k)} | {int(m)} | {n} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline", nargs="*", default=DEFAULT_BASELINE,
        help="substrings selecting the baseline run(s), e.g. 80us_pooled",
    )
    ap.add_argument(
        "--test", nargs="*", default=DEFAULT_TEST,
        help="substrings selecting the test run(s), e.g. sortto0",
    )
    ap.add_argument("--images", default=None, help="override the Sorting1D images folder")
    ap.add_argument("--min-shots", type=int, default=20)
    ap.add_argument(
        "--report", default=None,
        help="path to write a markdown summary (e.g. IndependenceModel_sortto0_Results.md)",
    )
    a = ap.parse_args(argv)

    r = run_analysis(a.baseline, a.test, images=a.images, min_shots=a.min_shots)
    if r is None:
        print("no matching runs -- nothing to analyze")
        return 1

    print_report(r)
    if a.report:
        write_markdown(r, a.report)
        print(f"\nmarkdown report written to {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
