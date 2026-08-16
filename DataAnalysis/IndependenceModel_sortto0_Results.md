# Independence-Model Test: Baseline Dispersion & sortto0 Prediction

Source: `IndependenceModelTest.py`, baseline=`tweezerLoad1x11-80us_pooled` (575 shots), test=`tweezerLoad1x11-sortto0` (209 shots).

Eq. (6.3) independence model: each atom survives as an independent Bernoulli draw, `p_stat` for stationary atoms and `p_move` for movers. Both stories (independent per-atom loss vs. occasional whole-shot wipeout) give the same *mean* loss -- only the *variance* tells them apart.

## 1. Baseline dispersion test

Fit: p_stat = 0.7570, p_move = 0.7072

| K | M | n | mean lost (obs) | var (obs) | var (model) | ratio |
|---|---|---|---|---|---|---|
| 1 | 0 | 28 | 0.107 | 0.096 | 0.184 | 0.52 |
| 2 | 0 | 12 | 0.250 | 0.188 | 0.368 | 0.51 |
| 3 | 0 | 10 | 0.300 | 0.410 | 0.552 | 0.74 |
| 3 | 1 | 50 | 1.120 | 0.786 | 0.575 | 1.37 |
| 3 | 2 | 54 | 1.037 | 0.702 | 0.598 | 1.17 |
| 4 | 1 | 32 | 0.719 | 1.077 | 0.759 | 1.42 |
| 4 | 2 | 70 | 0.886 | 0.958 | 0.782 | 1.23 |
| 4 | 3 | 19 | 0.421 | 0.560 | 0.805 | 0.69 |
| 5 | 1 | 15 | 0.533 | 0.516 | 0.943 | 0.55 |
| 5 | 2 | 36 | 0.556 | 0.469 | 0.966 | 0.49 |
| 5 | 3 | 66 | 0.773 | 0.933 | 0.989 | 0.94 |
| 5 | 4 | 5 | 1.000 | 1.600 | 1.012 | 1.58 |
| 6 | 2 | 10 | 1.400 | 1.440 | 1.150 | 1.25 |
| 6 | 3 | 27 | 1.407 | 1.501 | 1.173 | 1.28 |
| 6 | 4 | 14 | 0.786 | 1.168 | 1.196 | 0.98 |
| 7 | 2 | 6 | 2.000 | 2.667 | 1.334 | 2.00 |
| 7 | 3 | 8 | 1.875 | 0.859 | 1.357 | 0.63 |
| 7 | 4 | 14 | 1.643 | 1.515 | 1.380 | 1.10 |
| 8 | 4 | 5 | 4.200 | 2.960 | 1.564 | 1.89 |
| 2 | 1 | 60 | 2.000 | 0.000 | 0.391 | 0.00 ⚠ (deterministic) |

**Anomaly:** cell(s) marked ⚠ lost every atom in every shot (var_obs = 0 despite n >= 20) -- a *deterministic* failure, not extra scatter. These are excluded from the pooled test below and worth a separate look.

**Pooled dispersion test** (excluding anomalies): T = 482.4, df = 462, mean ratio = 1.044, chi2 p-value = 0.2470.

**Verdict:** the baseline's observed variance is consistent with the independence model (ratio close to 1, not significant).

## 2. Does the model explain the test run's losses?

Test run's own fit: p_stat = 0.4851, p_move = 0.4544 (for reference only -- the prediction below uses the *baseline's* p_stat/p_move, applied to the test run's own per-shot K,M pairs).

| quantity | value |
|---|---|
| predicted total atoms lost | 271.4 |
| actual total atoms lost | 374 |
| excess | 102.6 atoms (37.8%) |
| z-score | 7.29 |

**Verdict:** the test run lost significantly more than the independence model predicts from its (K, M) mix alone -- evidence for a real per-shot cost (e.g. long/complex sequences failing outright) on top of ordinary per-atom loss.

## 3. Test run's (K, M) mix

| K | M | n |
|---|---|---|
| 4 | 2 | 33 |
| 5 | 3 | 28 |
| 3 | 2 | 13 |
| 3 | 1 | 12 |
| 4 | 3 | 12 |
| 5 | 2 | 12 |
| 6 | 3 | 11 |
| 7 | 3 | 9 |
| 6 | 2 | 9 |
| 2 | 1 | 8 |
| 7 | 4 | 7 |
| 6 | 4 | 7 |
| 4 | 1 | 5 |
| 1 | 0 | 5 |
| 7 | 2 | 4 |
| 8 | 3 | 4 |
| 5 | 1 | 4 |
| 5 | 4 | 4 |
| 7 | 1 | 3 |
| 8 | 4 | 3 |
| 3 | 0 | 2 |
| 2 | 0 | 2 |
| 8 | 5 | 2 |
| 6 | 1 | 2 |
| 9 | 2 | 2 |
| 8 | 2 | 2 |
| 9 | 3 | 1 |
| 6 | 0 | 1 |
| 9 | 4 | 1 |
| 7 | 5 | 1 |
