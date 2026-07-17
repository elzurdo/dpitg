# DPitG Paper Reference

Paper: *"Precision and Decisiveness as Goals: Reliable Sequential Testing with a Dual Stopping Criterion"*  
Author: Eyal A. Kazin  
Source: `../precision-goal/latex/dpitg.tex` and section files in the same directory.

---

## Overview

The paper proposes **DPitG (Decisive Precision is the Goal)**, a sequential hypothesis
testing algorithm that requires *both* a posterior precision target *and* a conclusive
hypothesis verdict to be satisfied simultaneously before stopping.

It is compared against two baselines:
- **HDI+ROPE**: stops as soon as a conclusive decision is reached (coupled stopping/decision)
- **PitG (Precision is the Goal)**: stops when precision target is met, then applies the decision rule (decoupled)

The key result (fair coin, ω_goal=0.08, ROPE=[0.45,0.55]):
- HDI+ROPE: 98.2% conclusive, but 6.2% false-positive rate
- PitG: 0% false positives, but only 36.7% conclusive
- DPitG: 97.7% conclusive, 0% false positives, at only 4.7% median extra samples vs PitG

---

## Shared Components (all three algorithms use these)

### ROPE (Region of Practical Equivalence)
- Pre-specified interval around the null: `[ROPE_min, ROPE_max]`, width `Δ_ROPE`
- Default in paper: `θ_null=0.5`, `ROPE=[0.45, 0.55]`, `Δ_ROPE=0.1`
- Encodes practical equivalence — effect must exceed this to matter

### HDI (Highest Density Interval)
- 95% Bayesian credible interval of the posterior
- Width: `ω_HDI = HDI_max - HDI_min`
- For Bernoulli data with conjugate Beta prior: fully analytical

### Decision Rule (shared by all three algorithms)
Given `ROPE=[ROPE_min, ROPE_max]` and `HDI=[HDI_min, HDI_max]`:
- **Accept null**: HDI fully inside ROPE → `ROPE_min ≤ HDI_min` AND `HDI_max ≤ ROPE_max`
- **Reject null**: HDI fully outside ROPE → `HDI_max < ROPE_min` OR `ROPE_max < HDI_min`
- **Inconclusive**: HDI straddles ROPE boundary

---

## Three Algorithms

### 1. HDI+ROPE
- **Stop criterion**: Decision is conclusive (HDI fully in or fully out of ROPE)
- **Key input**: `N_min` (minimum sample size, guards against aleatoric noise; default 30)
- **Risk**: susceptible to early peeking → systematic false positives

### 2. PitG (Precision is the Goal) — Kruschke 2015
- **Stop criterion**: `ω_HDI ≤ ω_goal`
- **Key input**: `ω_goal` (precision target; typically `0.8 × Δ_ROPE`)
- **Risk**: high inconclusive rate when null is true

### 3. DPitG (Decisive Precision is the Goal) — proposed in this paper
- **Stop criterion**: `ω_HDI ≤ ω_goal` AND decision is conclusive
- **Key input**: `ω_goal` (same as PitG)
- Continues collecting data beyond `ω_goal` until both conditions are met (up to `N_max`)

All three share: `N_max` (budget cap) and `ROPE` boundaries.

---

## Planning Formula

For Bernoulli rate parameter θ:

```
N_goal(θ, ω_goal) ≈ (4 × z*²) / ω_goal² × θ(1-θ)
```

where `z* ≈ 1.96` for 95% credibility.

- Maximised at θ=0.5 (fair coin is hardest case)
- `N_goal(θ_null)` is the researcher's prospective planning estimate
- When `θ_null=0.5`, it is a conservative upper bound on actual `N_stop`

---

## Figure Inventory

Each figure below: name used in the paper (= image filename without path), description,
and the function(s) in `utils_viz.py` that produce it.

### Figure 1 — `cherry_posteriors.png`
Three vertically stacked panels showing posteriors at three specific stop iterations
of the hand-picked experiment, illustrating all three decision outcomes:
- Top (iter 126): HDI fully outside ROPE → Reject
- Middle (iter 598): HDI straddles ROPE → Inconclusive  
- Bottom (iter 804): HDI fully inside ROPE → Accept
**Source function**: `viz_sequence_stats()` in `utils_viz.py`

### Figure 2 — `cherry_iterations.png`
Cumulative sample proportion `θ̂(N)` vs iteration for the hand-picked fair coin experiment.
Shows: grey HDI band narrowing, orange ROPE dashed lines, red vertical dashed line at
HDI+ROPE stop (iter 126), purple dots where precision goal first met (from iter 598),
green vertical lines where DPitG accepts (from iter 804).
**Source function**: `plot_multiple_decision_rates_separate()` or similar in `utils_viz.py`
(check the notebook for the exact call)

### Figure 3 — `min_sample_by_goal.png`
Four concave curves of `N_goal(θ, ω_goal)` vs θ ∈ [0,1], one per ω_goal value
(blue: 0.10, orange: 0.08, green: 0.06, red: 0.04). All peak at θ=0.5.
**Source function**: `plot_n_goal_by_parameter()` in `utils_viz.py`

### Figure 4 — `fair_experiments_iter_vs_rate.png`
Three-panel figure (scatter + marginal histograms) for M=2000 fair coin experiments.
Main panel: `θ̂(N_stop)` vs `N_stop`. Marginal panels: distributions of each.
Symbols: red squares (HDI+ROPE), blue circles (PitG), green Xs (DPitG).
**Source function**: `scatter_stop_iter_sample_rate()` in `utils_viz.py`

### Figure 5 — `fair_experiment_decision_rates.png`
Three side-by-side panels (one per algorithm) showing cumulative decision rates
(accept / reject / inconclusive) as a function of iteration, for M=2000 fair coin experiments.
**Source function**: `plot_multiple_decision_rates_separate()` in `utils_viz.py`

### Figure 6 — `conclusive_rates.png`
Three side-by-side panels across `θ_true ∈ [0.5, 0.7]`:
- Left: conclusiveness rates (all 3 algorithms)
- Middle: acceptance (solid) and rejection (dashed) rates
- Right: accuracy rates
Colors: red (HDI+ROPE), blue (PitG), green (DPitG). Vertical dashed line at ROPE_max=0.55.
**Source function**: `plot_conclusiveness_decisions_and_correctness_rates()` in `utils_viz.py`

### Figure 7 — `success_by_truth_conclusive.png`
Two panels (IQR and median bias of `θ̂(N_stop)` vs `θ_true`) for conclusive experiments only.
**Source function**: `plot_success_by_truth_absolute_and_diff()` in `utils_viz.py`

### Figure 8 — `stop_iterations_by_truth_conclusive.png`
3×2 grid: rows = HDI+ROPE / PitG / DPitG, columns = `θ_null=0.5` and `θ_null=0.7`.
Shows IQR band and median of `N_stop` vs `θ_true` for conclusive experiments.
**Source function**: `plot_stop_iterations_by_truth_two_panel()` in `utils_viz.py`

### Figure 9 — `stop_conclusiveness_ratios.png`
Single panel showing DPitG/PitG ratios: conclusiveness rate (solid) and median `N_stop`
(dot-dashed) across `θ_true`. IQR band for the `N_stop` ratio.
**Source function**: `plot_stop_and_conclusive_ratios()` in `utils_viz.py`

---

## Source Code Map

### `../precision-goal/py/utils_stats.py`
Key functions:
- `HDIofICDF(dist, credMass=0.95)` — computes HDI for a distribution via inverse CDF
- `successes_failures_to_hdi_ci_limits(successes, failures, ci_fraction=CI_FRACTION)` — returns `(hdi_min, hdi_max)` for Beta posterior
- `binomial_rate_ci_width_to_sample_size(theta, omega_goal, ci_fraction=0.95)` — computes `N_goal` via the planning formula
- `CI_FRACTION = 0.95` — global constant

### `../precision-goal/py/utils_experiments_binomial.py`
Key classes and functions:
- `SEQUENCE_HANDPICKED` — the canonical deterministic sequence used in the paper's figures (fair coin, 1500 trials). HDI+ROPE stops at 126, PitG at 598, DPitG at 804.
- `BinaryAccounting` — caches HDI computations keyed by (successes, failures) to avoid redundant calculation
- `BinomialHypothesis(success_rate_null, dsuccess_rate, rope_precision_fraction)` — encodes ROPE and ω_goal
- `BinomialSimulation(sequence, hypothesis, binary_accounting)` — runs one simulation sequence
- `stop_decision_multiple_experiments_multiple_methods(sequences, hypothesis, binary_accounting, methods)` — core function: runs M experiments through all three algorithms, returns nested dict of results
- `run_simulations_and_analysis_report(...)` — high-level wrapper used in the notebook

### `../precision-goal/py/utils_experiments_shared.py`
Key functions:
- `sims_hypo_dict_to_algo_stats_dfs(...)` — converts simulation result dicts to DataFrames
- `sims_hypo_to_correctness_stats(...)` — computes accuracy/error statistics

### `../precision-goal/py/utils_viz.py`
All visualization functions (1,187 lines). Key ones mapped to figures above.
Also: `ALGO_COLORS` — color dict keyed by algorithm name.

---

## Simulation Configuration (paper defaults)

```python
SEED = 42
M = 2000           # number of independent experiments
N_max = 1500       # maximum sample size
N_min = 30         # HDI+ROPE minimum (guards against early aleatoric noise)
theta_null = 0.5
dsuccess_rate = 0.05
rope_precision_fraction = 0.8   # ω_goal = 0.8 × Δ_ROPE
# → ROPE = [0.45, 0.55], ω_goal = 0.08
```

---

## Key Numerical Results (fair coin, default params)

| Algorithm | Conclusive Rate | False-Reject Rate | Median N_stop |
|-----------|----------------|-------------------|---------------|
| HDI+ROPE  | 0.982          | 0.062             | 523           |
| PitG      | 0.367          | 0.000             | 599           |
| DPitG     | 0.977          | 0.000             | 627           |

Sensitivity to ω_goal (θ_true=0.5, M=2000, N_max=1500):

| ω_goal | N_goal | PitG conclusive | DPitG conclusive |
|--------|--------|-----------------|------------------|
| 0.06   | 1066   | 0.825           | 0.972            |
| 0.07   | 783    | 0.594           | 0.975            |
| 0.08   | 599    | 0.367           | 0.977            |
| 0.09   | 473    | 0.127           | 0.978            |
| 0.10   | 383    | 0.000           | 0.979            |
