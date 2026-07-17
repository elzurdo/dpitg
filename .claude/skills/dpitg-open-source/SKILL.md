---
name: dpitg-open-source
description: >
  Use this skill whenever working in the dpitg repo — whether setting up the repo
  structure, porting code from the precision-goal repo, writing unit tests, or
  composing notebooks that reproduce paper figures. The dpitg repo is the
  open-source companion to the paper "Precision and Decisiveness as Goals" (DPitG).
  Trigger on any task involving: creating or editing files in dpitg/, reproducing a
  figure from the paper, writing tests for DPitG/PitG/HDI+ROPE algorithms, or
  setting up the repo (README, requirements.txt, directory structure). Also trigger
  when the user references figure names like cherry_posteriors, fair_experiments,
  conclusive_rates, etc., or function names like BinomialSimulation, BinomialHypothesis,
  successes_failures_to_hdi_ci_limits, stop_decision_multiple_experiments_multiple_methods.
---

# DPitG Open-Source Skill

## Purpose

This skill provides Claude with full context for building the `dpitg` open-source repository,
the companion code release for the paper *"Precision and Decisiveness as Goals: Reliable
Sequential Testing with a Dual Stopping Criterion"* by Eyal A. Kazin.

The dpitg repo allows readers of the paper to reproduce its figures and understand
the algorithms via notebooks composed incrementally. Work is done **one task at a time**
by the user — never chain multiple tasks autonomously. When given a task, do exactly
that task and stop.

## Key Reference: Paper Content

Read `references/paper_summary.md` before starting any task. It contains:
- Paper overview and all three algorithms (HDI+ROPE, PitG, DPitG)
- Full figure inventory with descriptions and the functions that generate them
- The source code map (where everything lives in `../precision-goal/`)

## Repository Structure

Model the dpitg repo on the **are-we-there-yet** repo (`../are-we-there-yet/`) for
structure and user-facing communication (README, requirements.txt, environment setup).

The dpitg repo structure should grow to look like:

```
dpitg/
├── README.md              # Setup instructions and paper overview
├── requirements.txt       # Python dependencies
├── LICENSE
├── .gitignore
├── py/                    # Ported Python modules (from ../precision-goal/py/)
│   ├── utils_stats.py
│   ├── utils_experiments_binomial.py
│   ├── utils_experiments_shared.py
│   └── utils_viz.py
├── notebooks/             # Reproducible notebooks (composed fresh, not copied)
│   └── *.ipynb
└── tests/                 # Unit tests (pytest)
    └── test_*.py
```

Notebooks are **not** copied from `../precision-goal/notebooks/`. Each notebook in dpitg
is composed fresh, purpose-built for a reader who wants to reproduce a specific figure or
explore a specific concept from the paper.

## Source Code Location

All source code originates in `../precision-goal/`. The key files are:

| File | Location | Purpose |
|------|----------|---------|
| `utils_stats.py` | `../precision-goal/py/` | HDI computation, sample-size formula |
| `utils_experiments_binomial.py` | `../precision-goal/py/` | Core algorithms: BinomialSimulation, BinomialHypothesis, BinaryAccounting, stop_decision_*() |
| `utils_experiments_shared.py` | `../precision-goal/py/` | Cross-experiment result aggregation |
| `utils_viz.py` | `../precision-goal/py/` | All binomial visualizations |
| `binomial_experiment_analysis.ipynb` | `../precision-goal/notebooks/` | The original monolithic notebook driving all paper figures — use as a reference to understand what each figure requires |

Always read the relevant source file from `../precision-goal/` before porting or referencing it.

## Python Environment

The target audience is statisticians comfortable with Python. Use a clean virtual
environment setup (modelled on `../are-we-there-yet/README.md`):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt** should pin to the versions used in the paper:
```
numpy>=1.23
pandas>=1.4
matplotlib>=3.5
scipy>=1.9
```

Optionally add `notebook>=6.4` if shipping notebooks. Mention `pytest` separately
in the README (or a `requirements-dev.txt`) for testing.

## Tests

Tests live in `tests/` and use `pytest`. Follow the pattern in `../precision-goal/tests/`
and `../are-we-there-yet/tests/`:
- Tests are deterministic regression tests using handpicked sequences from the paper
- The canonical handpicked sequence is `SEQUENCE_HANDPICKED` in `utils_experiments_binomial.py`
  (the fair coin experiment: HDI+ROPE stops at iter 126, PitG at 598, DPitG at 804)
- Import modules via `sys.path.append` pointing to `../py/` (or the local `py/` once ported)

Run all tests with:
```bash
python -m pytest tests/ -v
```

## Coding Principles

- Target audience: statistician comfortable with Python — no need to over-explain basics
- Keep notebook cells focused and logically ordered following the paper's narrative
- Prefer explicit parameter passing over global state
- When porting from `../precision-goal/py/` to `dpitg/py/`, preserve the logic faithfully;
  note any simplifications made for readability
- Figure-reproducing notebooks should produce the same figure as the paper (or clearly
  labelled variations/extensions)

## BinaryAccounting Cache

`BinaryAccounting` is a memoisation object: it stores HDI results keyed by `(successes, failures)`
so expensive `successes_failures_to_hdi_ci_limits()` calls are never repeated.

### Cache directory convention

Cache files live in `dpitg/cache/`. This directory is git-ignored. Files use `.pkl` extension.
Callers pass a bare filename (e.g. `"fair_coin"`) and the code resolves the full path to
`cache/fair_coin.pkl`.

### Save/load design (pickle, native Python only)

- `BinaryAccounting.load_or_create(filepath)` — `@classmethod` factory. If the file exists,
  load and return a populated object; otherwise return a fresh empty one. Stores `filepath`
  on the instance as `self.filepath`.
- `BinaryAccounting.save()` — saves to `self.filepath` using the size-check rule below.
- `BinaryAccounting.save_as(filepath)` — saves to a new path (updates `self.filepath`).

**What is pickled:** a plain dict `{"version": 1, "hdi_limits": <dict>}`.
The counter dict (`dict_successes_failures_counter`) is **not** saved — it is usage metadata
and roughly doubles file size for no benefit to correctness.

**Atomic save:** write to `filepath.with_suffix(".pkl.tmp")` then `rename()` to final path.

**Size-check rule on save:**
- `entries_new > entries_old` → save silently
- `entries_new == entries_old` → save, print `"Cache unchanged (N entries). Saved."`
- `entries_new < entries_old` → print warning with both counts, prompt `[y/N]`, abort if no

## README Style

Reference `../are-we-there-yet/README.md` for tone and structure. A good dpitg README includes:
1. One-line description of the paper
2. Quick setup (venv + pip install)
3. How to run tests
4. What the notebooks cover and how to launch them
5. Link to the paper / arXiv
