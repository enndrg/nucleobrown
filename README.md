# nucleobrown
Python project that simulates Brownian motion driven by radioactive decay.

Each decay (α, β, γ) transfers a momentum kick in a random isotropic direction to the particle. The sum of these random kicks produces, at large scales, Brownian motion. This goes beyond a plain random walk: this project models decay chains, a viscous medium with inertia and thermal noise, and validates the simulation against theoretical results.

## Installation

```bash
pip install -e ".[dev]"      # numpy, scipy, matplotlib (+ pytest)
```

## Usage

```bash
python -m nucleobrown chains                                   # available chains
python -m nucleobrown selftest                                 # simulation vs theory
python -m nucleobrown run --config configs/toy_continuous.json --out results/demo
python -m nucleobrown run --config configs/underdamped_thermal.json --workers 4
python -m nucleobrown run --set medium.dim=3 --set medium.gamma=0.5 --set run.n_runs=500
python -m nucleobrown sweep --config configs/toy_continuous.json --param medium.gamma --values 0.5,1,2,4
pytest                                                         # 30 tests
```

Outputs of `run`: `report.json` (all numerical results), `ensemble.npz` (raw data),
`config.json`, and the figures `trajectory`, `msd`, `distributions`, `activity` (+ `vacf` with inertia).

## The model

**Source.** N nuclei (`batch` mode) or a Poisson source of rate R (`continuous` mode, stationary regime).
Each nucleus goes through a chain `A → B → C…`; each stage waits for an exponentially distributed time set by its half-life `T1/2`.
Each decay emits a primary particle (α or β, with branching ratios) and possibly a γ.
Times are sampled exactly (no time step).

**Kicks.** Direction uniform on the sphere (1D/2D/3D); fixed magnitude per particle type, or a continuous spectrum for β (`p·Beta(2,3)`).

**Medium.** Langevin equation with kicks: `m dv = -mγ v dt + √(2mγ kT) dW + Σ pⱼ δ(t-tⱼ)`.
Between events the solution is **exact** (integrated Ornstein-Uhlenbeck), so there is no discretization error, and
the computation is fully vectorized (the recurrence is solved with a block-wise cumulative sum).

* `overdamped`: a kick `p` displaces the particle by `p/(mγ)` (this is the model of the original script); thermal noise is ordinary Brownian motion with `D = kT/(mγ)`.
* `underdamped`: full inertia. With `γ = 0` the velocity performs a random walk and diffusion becomes **anomalous** (MSD ∝ t³).

## Analyses

Ensemble MSD with standard error, power-law fit (exponent α) and linear fit (coefficient D) with
**bootstrap** confidence intervals over trajectories, local exponent, time-averaged MSD and ergodicity-breaking
parameter (EB), non-Gaussianity parameter α₂, normality tests, velocity autocorrelation, Fano factor
of the counts.

## Validation (`selftest` and `pytest`)

| Check | Expected result |
|---|---|
| Single stage + Poisson source | Fano = 1 |
| Parent→daughter chain | Fano > 1 (correlated counts) |
| Fixed number N of nuclei | Fano < 1 (binomial) |
| MSD in batch mode | = exact formula via matrix exponential |
| D in continuous mode | = `R·Σ E[p²] / (2d(mγ)²)` |
| No decays, with kT | D = kT/(mγ) (Einstein relation) |
| Exact OU | MSD = `2d·vth²/γ²·(γt − 1 + e^{−γt})`, ⟨vᵢ²⟩ = kT/m |
| γ = 0 | exponent α = 3 |

Comparisons are expressed in standard deviations (z-scores, bootstrap) with fixed seeds, so results are reproducible.

## Structure

```
nucleobrown/
  config.py      typed configuration + validation + overrides
  isotopes.py    isotopes and chains (toy_*, rn222, custom JSON chains)
  decay.py       exact generation of decay events
  kicks.py       momentum kicks (direction, β spectrum, second moments)
  dynamics.py    exact, vectorized Langevin propagation
  simulation.py  single trajectory, ensemble, multiprocessing (SeedSequence)
  theory.py      exact predictions (absorbing Markov chain, D, VACF)
  analysis.py    statistics, fits, bootstrap
  plotting.py    figures
  selftest.py    physical validation
  cli.py         command line
```

## Limitations
blablabla

## Outlook
The idea is to, in the end, implement a similar code to utilize the "Quantum aspect" of Quantum computing, to compare such an approach to one based on physical decay.
