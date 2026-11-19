# ADM1 constrained control benchmark

Code and per-run evaluation records for

> *Reinforcement Learning for Anaerobic Digestion Control under Violation
> Budgets: A Comparison with Extensively Tuned PI and Rule-Based Controllers*

The benchmark compares reinforcement learning with conventional control on a
plant-anchored 38-state ADM1 digester, on the axis operators trade along:
methane flow against the share of time total VFA sits above a soft limit.

- **Conventional reference.** 2490 tuned evaluations of four controller
  families (constant feed, rule-based, single-loop PI and dual-loop PI) are
  reduced to a Pareto upper envelope. Every learned run is scored as its
  methane flow minus the envelope interpolated at the run's own violation
  rate.
- **Learned controllers.** Ten reward-penalty families (four penalty weights
  × ten seeds) and fourteen constrained (CMDP) families (four cost limits ×
  ten seeds).

## Reproducing the tables and figures

The evaluation records ship with the repository, so every table and figure
rebuilds from a clean clone with no training and no GPU:

```bash
git clone https://github.com/radiuson/ADM1-constrained-control.git
cd ADM1-constrained-control
pip install numpy scipy matplotlib       # tables and figures
pip install -e .                          # also needed for the simulation-based scripts

python3 scripts/build_tables.py out/      # main and constrained tables as LaTeX
python3 scripts/plot_figures.py           # fig_frontier, fig_tracking -> figures/
```

`build_tables.py` prints a summary that is the quickest check that a clone is
intact:

```
envelope 2490 evaluations -> 43 points, 0.00-41.9%, best at zero violation 1553
best constrained median PCPO -43 m3/d; closest to specification TRPOPID at 1.22 pp
```

Run the scripts from the repository root. Figures are written to `figures/`.

### Script for each result

| Result | Script |
|---|---|
| Main and constrained tables, envelope | `scripts/build_tables.py` |
| Class comparison and its stratified intervals | `scripts/stratified_stats.py` |
| Variance decomposition (Type I and Type III) | `scripts/variance_decomp.py` |
| Zero-violation comparison | `scripts/zero_violation_gain.py` |
| Algorithm against penalty weight on one scale | `scripts/effect_sizes.py` |
| Hyperparameter sweep table | `scripts/build_ablation_table.py` |
| Weights of 10 and 20, longer training | `scripts/weight_saturation.py`, `scripts/analyse_lowviol.py` |
| Slope of methane against violation rate | `scripts/margin_slope.py` |
| Peak VFA during evaluation | `scripts/evaluation_vfa_peak.py` |
| Off-policy policy stochasticity | `scripts/offpolicy_stochasticity.py` |
| Single-loop PI gain identification | `scripts/recover_pi_gains.py` |
| Auxiliary reward ablation | `scripts/export_figure_data.py`, `scripts/experiment_a_pairs.py` |
| Independent-disturbance evaluation | `scripts/independent_eval_intervals.py`, `scripts/independent_eval_oracle.py`, `scripts/independent_eval_deployable.py`, `scripts/plot_indep.py` |
| Scenario profiles | `scripts/scenario_traces.py`, `scripts/plot_env_figures.py` |
| Frontier, tracking | `scripts/plot_figures.py` |
| Excursion severity | `scripts/plot_severity.py` |
| Reference density | `scripts/plot_density.py` |
| Conventional actuator range | `scripts/plot_range.py` |
| Plant VFA thresholds and correlations | `scripts/plant_thresholds.py`, `scripts/plant_block_bootstrap.py` |
| Simulated against recorded VFA and alkalinity | `scripts/vfa_scale_comparison.py`, `scripts/alkalinity_comparison.py` |
| Discretisation check | `scripts/discretisation_check.py` |

Auxiliary reward ablation:

```bash
python3 scripts/export_figure_data.py figdata/ experiment_A
python3 scripts/experiment_a_pairs.py experiment_A figdata/
```

### Independent-disturbance evaluation

`results_independent/` holds the scenarios, the selected conventional tunings
and the per-scenario test records of the 195 frozen constrained policies;
`PREREGISTRATION.md` gives the rules fixed before any result. The reported
numbers rebuild from these records:

```bash
python3 scripts/independent_eval_intervals.py   # medians, intervals, rank correlation
python3 scripts/independent_eval_oracle.py results_independent
python3 scripts/independent_eval_deployable.py results_independent
python3 scripts/plot_indep.py                   # fig_indep -> figures/
```

`scripts/independent_eval.py` is the full pipeline (scenario generation,
screening, conventional selection, test runs); its learned-policy stage needs
the trained weights, which are not distributed.

### Figures

Every figure is drawn at its printed width (89 or 184 mm) with 8 pt axis labels
and 7 pt text, using the shared settings in `scripts/figstyle.py`.
`scripts/plot_env_figures.py` reads `figures/scenario_traces.json`, written by
`scripts/scenario_traces.py`.

### Plant record

The plant analyses use the openly archived operating series of the Muscatine
Water Resource Recovery Facility (Schroer et al., 2023), released under the
Open Data Commons Attribution Licence v1.0. It is not redistributed here.
Download the archive and point `MUSCATINE_DIR` at its root, the directory that
contains `LAB/LABS-raw.csv`:

```bash
export MUSCATINE_DIR=/path/to/muscatine-archive
python3 scripts/plant_thresholds.py
```

The default is `data/muscatine/` inside the repository.

## Data

| Directory | Records | Contents |
|---|---|---|
| `paper_data/evbase/` | 346 | static conventional configurations (constant feed, rule-based, single-loop PI) |
| `results_dualloop/` | 2160 | dual-loop PI grid |
| `paper_data/evres/` | 713 | reward-penalty runs |
| `paper_data/evres_hp/` | 108 | hyperparameter sweep |
| `paper_data/evres_grad/` | 18 | matched gradient-update runs |
| `paper_data/evcmdp/` | 563 | constrained runs |
| `results_lowviol/` | 168 | penalty weights 10 and 20 |
| `results_b600k/` | 24 | 600k-step training |
| `experiment_A/` | 2 files | auxiliary reward ablation and its paired control |
| `results_independent/` | 195 runs | independent-disturbance evaluation |

Each record carries the run's mean methane flow, its pooled and per-scenario
violation rates and the excursion-severity fields computed by
`evaluation/severity.py`. The filters that select the reported design from
these records are in `scripts/build_tables.py`. `ADM1_PAPER_DATA` points the
scripts at another record set.

## Training and evaluation

Reward-penalty families:

```bash
python3 training/train_sac_std_cur.py \
    --algo ppo --reward-config lw1 --seed 42 \
    --normalize --step-size 1.0 --timesteps 150000 \
    --output-dir models_out
```

`--algo` accepts `sac, tqc, ppo, recurrentppo, trpo, ddpg, td3, a2c, crossq,
ars`. Constrained families are trained with `training/train_omnisafe.py` and
evaluated with `evaluation/eval_cmdp.py`; reward-penalty runs are evaluated
with `evaluation/eval_sb3.py` and the conventional sweep with
`evaluation/eval_baseline_ext.py` and `scripts/sweep_dualloop.py`.
`scripts/export_hyperparams.py` exports the effective hyperparameters of every
family. Trained weights are not included.

The two formulations need separate Python environments, because the
constrained-RL dependency pins an older gymnasium:

| | Reward-penalty | Constrained |
|---|---|---|
| Install | `pip install -e .` | `pip install -e ".[cmdp]"` |
| gymnasium | 1.2.2 | 0.28.1 |
| stable-baselines3 / sb3-contrib | 2.9.0 | — |
| omnisafe | — | 0.5.0 |
| numpy | 2.2.6 | 1.26.4 |

## Layout

```
env/          ADM1 solver, benchmark environment, scenarios, influent data
training/     trainers, reward settings, conventional control laws
evaluation/   evaluation of learned and conventional controllers, severity
scripts/      analyses and figures
paper_data/   per-run evaluation records
results_*/    dual-loop grid and additional training arms
experiment_A/ auxiliary reward ablation
results_independent/ independent-disturbance evaluation
```

## Numerical setup

Each one-day control step is integrated as 96 operator-splitting sub-steps of
15 minutes, with the acid–base equilibrium solved algebraically between them,
using LSODA at `rtol=1e-5`, `atol=1e-7`. Episodes are 60 days. The plant
record sets reactor geometry, feed range and the observation set; the ADM1
kinetic parameters are literature values (`env/PARAMETER_SOURCES.md`).

## License

MIT; see `LICENSE`.
