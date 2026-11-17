# Independent-disturbance evaluation: rules fixed before any result (2026-11-11)

Purpose: check whether the reported comparisons (conventional reference against
constrained RL, and the zero-violation observation) hold on disturbance profiles
that neither training nor conventional tuning saw. No training; every controller
is frozen. Simulation evidence only.

## Scenario generator (seed 20261111, 200 candidates drawn in order)
- Influent window: start day uniform on {60, ..., 220} of the benchmark influent
  series (training and the original evaluation use days 0-60 only).
- Load multiplier on particulate substrate: uniform on (0.6, 2.4), the trained
  range (OLR about 1.5-5.9 kg COD/m3/d).
- Initial state: steady state (p = 0.5), prestressed, VFA at 1.5 times steady
  state (0.25), elevated (0.25).
- Event: none (p = 0.5) or one influent spike: magnitude uniform on (1.5, 3.0),
  start day uniform on {5, ..., 45}, duration 2-7 days, composition chosen
  uniformly from lipid, carbohydrate or all particulate fractions.
- Temperature fixed at 35 C, as in the benchmark.
- Horizon 60 days, daily control step, reset seed 7, deterministic actions.

## Feasibility screen
A candidate is kept if at least one screening controller (the dual-loop PI
tunings on the full envelope, or constant feed at 41/80/120/159 m3/d with
multiplier 1.0) runs all 60 days without termination. Kept candidates in draw
order: first 10 = validation, next 30 = test.

## Conventional side
Candidates: every configuration that can be re-run from stored parameters (the
2160-point dual-loop grid, the constant-feed and rule-based configurations, and a
single-loop PI grid). Selection on validation only: for budgets 0, 1, 3, 5 and
10 %, the configuration with the highest pooled methane among those whose
failure-adjusted rate on validation is within the budget. Frozen, then evaluated
on test.

## Learned side (two pre-specified groups, union evaluated once)
- G1: every constrained run with zero violations in the original evaluation.
- G2: every run of the 14 constrained families at d = 0.6, without selection on
  outcome.
Runs that fail on test are kept and counted.

## Metrics on test (pooled over the 30 test scenarios)
Completed-step and failure-adjusted violation rate, mean methane, scenarios with
zero violations; for G1, the share still at zero violations; family medians and
the on-policy/off-policy comparison as in the paper. Reported as descriptive
comparisons with family-then-run resampling; no superiority claim from post-hoc
subgroups.

## Optional
Oracle test envelope: every conventional candidate run on the test set, labelled
as selected on test.
