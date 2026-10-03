# EXP3.4B implementation lock

Frozen before matched-dose model results.

## Inputs

EXP3.4A final artifact from run 37095887502 supplies:
- `evidence/EXP3_4A_FINAL/common_physical_dose_targets_for_EXP3_4B.csv`;
- `evidence/EXP3_4A_M1/physical_dose_grid.csv`;
- `evidence/EXP3_4A_M2/physical_dose_grid.csv`.

The target table SHA256 must equal:
`e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292`.

EXP3.1 final artifact from run 37024060571 supplies the locked old event-level decision metrics used only for the
predeclared paired comparison.

## Coordinates

Primary `DV` maps to model output column:
`specific_volume_dose_m3_m2`.

Secondary `DN` maps to:
`air_volume_equiv`.

Each model must use the exact EXP3.4A physical flux instrumentation already source-locked in EXP3.4A.

## Matching

Each target is solved independently inside the U9 bracket frozen by EXP3.4A.

Initial guess:
piecewise-linear inverse interpolation in dose.

Correction:
safeguarded bisection with monotone bracket updates.

Stopping:
precision-aware target-dose tolerance from PROTOCOL.md and `EXP3_4B_NUMERICAL_MATCH_AMENDMENT_2.md`, maximum 30 iterations.
The best evaluated command by absolute target-dose residual is retained.

If the initial interpolation already satisfies the tolerance, no additional root iterations are required.

M1 representable-command fallback:
`EXP3_4B_NUMERICAL_MATCH_AMENDMENT_3.md` permits a fallback only after the 30-iteration continuous solver fails, only when the final lower/upper commands are adjacent float32 values and their explicitly re-evaluated physical doses bracket the target. The nearer endpoint is accepted only within half of that local physical-dose quantization gap plus the frozen numerical epsilon. No global tolerance is enlarged.

## Decision action

Decision action labels are the shared q values:
`[0,0.25,0.5,0.75,1]`.

Cost:
`[0,0.0625,0.25,0.5625,1]`.

Native command is an implementation variable only and is never used in EXP3.4B decision cost or action-gap metrics.

## Reproducibility

Every model output row stores target dose, achieved dose, solved native command, bracket endpoints, dose error,
nominal tolerance, effective acceptance tolerance, acceptance mode, any local quantization gap, iteration count, T and AH.

Aggregate must independently verify five complete q rows per eligible model/event/horizon/coordinate cell before
decision analysis.
