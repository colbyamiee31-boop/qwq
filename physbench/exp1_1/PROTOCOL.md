# PhysBench-GH EXP1.1 — Matched Natural-Ventilation Directional Response

## Scientific question
Under a matched common observable initial state and the same benchmark-owned external forcing, do M1 GreenLight-Gym2 and M2 CSGtom predict the same **direction** of response to a stronger natural-ventilation command?

## Frozen upstream gates
EXP1.1 inherits EXP0.2–EXP0.5 without alteration:
- M1 frozen commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 frozen commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`
- common grid: 900 s
- benchmark forcing: `physbench/exp0_5/benchmark_forcing.csv`
- matched common observable initial state: `physbench/exp0_5/matched_initial_state.json`

## Intervention arms
All arms start at t=0 from the same matched observable initial state and receive the same forcing.

- LOW: `ventilation_command_fraction = 0.1`
- BASE: `ventilation_command_fraction = 0.3`
- HIGH: `ventilation_command_fraction = 0.9`

The primary finite-difference intervention contrast is:

`R_y(t) = y_HIGH(t) - y_LOW(t)`

The BASE arm is retained for audit and gradient interpretation, not for selecting the primary result post hoc.

## Common outputs
- `air_temperature_c`
- `air_vapor_pressure_pa`
- `air_rh_pct`
- `air_co2_ppm`
- `canopy_temperature_c`

## Primary directional endpoints
Primary horizon: **900 s (15 min)**.

Primary variables with a pre-specified physical direction:
- air temperature
- air vapor pressure
- air CO2

At t=0 the matched indoor state exceeds the outdoor forcing for all three variables. Therefore the pre-specified one-step expectation for a stronger natural-ventilation command is:

- `R_T(900) < 0`
- `R_VP(900) < 0`
- `R_CO2(900) < 0`

RH is not assigned a pre-specified sign because simultaneous changes in temperature and vapor pressure can move RH in either direction. Canopy temperature is exploratory because it is not a directly ventilated state.

## Secondary trajectory endpoints
The same HIGH-minus-LOW contrast is reported at 30, 45, 60, 90, 120, 150 and 180 min. These are trajectory-persistence results and must not be over-interpreted as independent physical guarantees.

For secondary temperature/VP/CO2 interpretation, the BASE arm at the preceding common timestamp is used only as a diagnostic of the prevailing indoor–outdoor gradient. This diagnostic must not redefine the primary endpoint.

## Numerical sign rule
For a contrast d with characteristic scale s = max(1, |HIGH|, |LOW|):

- positive if `d > 1e-8*s`
- negative if `d < -1e-8*s`
- neutral otherwise

This prevents machine-level floating-point noise from creating a sign.

## Primary scores
For each model:
1. one-step physics-direction consistency for T/VP/CO2 (0–3 passes);
2. exact requested-versus-applied native ventilation audit;
3. finite-state runtime audit.

Across models:
1. one-step sign agreement for each common output;
2. secondary sign-agreement matrix over the full 3-h trajectory.

## Interpretation firewall
This experiment does NOT compare raw response magnitude as a model-quality ranking because `0.1` and `0.9` are normalized native commands, not calibrated equal airflow doses across M1 and M2.

A directional mismatch is evidence of cross-model interventional disagreement under this benchmark case. It is not, by itself, proof that either model is physically wrong; empirical and boundary evidence are required for that stronger claim.
