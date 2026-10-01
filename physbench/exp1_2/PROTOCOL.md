# PhysBench-GH EXP1.2 — State-Conditioned Directional Validity + CO2 Gradient-Crossing Analysis

## Status
Frozen follow-up protocol derived from the EXP1.1 observation that cumulative M1/M2 CO2 intervention signs diverged after their baseline trajectories entered different indoor–outdoor CO2 gradient regimes.

## Scientific question
Does the *local* one-step natural-ventilation response obey the exchange-direction rule when the two intervention arms start from exactly the same native model state?

For an exchange-dominated scalar `y`:

`g_y(t) = y_in(t) - y_out(t)`

`Delta_y(t) = y_HIGH(t+900 | x_t) - y_LOW(t+900 | x_t)`

The state-conditioned directional expectation is:

`sign(Delta_y) = -sign(g_y)`.

The primary variable is indoor CO2 concentration. Air temperature and indoor vapor pressure are secondary exchange variables. Relative humidity and canopy temperature are exploratory and are not assigned a simple gradient-rule pass/fail label.

## Why local probes are required
EXP1.1 compared cumulative HIGH and LOW rollouts. After several steps those arms occupy different internal states, so a later HIGH-minus-LOW contrast mixes current actuator response with prior trajectory divergence.

EXP1.2 instead uses matched-state local probes:

1. Generate a BASE trajectory with natural ventilation command = 0.3.
2. At each common time `t = 0, 900, ..., 9900 s`, snapshot the model's *complete native state* from the BASE trajectory.
3. Starting independently from that exact native state, execute one 900-s LOW probe (`vent = 0.1`) and one 900-s HIGH probe (`vent = 0.9`) under the same benchmark forcing interval.
4. Compute HIGH-minus-LOW responses.

Thus LOW and HIGH are matched on the model's full internal state within each model. Full hidden states are **not** claimed identical across M1 and M2.

## Frozen forcing and initial state
Use the EXP0.5 benchmark-owned forcing and matched common observable initial state without modification.

- common grid: 900 s
- benchmark duration for BASE trajectory: 3 h
- outdoor CO2: 415 ppm throughout the frozen forcing trace
- M1 native integration: 900 s
- M2 native integration: 30 s Euler, with results sampled/probed on the 900-s common grid

## Primary endpoint — CO2 conditioned directional consistency (CDC)
For each evaluable local probe:

`expected_CO2_sign(t) = -sign(CO2_in_BASE(t) - CO2_out(t))`

`observed_CO2_sign(t) = sign(CO2_HIGH(t+900) - CO2_LOW(t+900))`

`CDC_CO2 = mean(observed_sign == expected_sign)`

A zero-gradient probe is treated as boundary/indeterminate rather than automatically counted as a failure.

## Secondary endpoints
Apply the same conditioned-direction rule to:

- indoor air temperature vs outdoor temperature;
- indoor vapor pressure vs outdoor vapor pressure.

Report per-variable consistency counts and all raw probe values.

## CO2 gradient-crossing sweep
A separate fixed-state sweep directly tests the reversal boundary.

At benchmark time 0:

- preserve every native hidden state from the matched initial condition;
- preserve T, VP/RH, canopy temperature and all weather variables;
- change **only** indoor CO2 initial concentration;
- run matched one-step LOW/HIGH ventilation probes.

Frozen indoor CO2 grid (ppm):

`355, 375, 395, 405, 410, 412.5, 415, 417.5, 420, 425, 435, 455, 475`

For each model compute:

`Delta_CO2(C0) = CO2_HIGH(900) - CO2_LOW(900)`.

Estimate a response zero crossing `C*` by linear interpolation between adjacent tested concentrations that bracket `Delta_CO2 = 0`. Report:

- `C*`;
- shift from outdoor CO2: `C* - 415 ppm`;
- whether `Delta_CO2(C0)` is monotonically decreasing over the frozen sweep.

The simple pure-exchange reference boundary is `C_in = C_out = 415 ppm`. A model-specific shift of the finite-horizon response boundary is interpreted as coupled-dynamics behavior, not automatically as invalidity.

## Sign classification
For general local probe contrasts use a scale-aware numerical tolerance:

`eps = 1e-8 * max(1, |HIGH|, |LOW|)`.

For gradient classification use the same form with indoor/outdoor values.

For CO2 sweep zero-crossing interpolation, retain the continuous raw HIGH-minus-LOW values; do not snap them to zero except for numerical reporting.

## Runtime audit requirements
Scientific disagreement does **not** fail CI. Runtime fails only if:

1. requested native ventilation is not the applied/consumed ventilation;
2. any required native state becomes non-finite;
3. a probe cannot be started from the intended BASE native state;
4. frozen forcing/initial input hashes do not match the inherited benchmark files.

## Interpretation boundaries
- Natural-ventilation commands are normalized native commands, not calibrated equal airflow doses across M1 and M2.
- Raw response magnitude must not be used to rank the models.
- EXP1.2 tests local directional structure inside the models; it does not establish empirical correctness against real actuator logs.
- The experiment distinguishes **cross-model disagreement** from **state-conditioned physical inconsistency**.
