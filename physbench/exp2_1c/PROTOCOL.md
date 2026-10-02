# EXP2.1C — D2 matched observations vs M1/M2 (FROZEN BEFORE MODEL OUTCOMES)

## Scientific question
For the exact 61-event primary cohort frozen by EXP2.1B, do standardized stronger-natural-ventilation responses from frozen M1 GreenLight-Gym2 and M2 CSGtom agree in direction with the confounding-reduced D2 matched observational contrasts?

This is an observational model-alignment experiment. D2 `VentLee` remains HOLD under the actuator-evidence Gate. The experiment does not validate a realized actuator response, estimate a causal effect, or infer physical ventilation dose from the D2 status values.

## Frozen primary cohort and stratification
- Primary cohort: exactly the 61 EXP2.1B `primary_3ctrl_d5` higher-`VentLee` events.
- Primary analysis: all 61 events at 15 min.
- Pre-declared secondary stratification: `day` vs `night` from the frozen D2 `Iglob=20 W m^-2` proxy. The locked cohort contains 17 day and 44 night events.
- Secondary horizon: 30 min.
- No event is added, removed, or reweighted based on either model output.

## Frozen models
- M1: GreenLight-Gym2 commit `2d3febb1ea002b24b452e32293e990beb78d3ce1`.
- M2: CSGtom commit `bea8c3b0a1324162a4b5487db578aa674c8b587c`.
- No parameter calibration or model-specific retuning to D2 is allowed.

## Intervention probe
The D2 status increase is not mapped to physical airflow or to a model command fraction. Both models receive the same inherited standardized natural-ventilation probe:
- LOW = 0.1 native normalized ventilation command;
- HIGH = 0.9 native normalized ventilation command;
- primary model contrast = HIGH - LOW.

This is a direction probe only. It is not dose matched and model-response magnitude is not interpreted as a quantitative reproduction of the observed status transition.

## Event-specific initial state
The pre-event values are frozen from the EXP2.1B matching vector:
- indoor air temperature = D2 event Tair;
- indoor vapor pressure = deterministic conversion of indoor absolute humidity and Tair;
- indoor CO2 = frozen pre-event D2 CO2air;
- canopy temperature closure = indoor air temperature.

The canopy equality is an explicit neutral closure for an unobserved state, not an empirical measurement. All other non-common model states remain model-native.

## Event-specific forcing
To avoid future-information leakage, each 30-min LOW/HIGH pair uses constant pre-event forcing only:
- global radiation = pre-event Iglob;
- outdoor temperature = pre-event Tout;
- outdoor vapor pressure/RH = deterministic conversion of outdoor absolute humidity and Tout;
- wind = pre-event Windsp in the provider-reported unit;
- outdoor CO2 = fixed 415 ppm because D2 lacks synchronized outdoor CO2;
- sky-temperature proxy = outdoor temperature;
- soil-boundary temperature = fixed 18 °C.

These extension values are benchmark closure assumptions, not D2 measurements. HIGH and LOW always receive identical forcing.

## Native model time handling
- M1 retains the frozen GreenLight equations and 900-s action/integration step. Event day-of-year and local clock are supplied to the wrapper; outputs are evaluated at 15 and 30 min.
- M2 retains the frozen 30-s Euler implementation. Its native solar-geometry machinery is rebuilt at the event month/day/local clock while retaining model-native geometry parameters; outputs are sampled at 15 and 30 min.

## Canonical outputs
Primary variables:
1. indoor air temperature (°C);
2. indoor absolute humidity (g m^-3), deterministically converted from model T and vapor pressure.

CO2 is not primary because synchronized outdoor CO2 is unavailable in D2. RH is not primary because absolute humidity provides the cleaner transported-moisture comparison.

## Event-level quantities
For model m, variable y, horizon h:
`Delta_model = y_HIGH(h) - y_LOW(h)`.

The frozen observational comparator is the EXP2.1B matched contrast:
`Delta_obs = [event post-pre] - mean([matched-control post-pre])`.

The pre-event indoor-outdoor gradient G_y is inherited from EXP2.1B. Define:
- model gradient-aligned score = `-sign(G_y) * Delta_model`;
- normalized model score = model score / `|G_y|`;
- observed normalized score = the already-frozen EXP2.1B normalized matched score.

Positive means movement toward the outdoor state.

## Primary statistics at 15 min
For each model and each primary variable:
1. sign concordance fraction between `Delta_model` and `Delta_obs` among resolved non-zero pairs;
2. calendar-date cluster-bootstrap 95% interval for that fraction (2,000 resamples, seed 20261002);
3. Spearman correlation between normalized model score and frozen observational normalized score, reported as descriptive rank-shape alignment with cluster-bootstrap interval;
4. median normalized model score (descriptive; not dose-comparable to observation).

Numerical sign rules:
- model sign uses inherited relative epsilon `1e-8 * max(1, |HIGH|, |LOW|)`;
- observational sign is unresolved only when `|Delta_obs| <= 1e-12`.

## Secondary analyses
- same metrics at 30 min;
- 15-min day and night strata separately;
- cross-model M1/M2 sign agreement;
- event-level audit tables.

No post-hoc subgroup may replace the all-condition 61-event primary result.

## Runtime fail-closed gates
A model job fails if any event has:
- non-finite runtime state/output;
- requested/applied ventilation mismatch;
- incorrect event count or IDs;
- initial T/VP/CO2/canopy closure mismatch beyond numerical tolerance;
- HIGH and LOW receiving different forcing;
- inability to return both 15- and 30-min outputs.

## Interpretation boundary
Allowed: agreement with matched observational contrast; observational model alignment; standardized ventilation-direction probe; day/night heterogeneity.

Not allowed: causal actuator effect; realized actuator validation; ground-truth intervention response; dose-matched reproduction; ranking one model as globally better from this experiment alone.
