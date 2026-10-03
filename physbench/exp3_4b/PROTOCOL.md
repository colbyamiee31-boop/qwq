# PhysBench-GH EXP3.4B — Physical-Dose-Matched 61-Event Decision Consequence

## Status

Frozen before any EXP3.4B matched-dose decision result is inspected.

EXP3.4B asks whether the model-dependent decision differences seen under the old equal-native-command comparison
persist after both frozen greenhouse models are constrained to the same physically reachable ventilation dose.

No model is designated as truth. No observed D2 controller-status field is treated as realised airflow.

## Frozen ancestry

Repository: `colbyamiee31-boop/qwq`

Parent EXP3.4A commit:
`c25a549e654d1cfbec3d7d73886857c4e3819041`

Frozen models:
- M1 GreenLight-Gym2: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

Frozen 61-event input:
- `physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv`
- SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`

Frozen EXP3.4A hosted result:
- workflow run: `37095887502`
- final artifact: `11264407107`
- final artifact SHA256:
  `226e92ee213f9586a40f032f1b88edab4ab27bacef74af176b97a8120c33c47a`

Frozen EXP3.4B target table:
- `common_physical_dose_targets_for_EXP3_4B.csv`
- SHA256:
  `e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292`
- q levels: `{0,0.25,0.50,0.75,1.00}`

Locked EXP3.1 decision reference:
- workflow run: `37024060571`
- final artifact: `11234651046`
- final artifact SHA256:
  `09ab52edbdff6ffc8da24c087a528c790dbcc366fa34a4ac7a61bb23e9342af8`

## Scientific question

For the exact event/horizon/coordinate cells declared physically matchable by EXP3.4A:

1. can M1 and M2 be driven to the same target physical ventilation dose without extrapolation?
2. when decisions are made over a common matched-dose grid rather than a common native-command grid, do the two models
   still choose different ventilation doses?
3. how much of the original EXP3.1 disagreement survives after physical-dose matching?
4. what symmetric model-swap consequence remains when the selected physical doses differ?

## Cohorts and hierarchy

Primary analysis:
- physical coordinate: `DV`, cumulative external specific ventilation volume [m3 m-2];
- horizon: 15 min;
- cohort: exactly the 42 EXP3.4A cells marked `eligible_for_exp3_4b=True`;
- no extrapolation outside each model's native command interval `[0.1,0.9]`.

Secondary analyses:
- `DN` at 15 min: exactly 61/61 eligible events;
- `DV` at 30 min: exactly 48/61 eligible events;
- `DN` at 30 min: exactly 61/61 eligible events;
- primary DV 15-min day/night strata;
- secondary penalty domain `lambda in [0,3]`.

The 19 primary-DV no-overlap events are not imputed, extrapolated, or replaced.

## Frozen common physical action grid

For every eligible event i, horizon h, and physical coordinate c, EXP3.4A froze:

`q in {0,0.25,0.50,0.75,1.00}`

and

`D_target(i,h,c,q) = L + q*(H-L)`.

EXP3.4B treats **q**, not native model command, as the decision-action index.

The physical target values `D_target` cannot be changed.

## Native-command inversion

For each model, event, horizon, coordinate, and target dose:

1. use the frozen EXP3.4A U9 native command-dose grid
   `u={0.1,0.2,...,0.9}`;
2. find the adjacent U9 commands that bracket the target;
3. if the target equals a frozen endpoint within tolerance, use that endpoint;
4. otherwise linearly interpolate within the bracket to obtain the initial command;
5. run the frozen model at that command and measure the achieved physical dose with the same EXP3.4A instrumentation;
6. if the dose error exceeds tolerance, perform safeguarded monotone bisection inside the original U9 bracket;
7. never leave `[0.1,0.9]`;
8. never use a non-eligible EXP3.4A cell.

Maximum bisection iterations: 30.

Dose-match tolerance for coordinate x:

`tol = max(1e-8, 1e-6 * max(1, |D_target|, |H-L|))`.

A target is accepted only when:

`|D_achieved - D_target| <= tol`.

Also report the cross-model achieved-dose difference at every q.

No isotonic regression, smoothing, re-calibration, or post-hoc target adjustment is permitted.

## Event initialization and forcing

Exactly the same as EXP3.1 and EXP3.4A:
- indoor T = event `event_Tair`;
- indoor vapour pressure = `in_vp_pa`;
- indoor CO2 = `CO2_pre_ppm`;
- canopy T = event air T;
- all other hidden states remain model-native;
- constant pre-event forcing snapshot;
- outdoor CO2 = 415 ppm;
- frozen sky-temperature and soil-boundary representations.

## Matched-dose outcome table

For every accepted matched target record:

- model id;
- event id / date / day-night stratum;
- horizon;
- physical coordinate;
- q;
- target physical dose;
- solved native command;
- achieved physical dose;
- dose error;
- T outcome;
- AH outcome;
- frozen T and AH gradients.

The scientific outcome is generated from the native model trajectory at the solved native command.
The dose instrumentation may observe the trajectory but may not replace or perturb the native T/AH result.

## Matched-dose standardized benefit

For each event/model/coordinate/horizon and q, use the lowest common dose `q=0` as the reference:

`b_T(q) = -sign(G_T) * [T(q)-T(0)] / |G_T|`

`b_AH(q) = -sign(G_AH) * [AH(q)-AH(0)] / |G_AH|`

`B(q) = 0.5*b_T(q) + 0.5*b_AH(q)`.

The gradients are the same frozen D2 pre-event gradients used by EXP3.1.

Positive benefit means model-predicted movement toward the outdoor state relative to the lowest common physical dose.

## Intervention cost and decision rule

Because native commands are intentionally no longer comparable, intervention cost is defined only on the common
physical action index:

`C(q)=q^2`.

For penalty ratio lambda:

`J(q;lambda)=B(q)-lambda*C(q)`.

The chosen action is the **physical dose level q**, not the native actuator command.

Tie rule: if utilities are equal within `1e-12`, choose the lower q.

Primary penalty domain:
`lambda in [0,1]`.

Secondary domain:
`lambda in [0,3]`.

Decision regions are integrated exactly using analytic pairwise line intersections, as in EXP3.1.

## Primary metrics

For DV, 15 min, exact 42-event matched cohort, lambda in [0,1]:

1. decision-disagreement measure over lambda;
2. integrated normalized physical-action gap `|q_M1-q_M2|`;
3. symmetric cross-model normalized regret;
4. symmetric predicted T/AH outcome divergence.

Report mean, median, date-cluster bootstrap 95% interval for the mean, and fraction of events with any disagreement.

Bootstrap:
- date clusters;
- 2000 resamples;
- seed `20261002`.

## Predeclared comparison with EXP3.1

To isolate the effect of command-coordinate mismatch, compare EXP3.4B against the locked EXP3.1 result on the **same cohort**.

Primary comparison:
- restrict old EXP3.1 15-min lambda[0,1] event metrics to the exact 42 primary-DV events;
- compute paired event-level change:
  `Delta = matched-dose metric - old equal-command metric`;
- report mean/median paired Delta and date-cluster bootstrap 95% interval for mean Delta.

Secondary DN comparison:
- compare the 61-event 15-min DN matched result with the original 61-event EXP3.1 result.

No all-61 EXP3.1 summary may be contrasted numerically with the 42-event DV primary result without also reporting the
same-42 restricted EXP3.1 baseline.

## Locked prior disagreement events

Events `27, 86, 89` are predeclared before EXP3.4B outcomes because they were the three EXP3.1 primary events with
any disagreement and all three have positive primary DV overlap in EXP3.4A.

For each, report:
- matched native commands at all five q levels;
- achieved DV match error;
- matched-dose decision partitions over lambda[0,1];
- whether any disagreement survives;
- matched decision-disagreement measure.

These events are descriptive traces and do not replace the 42-event primary cohort.

## Runtime fail-closed gates

Fail if:
- the EXP3.4A target SHA differs;
- the EXP3.4A U9 dose grids are missing or not monotone;
- event identity or eligible cohort count differs from the frozen counts;
- a target is outside its frozen bracket;
- a solved command leaves [0.1,0.9];
- requested/applied native command differs;
- any model state/output/dose is non-finite;
- any accepted target exceeds the predeclared dose-match tolerance;
- a cell has fewer or more than five q targets;
- decision partition does not cover its full lambda domain within 1e-12.

No-overlap cells remain excluded by design and are not runtime failures.

## Interpretation boundary

EXP3.4B can establish whether model-dependent decision differences persist, disappear, or change magnitude after
physical-dose matching within the common reachable intervention space.

It cannot establish:
- which model is correct;
- real-world optimal ventilation;
- measured actuator airflow in D2;
- causal controller superiority;
- model ranking.

A reduction in disagreement after matching supports the interpretation that native actuator scaling contributed to
the earlier discrepancy. Residual disagreement supports a model-response difference within a common physical
intervention domain. Both outcomes are scientifically admissible.
