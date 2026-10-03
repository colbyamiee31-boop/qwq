# PhysBench-GH EXP3.4B — Physical-Dose-Matched 61-Event Decision Consequence

## Status

**PROTOCOL FREEZE — frozen before any EXP3.4B matched-dose decision outcome is generated or inspected.**

EXP3.4B is the direct continuation of EXP3.4A. It does not redesign manuscript positioning and does not reinterpret
observed controller status as realised airflow. It asks whether the locked EXP3.1 decision disagreement persists when
the two frozen models are compared at the same predeclared physical ventilation dose rather than at the same numerical
native command.

## Frozen ancestry and immutable inputs

Repository: `colbyamiee31-boop/qwq`

Parent EXP3.4A FINAL LOCK:
- branch: `exp3-4a-physical-ventilation-dose`
- commit: `c25a549e654d1cfbec3d7d73886857c4e3819041`
- workflow run: `37095887502`
- final artifact id: `11264407107`
- final artifact SHA256: `226e92ee213f9586a40f032f1b88edab4ab27bacef74af176b97a8120c33c47a`

Frozen EXP3.4B target file:
- `common_physical_dose_targets_for_EXP3_4B.csv`
- SHA256: `e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292`
- q levels: `{0, 0.25, 0.50, 0.75, 1.00}`

Frozen 61-event input:
- `physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv`
- SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`

Frozen models:
- M1 GreenLight-Gym2: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

Locked EXP3.1 reference:
- commit: `8c68e2262f9cbaffdc1a14a2309e9fdf7644df11`
- workflow run: `37024060571`
- final artifact id: `11234651046`
- final artifact SHA256: `09ab52edbdff6ffc8da24c087a528c790dbcc366fa34a4ac7a61bb23e9342af8`
- primary disagreement events: `{27, 86, 89}`.

No target dose, cohort membership, model commit, utility weight, intervention-cost form, lambda domain, tie rule,
matching tolerance, or scientific endpoint may be changed after matched-dose outcomes are viewed.

## Confirmatory scope

Primary physical coordinate: **DV**

`DV = integral(q_ext dt)` [m3 m-2].

Primary horizon: **15 min**.

Primary DV cohort: exactly the **42/61** events marked `eligible_for_exp3_4b=True` by EXP3.4A for `DV, 15 min`.
The 19 no-overlap events are not extrapolated and are not silently substituted with another coordinate.
Events 27, 86 and 89 are tracked individually as the pre-locked subgroup.

Secondary physical coordinate: **DN**

`DN = DV / H_eff` [air-volume equivalents].

Secondary horizon: **15 min**.

Secondary DN cohort: exactly **61/61** events, using the frozen EXP3.4A DN targets.
DN remains secondary regardless of outcome direction.

The 30-min EXP3.4A targets are not part of the confirmatory EXP3.4B decision result. This avoids adding a new horizon
family after the central action-equivalence question was already defined.

## Event initialization and forcing

Use exactly the EXP3.1 / EXP3.4A event representation:
- indoor T = `event_Tair`;
- indoor vapour pressure = `in_vp_pa`;
- indoor CO2 = `CO2_pre_ppm`;
- canopy T = event air T;
- all remaining states = frozen model-native reset values;
- constant pre-event forcing snapshot;
- outdoor CO2 = 415 ppm;
- sky-temperature proxy and soil-boundary value exactly as frozen in the 61-event table.

EXP0.6C latent histories are not propagated into EXP3.4B because latent-state robustness and action-coordinate
comparability are separate locked questions.

## Dose inversion rule

For each eligible event, coordinate and frozen target q:

1. Read the corresponding EXP3.4A U9 monotone command-to-dose curve for that model and horizon.
2. Locate the adjacent native command bracket in `u={0.1,...,0.9}` containing the frozen target.
3. Solve inside that bracket using a safeguarded secant/bisection search on **new model executions**.
4. Never evaluate or accept a native command outside `[0.1,0.9]`.
5. The final T/AH outcome is taken from the actual model execution at the solved command. Outcome-space interpolation
   of the old U9 T/AH table is prohibited.
6. M1 commands are canonicalized through the model's native float32 action interface; M2 uses its native scalar command.
7. Maximum inversion iterations per target: 24. Failure to meet the frozen dose tolerance is a runtime failure.

The inversion uses the locked U9 curve only to establish a non-extrapolated bracket and starting interpolation. It does
not replace the final physical-dose or outcome execution.

## Frozen dose-match tolerance

Because M1's native action interface is float32, exact real-number inversion is not representable at arbitrary commands.
The following tolerances are frozen before outcomes:

- DV: `abs(actual-target) <= 2e-5 + 2e-7*abs(target)` m3 m-2.
- DN: `abs(actual-target) <= 5e-6 + 2e-7*abs(target)` air-volume equivalents.

These tolerances are far smaller than the smallest locked positive primary DV overlap width (~0.0222 m3 m-2) and are
used only as numerical matching gates, not as scientific effect thresholds.

## Benefit normalization on the matched physical coordinate

For model m, event i, frozen q level, and variable y in {T, AH}:

`b_miy(q) = -sign(G_iy) * [y_mi(q)-y_mi(q=0)] / |G_iy|`

where `G_iT` and `G_iAH` are the same frozen D2 pre-event gradients used in EXP3.1.

Equal-weight matched-dose benefit:

`B_mi(q) = 0.5*b_miT(q) + 0.5*b_miAH(q)`.

Positive benefit retains the same meaning as EXP3.1: predicted motion toward the outdoor state relative to the lowest
common physically matched dose q=0.

## Intervention cost and decision rule

The old native-command quadratic cost is **not** used for matched-dose decisions.

Primary physical-coordinate cost:

`C(q) = q^2`.

Primary utility:

`Q_mi(q; lambda) = B_mi(q) - lambda*C(q)`.

Primary lambda domain: `[0,1]`.

Decision:

`q*_mi(lambda) = argmax_q Q_mi(q;lambda)`.

Tie rule: within `1e-12`, choose the smaller q.

Decision regions are integrated exactly over the lambda domain using analytic pairwise line-intersection breakpoints.
No single lambda is promoted as the primary result.

## Predeclared sensitivity analyses

These are secondary and cannot replace the primary quadratic `[0,1]` result:

1. no-cost anchor: `lambda=0`;
2. linear physical-coordinate cost `C(q)=q` over `[0,1]`;
3. quadratic physical-coordinate cost `C(q)=q^2` over `[0,3]`.

No cost parameter is tuned after viewing results.

## Primary endpoints

For each event in the relevant coordinate cohort:

1. integrated decision-disagreement measure over lambda;
2. integrated physical action gap `|q*_M1-q*_M2|`;
3. symmetric normalized cross-model regret;
4. directional normalized regret M1→M2 and M2→M1;
5. directional regret asymmetry;
6. symmetric predicted T/AH outcome divergence;
7. signed physical q shift;
8. cross-evaluator baseline-violation measure and deficit area (descriptive continuation of EXP3.3).

For model swap, a selected physical target q is evaluated in the alternate model at that alternate model's own solved
native command for the **same q**. A native command from one model is never transferred directly into the other model.
Neither evaluator is ground truth.

## Before/after comparison with EXP3.1

To avoid cohort-selection artifacts:

- DV before/after uses the old EXP3.1 primary event metrics **restricted to the same locked 42-event DV cohort**;
- DN before/after uses the full old EXP3.1 61-event cohort against the matched-DN 61-event cohort;
- events 27, 86, 89 receive explicit event-level old-vs-matched reporting.

The old all-61 EXP3.1 summary may be shown for context but is not used as the matched-DV comparator.

## Uncertainty summary

For cohort means, retain the EXP3.1 date-cluster bootstrap convention:
- seed `20261002`;
- 2000 bootstrap replicates;
- 95% percentile interval.

## Hosted fail-closed gates

Fail the workflow if any of the following occurs:

- frozen target CSV SHA256 differs from
  `e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292`;
- 61-event input SHA differs from the locked value;
- frozen model commit differs;
- target q set differs from `{0,.25,.5,.75,1}`;
- DV15 eligible event count differs from 42 or DN15 eligible event count differs from 61;
- event 27, 86 or 89 is absent from the DV15 eligible cohort;
- a target is outside its locked `[common_low,common_high]` interval;
- the inversion bracket exits native `u=[0.1,0.9]`;
- final actual dose error exceeds the frozen tolerance;
- requested/applied canonical command differs;
- initial common state gate fails;
- any required state, outcome, utility, dose, or command value is non-finite;
- target rows are missing or duplicated;
- matched M1/M2 q grids are not identical within each cohort;
- exact lambda partition coverage fails.

A change, disappearance, persistence, increase or decrease of scientific disagreement is **not** a runtime failure.

## Interpretation boundary

If 27/86/89 persist under primary matched DV, the old primary disagreement cannot be explained solely by native
command-to-dose mismatch.

If they disappear or materially shrink, report that directly: the old disagreement had a substantial actuator-coordinate
component.

In either case EXP3.4B does not identify a true model, validate realised airflow in D2, prove real-world optimal control,
or justify promoting DN above DV post hoc.
