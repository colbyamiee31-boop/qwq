# PhysBench-GH EXP3.4A — Physical Ventilation-Dose Equivalence

## Status

Frozen before inspecting EXP3.4A physical-dose results.

EXP3.4A addresses the action-comparability limitation of EXP3.1–EXP3.3. The prior decision experiments used the
same numerical command grid in both models but explicitly did not establish equal airflow. EXP3.4A does not reinterpret
those commands. It measures each frozen model's native external ventilation flow and constructs a predeclared common
physical-dose coordinate for the later dose-matched decision experiment.

EXP3.4A itself contains **no decision optimization** and does not select a dose definition according to whether
decision disagreement increases or decreases.

## Frozen ancestry

Repository: `colbyamiee31-boop/qwq`

Parent EXP0.6C commit:
`15b427e4664adc14d7e021866a0514ad385b77ae`

Frozen models:
- M1 GreenLight-Gym2: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

Frozen 61-event input:
- `physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv`
- SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`

Locked EXP3.1 hosted references:
- run `37024060571`;
- M1 action-grid artifact `11234970713`, artifact digest
  `0cc114b700cf2a985a5a5283445dce06ac8842451aacf0d228060ea438b1976d`;
- M2 action-grid artifact `11234595886`, artifact digest
  `04c07343a84309df654375faa53fc0b1459f5a59c9c0020caee07b3120023460`;
- M1 locked `action_grid_responses.csv` SHA256
  `0c6579ec4992c6513743ad5d535825bc07430fa346115d6d822e2249824a8818`;
- M2 locked `action_grid_responses.csv` SHA256
  `c00813e28ade4a0e85bb795851fe69bee49b34598db1e150eb7c2bc82979de1f`.

## Scientific question

For the same 61 HNR event contexts and frozen forcing used by EXP3.1:

1. what physical external ventilation dose does each native command produce?
2. is the command-to-dose mapping monotone?
3. how much of the two models' physically achievable dose ranges overlaps?
4. can a symmetric five-level common physical-dose grid be defined without extrapolation?
5. does the conclusion depend on expressing dose as floor-area-specific exchanged volume or as air-volume equivalents?

No model is designated the reference model.

## Event initialization

Use the exact HNR initialization and forcing representation from EXP3.1:
- indoor T = event `event_Tair`;
- indoor vapour pressure = `in_vp_pa`;
- indoor CO2 = `CO2_pre_ppm`;
- canopy T = event air T;
- all other states = frozen model-native reset values;
- constant pre-event forcing snapshot;
- outdoor CO2 = 415 ppm;
- sky-temperature proxy and soil-boundary value exactly as in the frozen event table.

EXP0.6C latent histories are not propagated here because EXP3.4A isolates the action-coordinate question.
Latent-state robustness has already been tested separately.

## Diagnostic command grid

Before matching, evaluate both models at:

`U9 = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}`.

Horizons:
- primary: 900 s;
- secondary: 1800 s.

The original EXP3.1 five actions `{0.1,0.3,0.5,0.7,0.9}` are a strict subset.

## Primary physical dose

For each model, event i, command u, and horizon h:

`D_V(m,i,u,h) = integral q_ext_m(t;u) dt`

where `q_ext` is the model-native whole-greenhouse external volumetric ventilation flux defined in
`EXP3_4A_PHYSICAL_SOURCE_LOCK.md`.

Units: `m3 m-2`.

This is the **primary** dose coordinate because both frozen source models explicitly express the relevant external
ventilation flux in the same physical units per greenhouse floor area.

## Secondary air-volume-equivalent dose

`D_N(m,i,u,h) = integral q_ext_m(t;u)/H_eff_m dt`.

This is dimensionless cumulative air-volume exchange.

Also report horizon-average:
`ACH_bar = 3600 * D_N / h`.

This secondary coordinate accounts for the different modeled greenhouse air volumes. It cannot replace the primary
coordinate after outcomes are viewed.

## Monotonicity audit

For each model × event × horizon × dose coordinate, the U9 dose sequence must be nondecreasing within a numerical
tolerance of `1e-10 * max(1, max|D|)`.

Report:
- monotone yes/no;
- minimum adjacent dose increment;
- command interval(s) producing any detected reversal.

Non-monotonicity is a scientific result. A non-monotone cell is not silently rearranged or isotonic-regressed.

## Symmetric physical overlap

For each event/horizon/coordinate:

`L = max(D_M1(0.1), D_M2(0.1))`

`H = min(D_M1(0.9), D_M2(0.9))`.

If `H > L`, the common physically reachable interval is `[L,H]`.

Report:
- overlap width `H-L`;
- overlap fraction relative to each model's native 0.1–0.9 dose span;
- endpoint model/command that determines L and H;
- native same-command dose ratios at all U9 levels.

If `H <= L`, the cell has no common physical dose interval and is marked `NO_PHYSICAL_OVERLAP`.
No extrapolation beyond u=[0.1,0.9] is allowed.

## Predeclared common five-level grid for EXP3.4B

For every monotone cell with positive overlap, freeze the later dose-matched decision targets as:

`q = {0,0.25,0.50,0.75,1.00}`

`D_target(q) = L + q*(H-L)`.

EXP3.4A stores these target doses but **does not optimize decisions on them**.

EXP3.4B must use this target grid exactly. It may solve each model's native command required to hit each target, but
it may not redefine q, L, H, or the physical coordinate after seeing decision results.

## Native-command distortion outputs

At each U9 command report:
- M1 and M2 `D_V`;
- M1/M2 dose ratio when denominator is non-zero;
- absolute and relative dose difference;
- M1 and M2 `D_N`;
- average ACH;
- rank/order preservation.

These values quantify how much same-number commands differ physically.

## Reconstruction gates

Before accepting dose outputs:

1. rerun the original EXP3.1 model action-grid scripts in the same hosted jobs;
2. require their runtime gates to pass;
3. retain the historical EXP3.1 CSV SHA256 values above as provenance fingerprints, not as cross-run floating-point hard gates;
4. for U5 rows generated by the dose instrumentation, require T/AH outcomes to match the **same-run** EXP3.1 regeneration
   under the precision-aware reconstruction gate frozen in `EXP3_4A_NUMERICAL_RECON_AMENDMENT.md`: temperature absolute error <= 1e-9 degC and scaled T/AH errors <= 1e-11.

This ensures physical-dose instrumentation does not alter the frozen model trajectory.

## Runtime gates

Fail closed if:
- event/input identity changes;
- frozen model commit differs;
- same-run original EXP3.1 reconstruction fails;
- any state or dose is non-finite;
- requested/applied command mismatch occurs;
- the physical flux units/source definitions differ from the frozen source lock;
- U9 rows are incomplete/duplicated;
- U5 instrumented outcomes fail the reconstruction tolerance.

Monotonicity failure or missing physical overlap is not a runtime failure.

## Interpretation

EXP3.4A can establish:
- whether native command coordinates are physically commensurate;
- the size of their common reachable physical-dose region;
- the exact dose grid on which a fair dose-matched decision comparison can be run.

EXP3.4A cannot establish:
- which greenhouse/model is physically correct;
- real actuator airflow in the public D2 datasets;
- real-world optimal ventilation;
- decision disagreement after dose matching. That is reserved for EXP3.4B.
