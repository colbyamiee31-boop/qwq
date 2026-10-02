# PhysBench-GH EXP3.3 — Non-Redundant Directional Model-Swap Consequence (Frozen Protocol)

## Purpose

EXP3.3 asks a question that is deliberately distinct from EXP3.1 and EXP3.2:

> When the action selected by one frozen greenhouse model is transferred to the other model's evaluator,
> what is the direction and severity of the predicted consequence?

EXP3.1 already reported symmetric cross-model regret and symmetric predicted outcome divergence.
EXP3.2 tested whether disagreement was robust to utility, intervention-cost, and action-grid representation.
Therefore EXP3.3 must not simply recompute another symmetric average.

The primary novelty of EXP3.3 is directional decomposition plus a stronger below-baseline consequence test.

## Locked ancestry

EXP3.1 FINAL LOCK:
- commit: `8c68e2262f9cbaffdc1a14a2309e9fdf7644df11`
- workflow run: `37024060571`
- final artifact SHA256: `09ab52edbdff6ffc8da24c087a528c790dbcc366fa34a4ac7a61bb23e9342af8`
- locked reference disagreement events: `27, 86, 89`

EXP3.2 FINAL LOCK:
- commit: `48feabd403e9a4f65aa438a5da3d507fc754e630`
- workflow run: `37026596313`
- final artifact SHA256: `11f3be486b0f74b588690955898b2d018666d31c30876f17493af19ad96139b9`
- primary robustness family: 18 pre-specified utility/cost/grid specifications

Frozen models and cohort:
- M1 commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`
- shared model-input SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`
- cohort: all 61 D2 matched events

No event may be newly selected for the primary analysis from EXP3.2 results.

## Model-swap definition

For model `m∈{M1,M2}`, event `i`, horizon `h`, and decision specification `s`:

`Q_m(u;lambda)=B_m(u)-lambda*C(u)`

where benefit normalization, utility weights, cost forms, tie-breaking, and action grids are exactly those
frozen in EXP3.1/EXP3.2.

Let:
- `a1*(lambda)` = action selected by M1
- `a2*(lambda)` = action selected by M2

Directional transfers:
- M1→M2: M1 selects `a1*`, M2 evaluates that transferred action.
- M2→M1: M2 selects `a2*`, M1 evaluates that transferred action.

Neither evaluator is treated as real-world ground truth.

## Primary specification

Primary analysis uses the locked EXP3.1 reference specification:
- all 61 events
- horizon = 15 min
- lambda domain = `[0,1]`
- action grid = `g5={0.1,0.3,0.5,0.7,0.9}`
- utility weights = T/AH = `0.50/0.50`
- intervention cost = quadratic

This preserves direct comparability with EXP3.1 and avoids choosing a specification based on EXP3.2.

## Primary directional endpoints

For selector `s` and alternate evaluator `e`, where `s != e`:

### 1. Directional normalized transfer regret

`R_{s→e}(lambda) = [Q_e(a_e*) - Q_e(a_s*)] / span(Q_e)`

where `span(Q_e)=max_u Q_e(u)-min_u Q_e(u)`.
If the span is numerically zero, regret is defined as zero.

Report the lambda-domain weighted mean using the same exact decision partitions and midpoint convention as EXP3.1.

This yields two separate endpoints:
- `R_M1toM2`
- `R_M2toM1`

### 2. Directional regret asymmetry

`A_R = R_M1toM2 - R_M2toM1`

Positive values mean transferring the M1-selected action to M2 produces greater normalized regret than
the reverse transfer, under the model-defined evaluators.

### 3. Cross-evaluator baseline-violation measure

The baseline action is `u0=0.1`. By construction,
`Q_e(u0)=0`.

For each transfer, compute the exact fraction of the lambda domain for which:

`Q_e(a_s*) < 0`

This is called the **cross-evaluator baseline-violation measure**.

It is a model-predicted decision consequence, not evidence of real greenhouse harm.

### 4. Cross-evaluator baseline-deficit area

For each transfer, compute the lambda-domain mean of:

`max(0, -Q_e(a_s*))`

integrated exactly over segments where the selected action is fixed.

This quantifies severity below the evaluator's baseline, without introducing an arbitrary materiality threshold.

### 5. Outcome-component transfer loss

Using the alternate evaluator's normalized physical benefit components:

`L_T = bT_e(a_e*) - bT_e(a_s*)`

`L_AH = bAH_e(a_e*) - bAH_e(a_s*)`

Report lambda-domain weighted signed means separately for M1→M2 and M2→M1.

Positive values indicate that the alternate evaluator's own selected action is better on that component.
Negative values are allowed and represent a trade-off.

### 6. Signed action shift

`S_a = (a1* - a2*) / 0.8`

Report the lambda-domain weighted signed mean and absolute mean.

Positive values mean M1 recommends a more intensive ventilation action than M2 over the domain.

## Reference-reconstruction gate

For the locked EXP3.1 reference specification, EXP3.3 must reconstruct:

- EXP3.1 symmetric cross-model regret =
  `0.5*(R_M1toM2 + R_M2toM1)`
- EXP3.1 symmetric predicted outcome divergence from the directional selected-action pair
- locked disagreement event IDs = `27,86,89`

All locked EXP3.1 numerical summary values must be reproduced within absolute tolerance `1e-10`.

Failure closes the gate.

## Secondary analyses

Pre-specified secondary analyses:
1. horizon = 30 min with lambda `[0,1]`
2. extended lambda domain `[0,3]`
3. day/night stratification
4. directional consequence robustness across the 18 locked EXP3.2 primary specifications

The 18-spec robustness analysis may report:
- persistence of nonzero directional regret
- persistence of baseline violation
- persistence of the sign of directional regret asymmetry

It must not select the specification with the largest consequence as the main result.

## Pre-locked subgroup

Only the EXP3.1 events `27, 86, 89` may be reported as a pre-locked subgroup.

Events that first appeared in EXP3.2 sensitivity analysis must remain descriptive only and may not be
promoted into a confirmatory subgroup.

## Explicit exclusions

EXP3.3 does not:
- treat M1 or M2 as real-world truth;
- infer realized actuator performance;
- claim causal greenhouse harm;
- introduce a post-hoc materiality threshold;
- tune utility weights, cost form, action grid, or lambda range to maximize consequence;
- include a receding-horizon toy controller.

A receding-horizon controller, if scientifically needed later, must be pre-specified as a separate experiment.

## Interpretation boundary

Supported language:
- directional predicted transfer consequence;
- cross-model decision-transfer asymmetry;
- below-baseline prediction under the alternate model evaluator;
- robustness or non-robustness across pre-specified decision representations.

Unsupported language:
- real-world control failure;
- actual crop/environmental harm;
- one model being correct and the other wrong;
- universal consequence independent of specification.
