# PhysBench-GH EXP3.2 — Decision Robustness (Frozen Protocol)

EXP3.2 is a pre-specified sensitivity analysis of the locked EXP3.1 decision-consequence protocol.
Its purpose is to test whether model-to-model decision disagreement is robust to reasonable
representations of utility, intervention cost, action-grid resolution, and prediction horizon.

## Locked ancestry

EXP3.1 final lock:
- repository: `colbyamiee31-boop/qwq`
- commit: `8c68e2262f9cbaffdc1a14a2309e9fdf7644df11`
- workflow run: `37024060571`
- final artifact SHA256: `09ab52edbdff6ffc8da24c087a528c790dbcc366fa34a4ac7a61bb23e9342af8`
- cohort: same 61 D2 matched events
- frozen M1 commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- frozen M2 commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`
- frozen input SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`

No EXP3.1 result is re-tuned in EXP3.2.

## Fine action-grid run

Both frozen models are rerun on the nested 9-point grid

`U9 = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}`

with the same initialization, weather/context mapping, horizons and runtime gates as EXP3.1.

Nested grids used by the sensitivity analysis:
- coarse: `U3={0.1,0.5,0.9}`
- reference: `U5={0.1,0.3,0.5,0.7,0.9}`
- fine: `U9`

Reference action is always `u0=0.1`.

## Benefit normalization

For model `m`, event `i`, horizon `h`, outcome `y∈{T,AH}`:

`b_miy(u) = -sign(G_iy) * [y_mi(u,h)-y_mi(u0,h)] / |G_iy|`

The empirical event gradient `G_iy` is unchanged from EXP3.1.

Three pre-specified utility weights:
- T75_AH25: `B=0.75*bT + 0.25*bAH`
- equal: `B=0.50*bT + 0.50*bAH`
- T25_AH75: `B=0.25*bT + 0.75*bAH`

## Intervention cost

Let `z=(u-0.1)/0.8`.

Two pre-specified cost forms:
- quadratic: `C=z^2` (EXP3.1 reference)
- linear: `C=z`

For lambda `λ`:

`Q_mi(u;λ)=B_mi(u)-λ*C(u)`

Decision = argmax Q. Ties within `1e-12` choose the lower action.

## Domains and strata

- horizons: 15 min and 30 min
- primary lambda domain for robustness comparison: `[0,1]`
- extended secondary lambda domain: `[0,3]`
- day/night are secondary strata only
- exact lambda partitions are used; no single lambda replaces the integrated metric

## Metrics

For every event and sensitivity specification:
- decision-disagreement measure
- integrated normalized action gap
- symmetric cross-model regret
- symmetric predicted outcome divergence
- any-disagreement indicator

The outcome-divergence metric retains the EXP3.1 equal physical normalization across T and AH so that
changes in selected actions can be compared across utility-weight specifications.

## Primary robustness family

The primary robustness family is restricted to:
- horizon = 15 min
- lambda domain = `[0,1]`
- 3 utility weights × 2 cost forms × 3 grids = 18 pre-specified specifications.

For each event, report disagreement persistence across these 18 specifications.
No post-hoc parameter selection is permitted.

## Reference-reproduction gate

The specification:
- equal weights
- quadratic cost
- U5 grid
- 15 min
- lambda `[0,1]`

must reproduce the locked EXP3.1 primary numerical results within `1e-10`.

This is an implementation-consistency gate, not a new scientific test.

## Interpretation boundary

EXP3.2 assesses robustness to decision representation. It does not establish either model as ground truth,
does not validate realized greenhouse control actions, and must not be used to tune a specification
for larger disagreement.
