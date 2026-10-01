# PhysBench-GH EXP1.3 — CO2 Flux-Level Decomposition of the Finite-Horizon Boundary Shift

## Scientific question
EXP1.2 showed that the 900-s HIGH-minus-LOW CO2 response reversal boundary differs from the instantaneous direct-transport equilibrium:

- M1 GreenLight-Gym2: direct density-equilibrium displayed at t=0 ≈ 423.552 ppm; 900-s total-response boundary ≈ 438.533 ppm.
- M2 CSGtom: direct ventilation equilibrium = 415 ppm; 900-s total-response boundary ≈ 358.279 ppm.

EXP1.3 asks: **which native CO2 ODE terms cancel or reinforce the direct ventilation contribution at the finite-horizon reversal boundary?**

## Hard methodological boundary
This experiment is read-only with respect to scientific model equations.

Allowed:
- evaluate native ODE terms;
- integrate/log those terms along the unmodified LOW/HIGH trajectories;
- refine the previously bracketed response root numerically;
- compute exact algebraic closure diagnostics.

Not allowed:
- disable a flux term;
- replace a flux term;
- structural replay;
- counterfactual equation repair;
- parameter retuning.

Thus EXP1.3 is a **flux accounting / decomposition experiment**, not a replay experiment.

## Frozen setup inherited from EXP0.5 / EXP1.2
- benchmark time: t = 0;
- one-step horizon: 900 s;
- LOW natural ventilation command = 0.1;
- HIGH natural ventilation command = 0.9;
- all other native controls fixed as in the common adapter;
- same benchmark-owned forcing interval;
- same matched observable initial state except indoor CO2 is varied to locate the finite-horizon root;
- M1 native integration: CasADi CVODES, 900 s;
- M2 native integration: forward Euler, 30 s.

## Root refinement
Use the EXP1.2 frozen sign-changing brackets:

- M1: 435–455 ppm;
- M2: 355–375 ppm.

Within each bracket, refine the zero of

`Delta_CO2(C0) = CO2_HIGH(900; C0) - CO2_LOW(900; C0)`

by bisection until either:
- bracket width < 1e-7 ppm, or
- |Delta_CO2| < 1e-9 ppm.

The refined root is used only to improve flux accounting at the already-discovered boundary; it does not redefine the experimental endpoint post hoc.

## M1 GreenLight native CO2 decomposition
The frozen main-compartment CO2-density ODE is:

`dC_air/dt = (mcBlowAir + mcExtAir + mcPadAir - mcAirCan - mcAirTop - mcAirOut) / cap_CO2_air`

with source indices in the frozen implementation:
- `a[223]`: blower source (currently 0);
- `a[222]`: external CO2 dosing source;
- `a[224]`: pad source (currently 0);
- `a[216]`: net air-canopy carbon exchange;
- `a[217]`: main-to-top compartment exchange;
- `a[219]`: main-to-outside air exchange.

EXP1.3 augments the *numerical integrator only* with CasADi quadratures for these signed ODE contributions. The state ODE is unchanged.

Because M1 stores CO2 as mass density but reports ppm using temperature, the final HIGH-minus-LOW ppm contrast is decomposed exactly into:

1. density-change contribution, itself split by integrated native CO2 flux term;
2. temperature-to-ppm conversion contribution.

The symmetric bilinear identity is used so the reported parts close exactly to final `Delta ppm` up to solver tolerance.

## M2 CSGtom native CO2 decomposition
The frozen CO2 ODE is separated into the terms already present in source:

- ventilation exchange;
- photosynthetic uptake;
- organic/maintenance respiration return;
- soil respiration;
- external CO2 source.

No term is removed. During the unmodified 30-s Euler rollout, wrappers log the values returned by the existing crop/control functions; each native rate is accumulated with the same 30-s Euler weights used by the solver.

## Primary decomposition quantity
For each model, at its refined 900-s response boundary, report the HIGH-minus-LOW contribution of every native CO2 term to the final CO2 contrast.

At a true response root:

`sum_j Delta FluxContribution_j + unit-conversion term (M1 only) ≈ 0`.

The terms can therefore be classified as:
- direct ventilation contribution;
- compensating coupled biological/inter-compartment contribution;
- residual/numerical closure.

## Secondary audit points
Also decompose two nearby initial-CO2 values on each side of the refined root:

- M1: 425 ppm and 455 ppm;
- M2: 355 ppm and 415 ppm.

These are descriptive sensitivity checks, not new root estimates.

## Runtime/closure gates
Scientific results never fail CI merely because one term is unexpectedly large.
CI fails only if:
1. runtime is non-finite;
2. requested/applied native ventilation differs;
3. M1 augmented-quadrature final state disagrees with the frozen native integrator beyond 1e-7 absolute state units;
4. M1 density-flux closure error exceeds 1e-5 mg m^-3;
5. M1 final ppm decomposition closure exceeds 1e-5 ppm;
6. M2 Euler flux closure exceeds 1e-8 ppm;
7. root refinement leaves the inherited sign-changing bracket.

## Interpretation boundaries
- Flux attribution is model-internal mechanistic accounting, not empirical truth.
- The two models use different native CO2 states/units and different crop formulations; raw term magnitudes must not be ranked as if they were the same equation family.
- The experiment explains *how each model produces its own finite-horizon boundary shift*.
- Empirical actuator-log validation remains a separate later stage.
