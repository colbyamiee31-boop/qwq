# PhysBench-GH EXP1.4 — Boundary Robustness across Horizon × State × Forcing

## Scientific question
EXP1.2 identified state-conditioned finite-horizon CO2 response reversals and EXP1.3 explained the native flux cancellations at the nominal 900-s boundary. EXP1.4 asks:

**Is the finite-horizon HIGH-minus-LOW CO2 reversal boundary a robust surface across response horizon, matched initial climate state, and external forcing regime, or is it a single-case artifact?**

The endpoint remains

`Delta_CO2(C0 | H,S,F) = CO2_HIGH(H; C0,S,F) - CO2_LOW(H; C0,S,F)`

with LOW natural ventilation = 0.1 and HIGH = 0.9.

## Hard methodological boundary
No scientific equation, flux, parameter, controller mapping, solver tolerance, or actuator implementation is repaired or retuned.

Allowed:
- vary declared benchmark horizon;
- vary only the common observable initial-state fields declared below;
- select frozen benchmark forcing snapshots;
- scan initial indoor CO2 and numerically refine every detected sign-changing boundary;
- report absence or multiplicity of roots without forcing a single-valued surface.

Not allowed:
- disable fluxes;
- structural replay;
- counterfactual equation replacement;
- model-specific parameter tuning to align boundaries;
- post-hoc deletion of inconvenient cells.

## Factorial design
Primary matrix: **3 horizons × 3 initial states × 3 forcing snapshots = 27 cells per model.**

### Horizon H
- H300 = 300 s (5 min)
- H900 = 900 s (15 min)
- H1800 = 1800 s (30 min)

These horizons are long enough to expose coupled dynamics while remaining local intervention-response tests rather than long closed-loop forecasts.

### Common observable initial climate state S
All model-native non-common hidden states remain exactly as in the frozen native reset. Only the common observable state fields are overwritten.

- `S_COOL_HUMID`: air T = 20.0 C; RH = 80%; canopy T = 20.5 C
- `S_NOMINAL`: air T = 24.0 C; RH = 70%; canopy T = 24.5 C; vapor pressure exactly inherited from EXP0.5 matched state
- `S_WARM_DRY`: air T = 30.0 C; RH = 55%; canopy T = 30.5 C

For non-nominal states, vapor pressure is constructed using the same saturation-vapor-pressure convention used by the benchmark adapter:

`VPsat(T) = 610.78 * exp(17.2694*T/(T+238.3)) Pa`,

`VP = RH/100 * VPsat(T)`.

Initial CO2 is not fixed by the state factor because it is the scanned boundary coordinate.

### External forcing F
Use rows already present in the frozen EXP0.5 benchmark forcing file:

- `F_ROW0`: forcing row index 0, benchmark time 0 s
- `F_ROW6`: forcing row index 6, benchmark time 5400 s
- `F_ROW10`: forcing row index 10, benchmark time 9000 s

These provide low-, intermediate-, and high-radiation benchmark regimes without inventing new weather values.

During each rollout, the selected **external weather snapshot is held constant** over the declared horizon. Model-native internal time/calendar calculations, where present, are allowed to evolve normally and are not frozen by equation edits. For CSGtom, native clock start is aligned to 08:00 plus the selected forcing-row benchmark time.

For M1, the model-native DLI/day flags associated with the selected row are reconstructed from the original benchmark sequence exactly as in EXP0.5: cumulative 900-s radiation integral through the selected row, plus the native day flags.

## Boundary scan
For every `(H,S,F)` cell:

1. Evaluate `Delta_CO2` on a fixed initial-CO2 grid from **250 to 600 ppm in 25-ppm increments**.
2. Preserve the complete response scan.
3. Detect every adjacent sign-changing bracket; an exact grid zero is also a boundary candidate.
4. Refine each bracket by bisection until either:
   - bracket width <= 1e-4 ppm, or
   - absolute response <= 1e-7 ppm,
   with a hard cap of 50 iterations.
5. Deduplicate numerically coincident roots within 1e-3 ppm.

The scan range is a declared benchmark window, not a claim that roots cannot exist outside it. A cell with no detected root is reported as **no boundary within 250–600 ppm**, not as proof of global nonexistence.

## Multiple-root policy
EXP1.4 does not assume monotonicity or uniqueness.

For each cell report:
- root count within the scan window;
- all refined roots;
- response residual and final bracket width for each root.

A `primary_boundary_ppm` is populated only when exactly one root is found. If zero or multiple roots are detected, the primary boundary is left null so that a complex topology is not silently collapsed to one number.

## Direct-transport reference
For M2, the direct ventilation equilibrium is the selected outdoor CO2 concentration (415 ppm in the frozen forcing file).

For M1, outside CO2 is represented natively as mass density. The comparable direct transport equilibrium is converted to displayed ppm at the selected **initial indoor air temperature**:

`Ceq_display = mg_m3_to_ppm(T_indoor_initial, ppm_to_mg_m3(T_outdoor, CO2_outdoor))`.

When a unique finite-horizon boundary exists, report

`boundary_shift_ppm = primary_boundary_ppm - direct_transport_equilibrium_ppm`.

## Frozen model implementations
### M1 — GreenLight-Gym2
- commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- native CO2 state: main-air CO2 mass density
- integrator: CasADi CVODES
- frozen tolerances: `abstol = reltol = 1e-4`
- natural ventilation command: `uVent`

Separate integrator objects are instantiated for 300, 900 and 1800 s; the scientific ODE and tolerances are unchanged.

### M2 — CSGtom
- commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`
- native CO2 state: ppm
- native forward Euler step: 30 s
- natural ventilation command: existing time-based `u_vent` path

The number of native Euler steps is therefore 10, 30 and 60 for the three horizons.

## Runtime and reproducibility gates
CI fails only for implementation/audit failures, not because a scientific boundary moves, disappears, or multiplies.

Required gates:
1. all evaluated model states and responses are finite;
2. M2 requested and applied natural-ventilation commands agree exactly;
3. all refined roots remain inside their detected scan brackets;
4. each bisection terminates by the declared tolerance or exact-response criterion;
5. the nominal `(H900, S_NOMINAL, F_ROW0)` cell reproduces the EXP1.3 refined boundary within 0.02 ppm:
   - M1 reference = 438.5370684042573 ppm;
   - M2 reference = 358.2789194211364 ppm;
6. M1 H900 nominal direct-transport reference reproduces the EXP1.3 displayed equilibrium to numerical precision.

Root count is explicitly **not** a pass/fail criterion.

## Primary outputs
Per model:
- complete cell-level JSON;
- cell summary CSV;
- complete CO2 response-scan CSV;
- refined-root CSV;
- factor-level robustness summary (existence fraction, unique-root range, median/IQR by horizon/state/forcing when defined);
- runtime/reproducibility audit.

## Interpretation boundaries
- EXP1.4 measures **model-internal intervention-boundary robustness**, not empirical greenhouse truth.
- The 27-cell matrix is a controlled benchmark perturbation design, not a claim that the three state cases represent the full empirical greenhouse-state distribution.
- Cross-model equality of command fraction does not imply equal physical airflow.
- A moving finite-horizon boundary is not itself a physics violation; it quantifies state/forcing/horizon dependence of the coupled intervention response.
- Empirical actuator-log alignment remains a separate later experiment.
