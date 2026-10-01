# PhysBench-GH EXP1.5 — Boundary Mechanism Stability / Flux Attribution across the Reversal Surface

## Scientific question
EXP1.4 established that the finite-horizon CO2 reversal boundary is not a scalar constant but a conditional surface:

`C* = C*(H, x0, w)`.

EXP1.5 asks whether the **native physical mechanism that produces the HIGH-minus-LOW reversal remains stable while the boundary moves across horizon × state × forcing**, or whether the dominant cancellation structure itself changes.

The experiment distinguishes:

1. **boundary drift without mechanism switching** — the reversal location moves while the dominant native CO2 contribution remains the same;
2. **boundary drift with mechanism switching** — the dominant native CO2 contribution changes across the conditional surface.

## Hard methodological boundary
EXP1.5 inherits the EXP1.3 read-only rule.

Allowed:
- evaluate the frozen native CO2 ODE terms;
- integrate/log native terms along unmodified LOW/HIGH trajectories;
- reproduce the EXP1.4 root surface using the frozen scan and bisection procedure;
- compute contribution-vector summaries and descriptive similarity statistics.

Not allowed:
- disable or replace fluxes;
- structural replay;
- counterfactual repair;
- parameter retuning;
- expanding the 250–600 ppm root-search domain after inspecting results.

Thus EXP1.5 is a **mechanism-accounting robustness experiment**, not a causal intervention-repair experiment.

## Frozen factorial design inherited from EXP1.4
- horizons: 300, 900, 1800 s;
- states:
  - `S_COOL_HUMID`: Tair = 20 °C, RH = 80%, Tcan = 20.5 °C;
  - `S_NOMINAL`: Tair = 24 °C, RH = 70%, Tcan = 24.5 °C;
  - `S_WARM_DRY`: Tair = 30 °C, RH = 55%, Tcan = 30.5 °C;
- forcing snapshots: F_ROW0, F_ROW6, F_ROW10 from the frozen EXP0.5 forcing table;
- LOW ventilation = 0.1;
- HIGH ventilation = 0.9;
- initial-CO2 search domain = 250–600 ppm;
- coarse spacing = 25 ppm;
- sign-changing brackets refined by bisection;
- 27 cells per model, 54 total.

Expected EXP1.4 topology is frozen as a regression gate:
- M1: 21 unique-root cells, 6 no-crossing cells, 0 multiple-root cells;
- M2: 18 unique-root cells, 9 no-crossing cells, 0 multiple-root cells.

The 900-s / S_NOMINAL / F_ROW0 roots must reproduce the EXP1.4 anchors within 0.02 ppm:
- M1: 438.5370684042573 ppm;
- M2: 358.2789194211364 ppm.

## Boundary and edge attribution
### Unique-root cells
For every cell containing exactly one root in the declared domain, evaluate the full EXP1.3-style HIGH-minus-LOW native flux decomposition at the refined root.

### No-crossing cells
Do **not** expand the search domain. Evaluate flux attribution only at the two predeclared domain edges:
- 250 ppm;
- 600 ppm.

These are labeled `EDGE_AUDIT` and are not treated as reversal boundaries.

### Multiple-root cells
If encountered despite the frozen EXP1.4 topology, record every detected root and fail the topology regression gate. Do not collapse multiple roots into one.

## M1 native terms
Primary native M1 contribution vector, in ppm-equivalent final HIGH-minus-LOW contribution:
- canopy_net;
- main_to_top;
- main_to_outside;
- co2_injection;
- blower_source;
- pad_source;
- temperature_conversion.

The first six are integrated native CO2-density ODE terms. `temperature_conversion` is the exact symmetric temperature-to-ppm conversion term inherited from EXP1.3.

Primary direct-exchange term: `main_to_outside`.

## M2 native terms
Primary native M2 contribution vector, in final HIGH-minus-LOW ppm contribution:
- ventilation_exchange;
- photosynthesis_uptake;
- organic_respiration;
- soil_respiration;
- external_co2_source;
- residual.

Primary direct-exchange term: `ventilation_exchange`.

## Mechanism-stability metrics
For each unique-root contribution vector `v`:

### 1. L1-normalized signed composition
`p_j = v_j / sum_k |v_k|`.

### 2. Dominant mechanism
`argmax_j |v_j|`.

### 3. Dominance share
`max_j |v_j| / sum_k |v_k|`.

### 4. EXP1.3-anchor cosine similarity
Compare each normalized mechanism vector with the 900-s / S_NOMINAL / F_ROW0 vector from the same model.

Cosine similarity is descriptive; no pass/fail threshold is attached.

### 5. L1 distance from anchor
`sum_j |p_j - p_anchor,j|`.

### 6. Active-term sign pattern
A term is called active only if its absolute L1 share is at least 0.10 in that cell. Sign comparisons use only terms active in both the cell and the anchor, preventing near-zero numerical terms from creating artificial sign flips.

### 7. Dominant-mechanism switch
A cell is flagged `dominant_switch_from_anchor = true` only if its dominant native term differs from the model-specific anchor dominant term.

No composite subjective score is used.

## Secondary conceptual grouping
For interpretation only, native terms may also be aggregated into broader within-model groups.

M1:
- direct_exchange = main_to_outside;
- canopy_biology = canopy_net;
- intercompartment = main_to_top;
- external_source = co2_injection + blower_source + pad_source;
- representation = temperature_conversion.

M2:
- direct_exchange = ventilation_exchange;
- canopy_biology = photosynthesis_uptake + organic_respiration;
- soil_biology = soil_respiration;
- external_source = external_co2_source;
- residual = residual.

These groups are descriptive alignment aids only. Raw native-term attribution remains primary, and magnitudes are not ranked across model families.

## Numerical / runtime gates
### M1
Inherit EXP1.3 tolerances:
- augmented quadrature vs native final state <= 1e-7 absolute state units;
- named signed fluxes vs integrated total CO2 ODE <= 1e-10 mg m^-3;
- paired density-flux closure <= 0.005 mg m^-3;
- final ppm decomposition closure <= 0.005 ppm;
- all states/contributions finite.

### M2
- requested/applied ventilation error = 0;
- all states/contributions finite;
- Euler integrated total-ODE vs final-state closure <= 1e-8 ppm;
- paired decomposition closure <= 1e-8 ppm.

### Surface regression gates
- expected EXP1.4 topology count must reproduce exactly;
- anchor root error <= 0.02 ppm;
- no detected root may lie outside its sign-changing bracket.

Scientific results do not fail merely because a contribution is unexpectedly large or because the dominant mechanism changes.

## Interpretation boundaries
- EXP1.5 is model-internal mechanism accounting, not empirical causal validation.
- Mechanism stability means stability of each model's own native equation decomposition.
- M1 and M2 use different state representations and crop/transport formulations; raw term magnitudes must not be interpreted as directly comparable physical measurements.
- A `NO_CROSSING_IN_WINDOW` cell means only that no root was detected in the frozen 250–600 ppm domain.
