# PhysBench-GH R3 — Common Physical Envelope / Geometry Harmonisation

## Protocol freeze
This file is committed before any R3 harmonised-model response, dose, empirical-alignment, or decision-consequence result is generated.

## Scientific question
How much of the previously observed M1/M2 discrepancy is attributable to semantically comparable greenhouse transport-envelope differences rather than residual model-form/process structure?

R3 does not force the two models into the same mathematical model. It harmonises only quantities that have a defensible one-to-one physical meaning in both implementations.

## Frozen ancestry
- R2 final successful head: `546939cfc4d89724da48dc0357dd12f490d1ed32`
- R2 successful run: `37145919425`
- R2 prepared-input artifact SHA256: `5b2c6d98ea17b916e9d4ccd490dce2a3c3280b1d4d7453a5c6f732f1953ec66e`
- R1.2 successful run: `37143141817`
- EXP3.1 native-decision reference run: `37024060571`
- D2 61-event input SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`
- M1 GreenLight-Gym2 commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

## External target envelope
The harmonisation target is the CTIFL Venlo compartment used in R1/R2.

Geometry:
- floor area = 43.2 × 24 = 1036.8 m2
- gutter height = 6.97 m
- ridge height = 7.78 m
- whole-greenhouse effective air height:
  `H_target = 6.97 + 0.5 × (7.78 − 6.97) = 7.375 m`

Roof vents:
- East-facing: 24 vents of 4.05 × 1.40 m
- West-facing: 18 vents of 4.05 × 1.40 m plus 12 vents of 1.35 × 1.40 m
- equivalent 4.05 × 1.40 m count = 46
- maximum opening angle = 44 degrees
- gross vent-panel area per floor area:
  `Avent_gross/Afloor = 0.2515625`
- projected maximum open-aperture area per floor area:
  `Rvent_target = 0.2515625 × sin(44 deg) = 0.17474999631859142 m2 m-2`

The projected aperture ratio is used because M1 and M2 ventilation equations consume an aperture/vertical opening area rather than total vent-panel surface.

## What is harmonised

### Stage N — native reference
No parameters changed. Native results are inherited from frozen R1.2/R2/EXP3.1 artifacts and are not regenerated to define the reference.

### Stage H — common air-volume envelope
M1:
- `hAir` = 6.97 m
- `hGh` = 7.375 m
- dependent main/top air heat and CO2 capacities are recomputed exactly from the modified heights.

M2:
- native surface/topology geometry is retained;
- `Vair / area_floor` is overridden to exactly 7.375 m.

No vent-area parameter is changed in Stage H.

### Stage HV — common air-volume + roof-aperture envelope (PRIMARY R3 HARMONISATION)
Stage H changes plus:

M1:
- `aRoof / aFlr = Rvent_target`.

M2:
- `Atop_vent × r_net / area_floor = Rvent_target`;
- native `r_net` is retained and `Atop_vent` is changed only as required to make the effective top-opening ratio exact.

## What is deliberately NOT harmonised
The following remain model-native because their semantics or topology are not one-to-one:
- discharge coefficient;
- wind-pressure coefficient;
- leakage formulation and leakage coefficient;
- vent buoyancy characteristic-height formulation;
- roof/wall topology and cover shape;
- north-wall thermal mass in CSGtom;
- screen topology;
- cover/material optical and thermal properties;
- crop, canopy, soil and condensation submodels;
- aerodynamic functional form;
- hidden-state structure.

Therefore, residual disagreement after Stage HV is explicitly interpreted as a mixture of remaining topology/process/model-form differences, not as pure numerical error.

## Track A — CTIFL realised-action response and dose
Use the exact frozen R2 prepared cohorts:
- primary: 97 matched CTIFL events;
- strict sensitivity: 36 matched CTIFL events.

For each stage H and HV, each model is run at the same event-specific state/forcing used in R1.2 for:
- `u_pre`
- `u_post`

Primary horizon: 15 min.
Secondary horizon: 30 min.

Outputs:
- temperature response contrast;
- absolute-humidity response contrast;
- model-native DV;
- DN, using the stage-specific effective air height.

### Empirical-response metrics
For M1 and M2 separately:
- sign concordance with the frozen R1.2 matched observational contrast;
- Cohen kappa;
- normalized-effect Spearman association using the same R1.2 definition;
- date-cluster bootstrap 95% intervals, 2,000 resamples, seed 20261004.

These are compared descriptively with the frozen native R1.2 result. No harmonised stage is selected because it improves empirical alignment.

## Primary R3 cross-model endpoint
For each event and variable y in {T, AH}:

`D_y = |Delta_y,M1 − Delta_y,M2| / (|Delta_y,M1| + |Delta_y,M2| + eps_y)`

where `eps_y = 1e-12 × max(1, |Delta_y,M1|, |Delta_y,M2|)`.

Equal-weight response divergence:
`D_resp = 0.5 × (D_T + D_AH)`.

Primary R3 statistic:
- mean `D_resp` on the 97-event, 15-min cohort;
- Stage HV minus native difference;
- date-cluster bootstrap 95% interval.

Interpretation:
- CI below zero: common transport envelope materially reduces cross-model response divergence;
- CI spanning zero: no clear reduction;
- CI above zero: harmonisation increases divergence.

Stage H is a predeclared decomposition sensitivity.

Secondary cross-model response metric:
- fraction of events with opposite resolved M1/M2 response sign for T and AH.

## Track B — external physical-anchor replay
On the frozen R2 primary wind-dominated subset:
- 40 events, wind 3–6 m s-1.

Recompute the R2 external-anchor metrics for H and HV:
- DV median absolute log-ratio distortion;
- DV normalized absolute error;
- DN median absolute log-ratio distortion;
- DN normalized absolute error;
- model-specific and pooled.

Important mathematical check:
Once both models use the same effective air height 7.375 m, `DN = DV / 7.375` in both models. Therefore model-specific DV-versus-DN normalisation asymmetry must disappear up to numerical tolerance. This is an identity check, not a fitted outcome.

## Track C — D2 native-command decision-consequence replay
Use the frozen 61-event D2 input and exactly the EXP3.1 action grid:
`u = {0.1, 0.3, 0.5, 0.7, 0.9}`.

Run stages H and HV at horizons 15 and 30 min.

The EXP3.1 decision formulation is copied without change:
- observational-gradient benefit scaling;
- equal T/AH weights;
- cost `((u−0.1)/0.8)^2`;
- primary lambda domain [0,1];
- same lower-action tie rule;
- same analytic partition calculation;
- same date-cluster bootstrap.

Primary decision bridge:
- mean integrated decision-disagreement measure at 15 min, lambda in [0,1];
- fraction of events with any disagreement;
- locked native reference from EXP3.1 is used only for before/after comparison.

This D2 replay remains a mechanistic decision sensitivity because the D2 observational-gradient scale is not promoted to realised-action ground truth.

## Prespecified interpretation hierarchy
1. If Stage HV reduces CTIFL cross-model response divergence and reduces D2 decision disagreement, geometry/transport-envelope confounding is a material contributor.
2. If CTIFL divergence remains despite exact H and aperture matching, residual process/topology/model-form difference remains substantial.
3. If one empirical model alignment improves while the other worsens, this is reported as model-specific sensitivity, not a ranking.
4. R3 does not claim full Venlo conversion of CSGtom or full CSG conversion of GreenLight.
5. No new parameter is tuned to CTIFL observed T/AH responses.

## Sensitivity
- Stage H isolates air-volume normalisation.
- Stage HV is primary.
- frozen strict 36-event CTIFL cohort repeats the 15-min empirical/sign analysis.

No post-result geometry target or aperture ratio may replace the frozen CTIFL target.

## Runtime fail-closed gates
Fail only for:
- upstream artifact/input identity mismatch;
- stage parameter target mismatch;
- non-finite state/dose;
- requested/applied action mismatch;
- initial observable-state mismatch;
- unequal forcing between paired pre/post arms;
- missing/duplicate event/action/horizon rows;
- failure of common-height identity:
  `DN × 7.375 = DV` for every H/HV dose row within numerical tolerance.

Scientific improvement or deterioration is never a runtime pass criterion.

## Bootstrap
- cluster = calendar date
- 2,000 resamples
- seed = 20261004
- 95% percentile interval
