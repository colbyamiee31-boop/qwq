# PhysBench-GH R4 — Aerodynamic Transport Formulation Isolation

## Protocol freeze
This file is committed before any R4 common-flux model response, empirical-alignment, or decision-consequence outcome is generated.

## Scientific question
After R3 harmonised the directly comparable physical envelope, how much of the remaining M1/M2 disagreement is caused by the models' different natural-ventilation aerodynamic formulations, and how much persists downstream in greenhouse topology / thermal-moisture process structure?

R4 does not tune Cd, Cw, leakage, vent height or any other aerodynamic coefficient to improve agreement. Instead it uses a stronger isolation: both models are driven by the same externally defined CTIFL roof-vent flux.

## Frozen ancestry
- R3 final clean head: `772d5fc1eb9079d1a0abac7e7b8707624dc7963e`
- R3 final clean run: `37163085468`
- R3 final artifact SHA256: `e2d8540dfdcbe5812d7e9f3011b70c96e32e48e7ec925fb88c05d3646c79a008`
- R2 successful run: `37145919425`
- R1.2 successful run: `37143141817`
- D2 61-event input SHA256: `8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023`
- M1 GreenLight-Gym2 commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

## R3 envelope retained
All R4 model runs use the R3 HV envelope:
- effective air height = 7.375 m in both models;
- projected maximum roof-aperture ratio = 0.17474999631859142 m2 m-2 in both models.

All non-aerodynamic model structure remains native.

## Why coefficient matching is not the primary isolation
M1 and M2 use non-equivalent natural-ventilation equations:
- M1 combines a discharge coefficient, wind shelter / pressure coefficient, buoyancy term, roof/top compartment exchange and leakage split;
- M2 uses a different top-vent equation with its own Cd, Cw, htop and leakage law.

Because the coefficients do not have guaranteed one-to-one semantics across the equations, setting the same numeric Cd or Cw would not constitute a clean physical harmonisation.

Therefore R4 bypasses the native vent-flow calculation rather than numerically matching non-equivalent coefficients.

## External common aerodynamic flux
The common action-dependent roof-vent flux is the frozen R2 CTIFL wind-driven reference.

For opening angle delta in degrees:
- G_L(delta) = 2.46e-2 * (1 - exp(-delta/14.5))
- G_W(delta) = -1.89e-5 * delta^2 + 2.23e-3 * delta

CTIFL geometry:
- floor area = 1036.8 m2
- principal vent = 4.05 x 1.40 m
- symmetric logical vent-bank count = 23 / 23
- maximum opening angle = 44 degrees

For leeward and windward opening fractions v_L, v_W in [0,1] and measured wind speed U:

q_common = U * 4.05 * 1.40 / 1036.8 *
           [23*G_L(44*v_L) + 23*G_W(44*v_W)]

Units: m3 m-2 s-1.

This is an external semi-empirical wind-driven roof-vent flux. It is not measured airflow / ACH.

## Primary empirical cohort
Primary R4 cohort is the frozen R2 wind-dominated subset:
- 40 R1.2 matched CTIFL events;
- event wind speed 3–6 m s-1;
- 15-min primary horizon;
- actual CTIFL leeward and windward feedbacks are used separately for the pre and post arms.

For each event:
- q_pre is computed from actual pre leeward/windward feedback;
- q_post is computed from actual post leeward/windward feedback.

The exact same q_pre and q_post values are supplied to M1 and M2.

## Primary R4 stage — CF0
`CF0` = common externally prescribed roof-vent flux with model-native external leakage suppressed in both models.

Purpose:
- remove native vent aerodynamic equations;
- remove leakage-law differences;
- retain each model's downstream air-zone topology and thermal/moisture equations.

M1 routing:
- q_common is applied as roof-to-outdoor exchange through the model's TopAir pathway;
- direct MainAir-to-outdoor ventilation is set to zero;
- native MainAir↔TopAir mixing remains unchanged.

M2 routing:
- q_common replaces the `Vent` value returned by the native ventilation controller;
- the native one-zone downstream heat, vapour and CO2 equations consume q_common directly.

This routing intentionally preserves each model's zone topology. Residual disagreement after CF0 therefore includes topology and downstream process structure, but not the native roof-vent flow formula or leakage law.

## Leakage sensitivity — CFN
`CFN` is secondary:
- the same common action-dependent q_common is used;
- each model's native closed-vent leakage contribution is added as a constant baseline within each arm.

CFN is not allowed to replace CF0 as the primary result.
Its purpose is to determine whether removing leakage in CF0 materially changes the conclusion.

## Native aerodynamic reference
The R3 HV result is the frozen native-aerodynamics reference and is not refitted:
- same 97-event CTIFL response archive;
- same empirical alignment;
- same D2 decision replay.

For the primary 40-event R4 comparison, the R3 HV responses are subset to exactly the same event IDs.

## Common-flux D2 mechanistic replay
A secondary mechanistic replay uses the frozen D2 events whose event wind lies in 3–6 m s-1.

Because D2 does not contain separate windward/leeward realised-feedback arms, each native action u is mapped symmetrically:
- v_L = u
- v_W = u

Both models then receive the same q_common(u, wind) under the R3 common air-volume envelope.

The frozen EXP3.1 decision formulation is copied unchanged:
- actions u = {0.1,0.3,0.5,0.7,0.9};
- observational-gradient benefit scaling;
- equal T/AH weights;
- cost ((u-0.1)/0.8)^2;
- primary lambda in [0,1];
- lower-action tie rule;
- exact analytic decision partitions.

This replay remains mechanistic and is not realised-action decision ground truth.

## Primary endpoints

### A. Aerodynamic contribution to cross-model response divergence
For each primary CTIFL event and y in {T, AH}:

D_y = |Delta_y,M1 - Delta_y,M2| /
      (|Delta_y,M1| + |Delta_y,M2| + eps_y)

D_resp = 0.5*(D_T + D_AH)

Primary statistic:
- mean D_resp under R3-HV native aerodynamics;
- mean D_resp under CF0;
- CF0 minus HV difference;
- calendar-date cluster-bootstrap 95% CI.

Interpretation:
- CI below zero: native aerodynamic formulation materially contributes to residual response divergence;
- CI spanning zero: no clear aerodynamic contribution;
- CI above zero: common flux increases downstream divergence.

Also report T and AH response-sign disagreement fractions.

### B. Downstream residual after aerodynamic removal
The CF0 D_resp itself is the downstream residual.

If CF0 remains large and/or response-sign disagreement remains frequent, the residual cannot be attributed to the native vent-flow equations or leakage law.

### C. Empirical realised-action alignment under common flux
For M1 and M2 separately:
- sign concordance with frozen R1.2 matched observational contrasts;
- Cohen kappa;
- normalized-effect Spearman rho;
- 2,000 date-cluster bootstrap 95% intervals.

A reduction in cross-model divergence is not called an improvement if empirical alignment worsens.

### D. Common-flux physical identity
For every CF0 arm:
- model-integrated DV must equal q_common * horizon_seconds within numerical tolerance;
- DN = DV / 7.375;
- the prescribed q_common must be identical between M1 and M2 for the same event/arm.

This is a runtime identity, not a scientific outcome.

### E. Secondary D2 decision residual
Report:
- mean integrated decision disagreement;
- fraction of events with any disagreement;
- locked events 27/86/89 only if they are inside the 3–6 m s-1 D2 subset.

No D2 result can override the primary realised-action response result.

## Bootstrap
- cluster = calendar date
- 2,000 resamples
- seed = 20261004
- 95% percentile interval.

## Runtime fail-closed gates
Fail only for:
- upstream artifact identity mismatch;
- missing or duplicate events;
- CF0 q_common mismatch between M1 and M2;
- q_common outside finite non-negative domain;
- non-finite state/output;
- initial observable-state mismatch;
- unequal forcing between paired pre/post arms;
- common-height / aperture target mismatch;
- common-flux DV identity failure;
- output row-count mismatch.

Scientific improvement or deterioration is never a runtime PASS criterion.

## Interpretation hierarchy
1. If CF0 markedly reduces cross-model response divergence and empirical alignment is preserved/improved, native aerodynamic formulation is an important contributor.
2. If CF0 reduces magnitude divergence but sign disagreement persists, aerodynamics explains scaling more than response direction.
3. If CF0 leaves large divergence, downstream topology / heat-moisture process structure dominates the remaining discrepancy.
4. If models converge under CF0 but both align worse with observations, convergence is not validation.
5. R4 does not establish a global model ranking.
6. R4 does not call q_common measured airflow.

## No-tuning firewall
After this protocol freeze:
- no Cd/Cw/leakage/vent-height coefficient may be fitted;
- no alternative q_common equation may replace the frozen CTIFL reference;
- no wind range may replace the 3–6 m s-1 primary cohort;
- no model-specific scaling multiplier may be introduced.
