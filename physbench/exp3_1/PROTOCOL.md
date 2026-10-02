# EXP3.1 — Decision-Consequence Test: model-dependent ventilation decisions

Frozen before EXP3.1 outcomes.

## Scientific question
Under the same 61 D2 greenhouse operating contexts used in EXP2.1B–D, do frozen M1 GreenLight-Gym2 and M2 CSGtom recommend different natural-ventilation commands when each model is used as the decision model, and what cross-model consequence follows from those action differences?

EXP3.1 is the third PhysBench-GH benchmark level: **decision consequence**. It does not use D2 actuator status as ground-truth physical dose and does not claim realized-actuator or causal validation.

## Frozen contexts
- Exactly the same 61 locked D2 primary contexts from EXP2.1B/EXP2.1C.
- 17 day and 44 night.
- No event is added, removed, reweighted, or selected by model output.
- D2 contributes only pre-event state/weather context and indoor-outdoor gradients; D2 post-event response is not used to choose actions.

## Frozen models
- M1 GreenLight-Gym2 commit \`2d3febb1ea002b24b452e32293e990beb78d3ce1\`.
- M2 CSGtom commit \`bea8c3b0a1324162a4b5487db578aa674c8b587c\`.
- No calibration, parameter tuning, equation changes, or D2-specific retuning.

## Event initialization and forcing
Use the exact EXP2.1C anchor representation:
- indoor T, vapor pressure, and CO2 from the frozen pre-event D2 context;
- canopy temperature closure: Tcan = Tair;
- all non-common hidden states remain model-native;
- forcing is the constant pre-event snapshot: Iglob, Tout, outdoor vapor pressure, wind;
- outdoor CO2 = 415 ppm, sky-temperature proxy = Tout, soil-boundary temperature = 18 °C;
- no post-event forcing.

Primary horizon: 15 min. Secondary horizon: 30 min.

## Candidate action set
The controller chooses one command from:
\[
U = \{0.1,0.3,0.5,0.7,0.9\}.
\]
These are native normalized commands and are **not** asserted to be equal airflow, ACH, or vent-area fractions across M1 and M2.

The reference action is \(u_0=0.1\).

## Standardized decision benefit
For each event i, model m, action u, horizon h, and variable y in {T, AH}:

\[
b_{miy}(u)=
-\operatorname{sign}(G_{iy})
\frac{y_{mi}(u,h)-y_{mi}(u_0,h)}
{|G_{iy}|},
\]

where \(G_{iT}=T_{in}-T_{out}\) and \(G_{iAH}=AH_{in}-AH_{out}\), both frozen from D2.

All 61 frozen contexts have non-zero T and AH gradients, so no denominator floor is introduced.

The equal-weight aggregate benefit is:
\[
B_{mi}(u)=0.5\,b_{miT}(u)+0.5\,b_{miAH}(u).
\]

Positive benefit means the model predicts motion toward the outdoor state relative to the reference action.

## Intervention penalty and decision rule
Define normalized intervention cost:
\[
C(u)=\left(\frac{u-u_0}{0.8}\right)^2.
\]

For penalty ratio \(\lambda\):
\[
Q_{mi}(u;\lambda)=B_{mi}(u)-\lambda C(u).
\]

The chosen action is:
\[
u^*_{mi}(\lambda)=\arg\max_{u\in U} Q_{mi}(u;\lambda).
\]

Tie rule: if utilities are equal within \(10^{-12}\), choose the smaller ventilation command.

### Primary penalty domain
\[
\lambda\in[0,1].
\]

Because benefit and cost are dimensionless, this domain spans from no intervention penalty to a full-scale penalty comparable to one unit of normalized environmental benefit.

No single \(\lambda\) is selected as the primary result. Decision regions are integrated exactly over the full domain using the analytic pairwise line-intersection breakpoints of the finite action set.

Secondary domain: \([0,3]\).

## Primary decision-consequence metrics
At 15 min and \(\lambda\in[0,1]\):

1. **Decision-disagreement measure**
\[
D_i=\int_0^1 I[u^*_{M1,i}(\lambda)\neq u^*_{M2,i}(\lambda)]\,d\lambda.
\]
Report mean, median, calendar-date cluster-bootstrap 95% interval for the mean, and fraction of events with any disagreement.

2. **Integrated normalized action gap**
\[
A_i=\int_0^1
\frac{|u^*_{M1,i}(\lambda)-u^*_{M2,i}(\lambda)|}{0.8}\,d\lambda.
\]

3. **Symmetric cross-model regret**
Each model's chosen action is evaluated in the other model's utility. Regret is normalized by the evaluator model's within-action utility range at that lambda. The two directions are averaged and integrated over lambda.

4. **Symmetric predicted outcome divergence**
When the selected actions differ, quantify the normalized T/AH consequence of applying the two selected actions under each evaluator model, then average the two evaluators and integrate over lambda.

## Secondary analyses
- 30-min horizon over \([0,1]\);
- 15-min day/night strata;
- secondary penalty domain \([0,3]\);
- descriptive disagreement fractions at lambda = 0, 0.05, 0.10, 0.25, 0.50, 1.00.

No post-hoc lambda or subgroup may replace the all-condition integrated primary result.

## Runtime fail-closed gates
Fail if:
- event identity/count differs from 61;
- any model output is non-finite;
- requested/applied action mismatch occurs;
- initial common state differs beyond numerical tolerance;
- action grid differs between models;
- anchor u=0.1 response cannot be generated;
- analytic decision partition does not cover the full lambda domain exactly within 1e-12.

## Interpretation boundary
Allowed: model-dependent decision reversal/disagreement; decision stability; cross-model regret; standardized control consequence; consequence under model swap.

Not allowed: one model is the true controller; real-world optimal action; realized-actuator validation; causal controller performance; global model ranking.
