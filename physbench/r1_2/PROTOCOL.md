# R1.2 — CTIFL E5 REALISED-ACTION RESPONSE VALIDATION
## PROTOCOL FREEZE — BEFORE OBSERVED RESPONSE / MODEL-ALIGNMENT OUTCOMES

### Frozen upstream cohorts
- Primary R1.1 E5 cohort: 385 events
  - SHA256: `a9be7ae1bc23f621c9287c0a9360e6347ec9695519d13af5908bf23a65c7faca`
- Strict R1.1 E5 sensitivity cohort: 90 events
  - SHA256: `a22f7dae3f54dc92c9e8b14f8047613da236f032bed00ed6ed6e2ceff45102aa`
- Raw CTIFL archive MD5: `d0e4486fa1041fac5e6e47673b6c95d3`

### Scientific question
When the CTIFL greenhouse records a realised roof-vent position change, do frozen M1 GreenLight-Gym2 and M2 CSGtom predict the same **direction of temperature and moisture response** as the confounding-reduced observed event contrast?

This is an empirical realised-actuator-position response validation. It is **not** airflow/ACH validation and does not assume that the CTIFL vent-position fraction is a model-independent physical ventilation dose.

### Primary horizon and variables
- Primary horizon: **15 min**
- Secondary horizon: **30 min**
- Primary variables:
  1. indoor process air temperature, °C;
  2. indoor absolute humidity, g m^-3, deterministically converted from process temperature and RH.
- CO2 is descriptive only and not a primary validation variable because no synchronized outdoor CO2 channel is available.

### Realised action mapping
CTIFL provides separate leeward and windward roof-vent opening feedbacks (0–100%).

For each event:
`u_obs = (V_leeward + V_windward) / 200`

- `u_pre` = mean opening fraction at t-5 min;
- `u_post` = mean opening fraction at event time t0.
- Each frozen model receives `u_pre` and `u_post` directly as its native normalised ventilation command.
- No clipping or rescaling is allowed.
- Events fail closed if `u_pre` or `u_post` lies outside [0,1].

This mapping is a **vent-position-fraction mapping**, not a dose match. DV/DN physical-dose questions remain reserved for R2.

### Observational comparator
For event e and horizon h:
`Delta_event = y_e(t0+h) - y_e(t0)`

A matched no-vent-change control set is built without using post-event indoor response.

For each matched control c:
`Delta_control_c = y_c(t0+h) - y_c(t0)`

Primary observational contrast:
`Delta_obs = Delta_event - mean(Delta_control_c)`

At least 3 controls are required; up to 5 nearest controls are used.

### Control-pool eligibility
A control timestamp must:
- lie outside all R1.1 documented bad/imputed periods;
- have complete [-15,+30] min data;
- keep both realised roof-vent feedbacks stable within **2 percentage points** over [-15,+30] min;
- have no heating-pump status transition over [-15,+30] min;
- keep thermal and shading screen setpoints constant at stable modes;
- satisfy the same primary heating/weather stability limits used for R1.1;
- not fall within ±60 min of any primary R1.1 event.

### Matching — exact strata
Controls must match the event on:
- thermal-screen stable mode;
- shading-screen stable mode;
- Rails51 / Forcas / PE pump-status tuple;
- day/night class using measured GHI <20 vs >=20 W m^-2.

### Matching — calipers
- calendar distance <= 30 days;
- circular local-clock distance <= 120 min;
- pre-event mean vent fraction difference <= 0.10;
- indoor process T difference <= 1.5 °C;
- indoor AH difference <= 1.5 g m^-3;
- outdoor T difference <= 2.0 °C;
- outdoor humidity-ratio difference <= 0.0015 kg kg^-1;
- measured GHI difference <= 100 W m^-2 for day events;
- wind-speed difference <= 2.0 m s^-1;
- wind-direction circular difference <= 45° when event and control wind speeds are both >=1 m s^-1;
- 15-min pretrend difference <= 0.5 °C for T and <=0.75 g m^-3 for AH.

Among eligible controls, choose the five smallest standardized Euclidean distances using the matching variables above.
Control reuse is allowed but capped at 5 event matches.

### Model event setup
Frozen models:
- M1 GreenLight-Gym2 commit `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom commit `bea8c3b0a1324162a4b5487db578aa674c8b587c`

Event-specific initial observable state at t0:
- indoor process T;
- indoor vapor pressure from measured process T/RH;
- indoor CO2 from measured CTIFL CO2;
- canopy temperature = indoor process T (neutral closure);
- all other hidden states remain model-native.

Forcing:
- identical between u_pre and u_post arms;
- frozen at the event-time measured external conditions for the 30-min pair:
  GHI, outdoor T, outdoor vapor pressure, wind speed;
- outdoor CO2 = 415 ppm;
- sky-temperature proxy = outdoor T;
- soil-boundary temperature = 18 °C.

No model calibration or CTIFL-specific parameter fitting is allowed.

### Model contrast
For model m:
`Delta_model = y_post_command(h) - y_pre_command(h)`

This directly follows the observed vent-position change direction:
- opening event => u_post > u_pre;
- closing event => u_post < u_pre.

### Primary validation metrics at 15 min
For M1 and M2 separately, for T and AH:
1. sign concordance between `Delta_model` and `Delta_obs`;
2. 95% calendar-date cluster-bootstrap interval, 2,000 resamples, seed 20261004;
3. Cohen's kappa for chance-adjusted sign agreement;
4. cluster-bootstrap interval for kappa;
5. Spearman correlation between:
   `Delta_model / (|Delta_u| * |indoor-outdoor gradient|)`
   and the corresponding observational normalized contrast;
6. unresolved-zero count.

Model sign epsilon:
`1e-8 * max(1, |y_post|, |y_pre|)`

Observed sign unresolved if:
`|Delta_obs| <= 1e-12`.

### Secondary analyses
- same metrics at 30 min;
- opening and closing events separately;
- day and night separately;
- repeat full 15-min sign analysis on the frozen 90-event strict R1.1 cohort;
- descriptive conventional-direction score:
  opening toward outdoor / closing away from outdoor.

No subgroup may replace the matched all-condition primary result.

### Statistical comparison between models
Report M1-minus-M2 concordance difference with date-cluster bootstrap interval.
This is cohort-specific comparative alignment, not a global model ranking.

### Runtime and analysis fail-closed gates
Fail if:
- upstream cohort SHA mismatches;
- raw CTIFL archive identity mismatches;
- any selected event/control uses a documented bad period;
- fewer than 3 controls for an event included in primary analysis;
- matching uses any post-event indoor outcome;
- requested/applied model action mismatch;
- non-finite model state/output;
- initial observable-state mismatch beyond tolerance;
- u_pre/u_post outside [0,1];
- HIGH/LOW or pre/post arms receive different forcing;
- event IDs duplicate or disappear after model execution.

### Interpretation boundary
Allowed:
- realised vent-position response validation;
- agreement/disagreement with confounding-reduced empirical event contrast;
- cohort-specific comparison of M1 and M2;
- opening/closing and day/night heterogeneity.

Not allowed:
- measured airflow/ACH validation;
- randomised causal effect;
- proof that DV or DN is the true physical action coordinate;
- global model ranking.

### Freeze status
This file is frozen before any R1.2 observed-response contrast, sign-concordance result, kappa, correlation, or model outcome is calculated.
