# EXP2.1D — Representation / Mapping Robustness

Frozen before robustness outcomes.

EXP2.1D tests whether the weak observational model alignment observed in EXP2.1C is stable to predeclared representation choices. It is not actuator calibration, model retuning, causal validation, realized-action validation, or a search for a mapping that maximizes agreement.

The primary cohort remains exactly the 61 EXP2.1B/EXP2.1C higher-VentLee matched events: 17 day and 44 night. No event is added, removed, reweighted, or selected by model response. Primary horizon remains 15 min; 30 min is secondary. Day/night remains secondary.

A full factorial grid is evaluated: 4 command mappings × 3 canopy closures × 2 forcing representations = 24 scenarios per model.

Command mappings:
1. STD_0p1_0p9: LOW=0.1, HIGH=0.9; the EXP2.1C anchor direction probe.
2. ABS_LINEAR: LOW=status_before/100, HIGH=status_after/100.
3. ABS_COMPRESSED: LOW=0.2+0.6(status_before/100), HIGH=0.2+0.6(status_after/100).
4. DELTA_FROM_0p3: LOW=0.3, HIGH=min(1,0.3+status_delta/100).

All mappings are monotone, constrained to [0,1], and HIGH>LOW. None is asserted to equal realized vent area or airflow.

Canopy-temperature closures:
- Tcan=Tair-0.5 C
- Tcan=Tair
- Tcan=Tair+0.5 C
All other non-common hidden states remain model-native.

Forcing representations use no post-event weather:
- SNAPSHOT_T0: constant event-time pre-action weather snapshot, the EXP2.1C anchor.
- PRE15_MEDIAN: constant median of t-15, t-10, t-5 and t0 provider weather records.

Frozen models:
- M1 GreenLight-Gym2 commit 2d3febb1ea002b24b452e32293e990beb78d3ce1.
- M2 CSGtom commit bea8c3b0a1324162a4b5487db578aa674c8b587c.
No equation changes, calibration, fitting, or model-specific retuning.

At 15 min, for every model × scenario × variable T/AH report sign concordance, calendar-date cluster-bootstrap 95% interval using 2000 resamples with seed 20261002, Spearman rho with cluster-bootstrap interval, median normalized model response, and difference from the EXP2.1C anchor. Secondary outputs are 30-min all-condition metrics, 15-min day/night strata, and cross-model sign agreement.

Fail closed if cohort identity changes, pre-event forcing is incomplete, a mapping violates [0,1] or HIGH>LOW, the anchor scenario fails to reproduce the EXP2.1C event-response fingerprint, any output is non-finite, or requested/applied ventilation differs.

Allowed language: representation robustness, observational model alignment, sensitivity to command mapping, hidden-state closure, pre-event forcing representation.
Not allowed: causal actuator effect, physical dose calibration, realized-action validation, preferred mapping selected by agreement, or global model ranking.
