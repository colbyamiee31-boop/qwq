# PhysBench-GH R2 — Empirically Anchored Action Coordinate

## Protocol freeze
This file is committed before any R2 model-dose outcome or coordinate-comparison result is generated.

### Scientific question
Can the cross-model action coordinate be anchored to an external physical ventilation reference rather than selected by the amount of M1/M2 agreement?

### Frozen upstream identity
- R1.2 primary 97-event input SHA256: `236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051`
- R1.2 strict 36-event input SHA256: `f144244dda17c88f7e1f2ed1eecc62810fb7e4dce449602f231d41b5254b84f0`
- CTIFL raw archive MD5: `d0e4486fa1041fac5e6e47673b6c95d3`
- M1 GreenLight-Gym2 commit: `2d3febb1ea002b24b452e32293e990beb78d3ce1`
- M2 CSGtom commit: `bea8c3b0a1324162a4b5487db578aa674c8b587c`

### External empirical/semi-empirical anchor
The anchor is independent of M1/M2 agreement. It uses:
1. CTIFL realised leeward and windward roof-vent opening feedbacks;
2. CTIFL measured wind speed;
3. CTIFL greenhouse geometry;
4. wind-driven roof-vent functions reported by Sourisseau et al. (2026), inherited from experimentally determined greenhouse-ventilation relationships.

Primary anchor cohort is restricted to event wind speed 3–6 m s-1, the range for which the adopted wind-vent functions were experimentally determined. The source paper states that buoyancy becomes minor once wind speed exceeds approximately 2 m s-1.

### CTIFL geometry lock
- floor area A = 43.2 × 24 = 1036.8 m2
- principal roof vent dimensions = 4.05 × 1.40 m
- East-facing: 24 principal-equivalent vents
- West-facing: 18 principal vents + 12 vents of 1.35 × 1.40 m = 22 principal-equivalent vents
- primary symmetric equivalent count: 23 vents per logical leeward/windward bank
- maximum opening angle = 44 degrees
- geometric whole-greenhouse effective air height:
  H_CTIFL = 6.97 + 0.5 × (7.78 − 6.97) = 7.375 m

The 23/23 primary count avoids using outcome-dependent assumptions about dynamic assignment of the unequal East/West banks. Sensitivities bracket the count asymmetry with 22/24 and 24/22.

### Wind-driven ventilation functions
For opening angle delta in degrees:
- G_L(delta) = 2.46e-2 × (1 − exp(−delta/14.5))
- G_W(delta) = −1.89e-5 × delta^2 + 2.23e-3 × delta

For logical bank opening fractions v_L and v_W:
- delta_L = 44 × v_L
- delta_W = 44 × v_W

Primary specific wind flux:
F_emp = U × L_o × H_o / A_floor × [23 G_L(delta_L) + 23 G_W(delta_W)]

The external 15-min specific-volume dose is:
DV_emp = 900 × F_emp

The external air-volume-equivalent dose is:
DN_emp = DV_emp / 7.375

For each R1.2 event, the empirical action contrast is the post-opening arm minus the pre-opening arm under the same event wind speed:
- DeltaDV_emp = DV_emp(post) − DV_emp(pre)
- DeltaDN_emp = DN_emp(post) − DN_emp(pre)

This is a wind-dominated arm-equivalent physical reference, not measured airflow ground truth.

### Primary cohort
- R1.2 matched primary events with 3 <= event_Windsp <= 6 m s-1.
- Expected n = 40 before any R2 outcome.
- Both opening and closing events remain eligible.

### Predeclared sensitivity cohorts
1. Expanded wind range 2–8 m s-1 on the R1.2 primary matched cohort.
2. R1.2 strict 36-event cohort restricted to 3–6 m s-1 (expected n = 14).
3. CTIFL bank-count bracketing: 22/24 and 24/22 equivalent vents.
These sensitivities cannot replace the primary 3–6 m s-1 result.

### Model-native coordinate extraction
For each event, each model is rerun at the frozen R1.2 initial state and forcing for two 15-min arms:
- u_pre
- u_post

No CTIFL-specific model calibration is allowed.

M1 physical flux:
q_ext = fVentRoof + fVentSide + fVentForced = aux-state indices 136 + 137 + 145.
M1 DV is its 900-s quadrature; M1 DN = DV / native effective air height.

M2 physical flux:
Vent returned by `ctl_csg1`.
M2 DV is the 900-s native 30-s integration; M2 DN = DV / native effective air height.

Coordinate contrasts:
- DeltaU = u_post − u_pre
- DeltaDV_m = DV_m(post) − DV_m(pre)
- DeltaDN_m = DN_m(post) − DN_m(pre)

### Primary comparison metrics
Native command is dimensionless and is not assigned an absolute physical-error score.

For DV and DN, for M1 and M2 separately and pooled across both models:
1. sign concordance with the empirical anchor;
2. Spearman rank correlation;
3. median absolute log magnitude ratio:
   median |ln(|DeltaC_model| / |DeltaC_emp|)|,
   restricted to same-sign, non-zero contrasts;
4. through-origin scale slope:
   sum(DeltaC_emp × DeltaC_model) / sum(DeltaC_emp^2);
5. normalized absolute error:
   median |DeltaC_model − DeltaC_emp| / median |DeltaC_emp|.

For native DeltaU:
- sign concordance and Spearman rank association with DeltaDV_emp are reported descriptively.
- no unit-based absolute-error comparison is permitted.

### Coordinate-level empirical distortion
The primary coordinate comparison is the pooled (M1+M2) median absolute log magnitude ratio and pooled normalized absolute error.
Date-cluster bootstrap, 2,000 resamples, seed 20261004, provides 95% percentile intervals and the paired DN-minus-DV distortion difference.

Interpretation:
- lower distortion = closer absolute physical scaling to the external CTIFL wind-ventilation anchor.
- a CI for DN-minus-DV distortion below zero supports DN as less distorted; above zero supports DV as less distorted; crossing zero means the external anchor does not resolve the coordinate choice.

### New-coordinate firewall
No new coordinate may be promoted to the primary result after seeing R2 outcomes.
If both DV and DN remain materially distorted, a new coordinate may only be proposed as a separately frozen follow-on experiment.

### Runtime fail-closed gates
Fail only for:
- upstream SHA mismatch;
- CTIFL raw identity mismatch;
- missing or duplicate event IDs;
- action outside [0,1];
- requested/applied action mismatch;
- non-finite dose/state;
- initial observable-state mismatch;
- pre/post arms receiving different event forcing;
- physical-dose extraction inconsistency.

Scientific agreement/disagreement is never a runtime pass criterion.

### Interpretation boundary
Allowed:
- external physical anchoring of action-coordinate scale;
- coordinate-specific physical distortion;
- evidence that DV or DN is better/worse scaled to this wind-dominated CTIFL anchor.

Not allowed:
- calling the anchor measured airflow or ACH;
- global model ranking;
- claiming CTIFL geometry is representative of all greenhouse archetypes;
- selecting a coordinate because it maximizes M1/M2 agreement.

### Bootstrap
- date-cluster bootstrap
- 2,000 resamples
- seed 20261004
- 95% percentile intervals
