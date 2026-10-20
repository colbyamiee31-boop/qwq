# R5: Field-identified ventilation flux in GreenLight (Codex execution protocol)

**Authority:** Claude (scientific/data). Execute exactly as written. The flux function and its parameters are fixed inputs and must not be refitted. Scientific improvement or deterioration is never a pass/fail criterion; only the runtime gates in Section 6 can fail a run.

## 1. Question

Does replacing GreenLight's native roof-vent flow law with the ventilation function identified from routine CTIFL vent events (multi-tracer UDE) improve GreenLight's predicted 15- and 30-min responses to realised roof-vent actions? Does an additional sensible-heat exchange factor (s_T = 0.34) improve the temperature response further?

## 2. Model and cohort

- **Model:** M1 = GreenLight-Gym2 at the frozen commit used in PhysBench R3/R4, under the R3 common air-volume envelope (effective height 7.375 m, roof-aperture capacity 0.17475 m² m⁻²).
- **Model scope:** CSGtom (M2) is **not** run. It is a Chinese-solar-greenhouse model and is out of domain for the CTIFL Venlo greenhouse.
- **Primary cohort:** the 97 R1.2 matched events, listed in `inputs/R5_event_list_97.csv`.
- **Secondary cohort:** the 40-event subset with `in_R4_wind_3_6_subset == True`. This subset is directly comparable with the frozen R4 CF0 results.
- **Initialisation, forcing and pre/post arms:** use the R1.2/R3 event archive unchanged. Use realised leeward and windward feedback separately for the pre arm and the post arm, identical forcing between paired arms, and the same initial observable state.
- **Horizons:** 15 min (primary) and 30 min.

## 3. Conditions (M1 only)

| Code | Roof-vent exchange (TopAir → outdoor, R4 routing) | Leakage | Notes |
|---|---|---|---|
| `HV` | native GreenLight aerodynamics | native | frozen R3 result; re-use, do not rerun |
| `REF0` | q_ref, the De Jong/Sourisseau reference flux (R4 formula) | suppressed | identical to frozen R4 CF0 on the 40-event subset; run on all 97 for the primary cohort |
| `FF0` | **q_field** from `inputs/field_flux.py` | suppressed | **primary new condition** |
| `FFN` | q_field | native closed-vent leakage added as a constant baseline | leakage sensitivity, as R4 CFN |
| `FFH` | q_field for vapour and CO₂; q_field × **0.3412** for the **sensible-heat** term of the TopAir→outdoor exchange only | suppressed | exploratory structural test of the heat-exchange factor |

### Routing

Routing is identical to R4 CF0:
- Apply the prescribed flux through the TopAir → outdoor pathway.
- Set direct MainAir → outdoor ventilation to zero.
- Leave native MainAir ↔ TopAir mixing unchanged.

### Computing q_field and q_ref

Evaluate both fluxes per arm from:
- the arm's realised vL and vW (fractions, 0–1);
- event wind speed U;
- indoor and outdoor temperature at the start of the arm.

q_ref is the R4 formula:

```
q_ref = U * 4.05 * 1.40 / 1036.8 * [23 * G_L(44 * vL) + 23 * G_W(44 * vW)]
```

### FFH implementation

In FFH, only the sensible-heat flux carried by roof ventilation, ρ c_p q (T_top − T_out), is multiplied by 0.3412. The vapour and CO₂ fluxes use the unscaled q_field. Record the code location of this change in the run log.

## 4. Inputs (authoritative)

- `inputs/field_ventilation_function.json`: parameters of the field-identified function. Treat as read-only.
- `inputs/field_flux.py`: reference implementation. Run `python -I field_flux.py`; the unit tests must pass before any model run. Import this function rather than re-implementing it.
- `inputs/R5_event_list_97.csv`: event IDs, rows, directions, realised vent positions and the 40-subset flag.
- **Observed matched effects (T, AH; 15/30 min):** the frozen R1.2 matched contrasts already used in R3/R4. These are not copied here; use the existing archive.

## 5. Endpoints

All endpoints are computed for `y ∈ {T, AH}` and both horizons, per condition and cohort.

1. **Intervention fidelity (new primary).** Fidelity = (mean of the signed model response) / (mean of the signed observed matched effect).
   - Sign each event by its expected physical direction: −sign(Δvent) × sign(y_in − y_out) at the last pre-event sample. This is the same gradient-conditioned direction used in R1.2/R4. Exclude events with |y_in − y_out| below the R1.2 resolution threshold, and report how many were excluded.
   - Fidelity = 1 means the correct average magnitude.
   - Report the point estimate and a 95% date-cluster bootstrap interval (2,000 resamples, seed 20261019).
2. **Event-level RMSE** of model response against the observed matched effect, with a bootstrap interval.
3. **R4 alignment metrics:** sign concordance, Cohen's κ and normalised-effect Spearman ρ, with a 95% date-cluster bootstrap.
4. **Paired differences** with paired date-cluster bootstrap intervals, for:
   - FF0 − REF0
   - FF0 − HV
   - FFH − FF0

   Compute each difference for |fidelity − 1| and for RMSE.

## 6. Runtime fail-closed gates (only these may fail a run)

- `field_flux.py` unit tests fail.
- Event IDs missing or duplicated, or the count is not 97 (primary) or 40 (subset).
- A prescribed q is non-finite or negative.
- The flux-identity check fails: the model-integrated ventilated volume must equal q × horizon seconds, within numerical tolerance, for every REF0/FF0/FFN/FFH arm.
- Non-finite model state or output.
- Mismatch in initial observable state or forcing between paired arms.
- Output row count does not equal events × conditions × horizons.

## 7. Outputs

- `R5_event_responses.csv`: one row per event × condition × horizon. Columns: event_id, date, direction, condition, horizon_min, dT_model, dAH_model, dT_obs, dAH_obs, q_pre, q_post.
- `R5_summary.csv`: endpoints from Section 5 with confidence intervals, for each condition, cohort, variable and horizon.
- `R5_paired_differences.csv`.
- `R5_run_log.md`: commit IDs, input SHA-256 values, the FFH code location and the result of every gate.

Do **not** interpret the results; return the files to Claude for the scientific audit.

## 8. What Claude expects (recorded before the run, for transparency)

These expectations do not change any gate:
- **Absolute humidity:** FF0 fidelity should be closer to 1 than REF0, because the field function was identified from mass tracers.
- **Temperature:** REF0 and FF0 should still over-predict the temperature response; FFH should reduce that over-prediction.

If the results contradict these expectations, they will be reported as found.
