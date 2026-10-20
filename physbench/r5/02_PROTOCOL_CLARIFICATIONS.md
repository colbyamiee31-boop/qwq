# R5 protocol clarifications (pre-outcome, 2026-10-20)

These items make `00_R5_CODEX_PROTOCOL.md` executable against the frozen artifacts. They were written before any R5 model response was generated.

They do not change the question, the cohort, the field flux parameters, the conditions or the endpoints. Where the original wording was ambiguous, each item fixes the meaning to the existing R1.2–R4 implementation.

**C1 Input.** "R1.2/R3 event archive" means `PRIMARY_97_R2_INPUT.csv` from artifact `r4-prepared-inputs` (ID 11289046331). Initial state, weather forcing, outdoor CO2 (415 ppm fixed) and the pre/post arm construction are taken unchanged from it, exactly as in `physbench/r4/m1_common_flux.py` (`initial_state`, `build_weather`, `run_arm`).

**C2 Observed effects.** The observed matched effects are `obs_T_matched_15`, `obs_AH_matched_15`, `obs_T_matched_30` and `obs_AH_matched_30` in the same file. Each is the event change minus the mean change of its matched controls, as computed in `physbench/r1_2/prepare_inputs.py`.

**C3 Model response.** Model response = post-arm state minus pre-arm state at the horizon, as in R3/R4.

**C4 Signing and resolution, intervention fidelity.**
- The expected-direction sign of event e is s_e = −sign(delta_u) × sign(gradient), with gradient = `T_gradient_C` for T and `AH_gradient_g_m3` for AH.
- An event is excluded from the fidelity endpoint for that variable when |gradient × delta_u| ≤ 1e-12, the R1.2 normalisation guard in `physbench/r1_2/aggregate.py`. Report how many events are excluded.
- Fidelity = mean(s_e × model_delta) / mean(s_e × obs_delta) over the retained events.
- The concordance, kappa and normalised-Spearman endpoints use the R1.2 rules unchanged: an event is resolved when sign_model ≠ 0 (eps = 1e-8 × scale) and sign_obs ≠ 0 (1e-12).

**C5 Secondary cohort.** The secondary cohort is `anchor_primary_3_6 == True` in the frozen input, which gives 40 events. See 01 §3 for the v1 flag correction.

**C6 Reused references.**
- HV comes from `r3-m1/.../ctifl_event_responses.csv`, with stage == 'HV', model M1, cohort primary, horizons 15 and 30. It is reused, not rerun.
- REF0 at 15 min on the 40-event subset must equal frozen R4 CF0 (`r4-m1/.../ctifl_common_flux_responses.csv`, stage CF0, cohort primary). This is checked as the reproduction gate in C8.
- REF0 on the other 57 events, and REF0 at 30 min for all 97, are new runs with the same code.

**C7 30-min horizon.** The 30-min response is obtained by integrating the same constant-flux arm for 1800 s, as two consecutive 900-s integrations. The forcing is held exactly as for 15 min. R4 itself reported 15 min only.

**C8 Reproduction gate (fail-closed, before any FF run).** In the R5 environment, rerun REF0 = R4 CF0 for the 40-event subset at 15 min, then compare with frozen `r4-m1` CF0.
- Pass: max |ΔT_model difference| ≤ 1e-6 °C and max |ΔAH_model difference| ≤ 1e-6 g m⁻³, together with all R4 runtime audits.
- Fail: stop and report. Do not run FF0/FFN/FFH, and do not loosen the tolerance without Claude's sign-off.

**C9 FFH.** The heat factor 0.3412 multiplies only the term `sensible(cp_rho*qtop, x[3], d[1])` for the prescribed TopAir→outdoor flux (`qmain = 0` in CF0 routing). The vapour and CO2 terms use the unscaled q_field.

**C10 q_field evaluation per arm.** Evaluate q_field from:
- vL = leeward_{pre|post}_pct / 100 and vW = windward_{pre|post}_pct / 100;
- U = event_Windsp;
- Tin = event_Tair and Tout = event_Tout.

Use `field_flux.q_field` imported unchanged (SHA256 `b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd`). The flux-identity gate compares the integrated ventilated volume with q × horizon seconds.
