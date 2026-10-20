# R5 inputs supplement: provenance of the frozen inputs and reference results

**Authority:** Claude (scientific/data), 2026-10-20. No R5 outcome has been generated. Nothing in this file changes the R5 design, event set, parameters or endpoints.

## 1. Codex's block is correct

The handoff package `R5_handoff` contained only the event list, the field flux and the protocol. The protocol points to "the R1.2/R3 event archive" and "frozen R3 HV / R4 CF0 results", but it does not say where these live. They exist. They are GitHub Actions artifacts of the same project that ran R1.2–R4:

- repository: `colbyamiee31-boop/qwq`
- Claude verified on 2026-10-20, through the GitHub API, that every artifact listed below exists, is not expired (expiry 2027-01-01) and has the digest shown. Claude could not download the zip contents because the cloud sandbox blocks the artifact storage host. The user's machine can download them.

## 2. Artifacts R5 needs

| Purpose in R5 | Artifact name | Artifact ID | Workflow run | Branch | SHA256 of artifact zip |
|---|---|---|---|---|---|
| **Primary input**: per-event forcing, initial state, leeward/windward pre/post %, wind, observed matched effects (T, AH; 15 and 30 min), `anchor_primary_3_6`, empirical reference flux | `r4-prepared-inputs` (identical content to `r2-prepared-inputs`) | 11289046331 | 37164458429 | r4-aerodynamic-transport-isolation | `ecff6bc7586f07aac393c26698488cd98164bee418f0a71fdae9f60fe4fafe0f` |
| Same file, original producer (cross-check) | `r2-prepared-inputs` | 11281134934 | 37145919425 | r2-empirically-anchored-action-coordinate | `5b2c6d98ea17b916e9d4ccd490dce2a3c3280b1d4d7453a5c6f732f1953ec66e` |
| Matched-control construction (source of the observed effects) | `r1-2-prepared-inputs` | 11281695347 | 37143141817 | r1-2-ctifl-realised-action-validation | `2230f9e6e8f197e4272d24ec0fb071ba4784f5b3f779c2e8e1c4503db547cc4d` |
| **HV reference** (frozen, reuse): M1 per-event responses, stages H and HV, horizons 15/30 min | `r3-m1` | 11288765689 | 37163085468 | r3-common-physical-envelope | `de6ae266728fed2ed535a5f591c5c1778b74ee75a750ddfb4f65ae32be1d0979` |
| R3 aggregate (cross-check) | `r3-final` | 11288054244 | 37163085468 | r3-common-physical-envelope | `e2d8540dfdcbe5812d7e9f3011b70c96e32e48e7ec925fb88c05d3646c79a008` |
| **REF0 reference on the 40-event subset** (frozen R4 CF0, reuse) + M1 pip freeze | `r4-m1` | 11289405542 | 37164458429 | r4-aerodynamic-transport-isolation | `8a9867cfd9c1935668a4123990595b8085bed03daeb5d9a4e49e07f1e06fd47c` |
| R4 aggregate (cross-check) | `r4-final` | 11289580354 | 37164458429 | r4-aerodynamic-transport-isolation | `9d9d16a0efd6ea8afb983a553b8860c1673ae64257692148924ae3949c219f7d` |

Files expected inside (from the producing scripts on those branches):

- `r4-prepared-inputs/PRIMARY_97_R2_INPUT.csv`: the R5 primary input. Key columns are `event_id, row_index, event_date, direction, u_pre, u_post, delta_u, leeward_pre_pct, leeward_post_pct, windward_pre_pct, windward_post_pct, event_Tair, in_vp_pa, CO2_pre_ppm, event_Iglob, event_Tout, out_vp_pa, event_Windsp, outdoor_co2_ppm_fixed (=415), sky_temperature_c_proxy, soil_boundary_temperature_c_fixed, T_gradient_C, AH_gradient_g_m3, obs_T_matched_15, obs_AH_matched_15, obs_T_matched_30, obs_AH_matched_30, n_controls, emp_flux_pre_23_23, emp_flux_post_23_23, anchor_primary_3_6`.
- `r1-2-prepared-inputs/PRIMARY_97_MODEL_INPUT.csv`: the same 97 rows before the R2 flux columns were added. Its SHA256 is asserted in `physbench/r1_2/prepare_inputs.py`: `236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051`.
- `r3-m1/evidence/R3/M1/ctifl_event_responses.csv`: per-event M1 responses (stages H, HV; cohorts primary 97, strict 36; horizons 15, 30).
- `r4-m1/evidence/R4/M1/ctifl_common_flux_responses.csv`: per-event M1 responses (stages CF0, CFN; 40 + 14 events; horizon 15). Also `evidence_R4_M1_PIP_FREEZE.txt`.

## 3. Mapping to the 97 R5 events

- Join `R5_event_list_97_v2.csv` to `PRIMARY_97_R2_INPUT.csv` on **`row_index`** (authoritative), then assert that `event_id` also matches.
- Gate: 97/97 rows must match on both keys, with no duplicates. Any mismatch stops the run.
- **Correction in v2 of the event list.** The 40-event subset flag is now defined exactly as the frozen R2/R4 `anchor_primary_3_6`, i.e. 3 ≤ wind speed at the event row ≤ 6 m/s. v1 used the event-window median wind and swapped one event pair (CTIFL-E5-P-0166 in, CTIFL-E5-P-0270 out). In R5, the subset is read from `anchor_primary_3_6` in the frozen input. The flag in the event list is only a cross-check and must agree 40/40.

## 4. Frozen model and environment (from the R3/R4 workflows)

- M1 GreenLight-Gym2: `https://github.com/BartvLaatum/GreenLight-Gym2.git`, commit **`2d3febb1ea002b24b452e32293e990beb78d3ce1`**.
- Python 3.11. `physbench/r1_2/requirements-m1.txt` pins: casadi==3.8.1, cloudpickle==3.1.2, gymnasium==0.29.0, numpy==1.26.4, pandas==2.2.3, scipy==1.14.1. GreenLight-Gym2 itself is installed from the pinned checkout.
- R3/R4 ran on GitHub-hosted `ubuntu-24.04` runners.
- R5 code should be adapted from `physbench/r4/common.py` and `physbench/r4/m1_common_flux.py` on branch `r4-aerodynamic-transport-isolation`. The adaptation should only replace q_common with q_field and add the FFH heat factor and the 30-min horizon.

## 5. What R5 cannot use

- The local copy `E:\GreenLight_R5\...\GreenLight-Gym2-master` has no git metadata. It may only be used after it is shown to be byte-identical to commit 2d3febb (see 03_CODEX_EXECUTION_INSTRUCTIONS.md, Route B, step B3).
- No simulated, substituted or re-selected inputs. If `PRIMARY_97_R2_INPUT.csv` cannot be obtained with the digest above, R5 does not start.
