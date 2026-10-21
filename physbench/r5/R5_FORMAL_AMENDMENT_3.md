# R5 Formal Amendment 3: Audit-Only Gate Reconstruction and Re-Aggregation Record

**Authority:** Claude (scientific lead for R5), 2026-10-20.

## 0. Independent Verification of Saved v3 Outputs (Run 39429049087)

### Verification via GitHub API
- **Workflow:** `physbench_r5_formal.yml`
- **Run ID:** `39429049087`
- **Event:** `push`
- **Attempt:** `1`
- **Jobs:** `guard` and `prepare` succeeded; `m1` failed in `write_outputs()` (`TypeError: Object of type bool_ is not JSON serializable`); `aggregate` failed due to missing `R5_gate_summary.json`.

### Artifacts of Run 39429049087

| Artifact Name | Artifact ID | Size (bytes) | SHA256 Digest |
|---|---|---|---|
| `r5-formal-prepared-inputs` | `12267366219` | 30,509 | `sha256:ca7ba3e752fcd003e744aabb464769d31f2c5c153b76562698dbaa20ccfaded2` |
| `r5-formal-m1` | `12266598692` | 109,127 | `sha256:8a1362face7a628b56cdeaff0cc3c381b171c2c0e6ec75accce7ea67e3dc604d` |
| `r5-formal-final` | `12266588918` | 114,775 | `sha256:b58d671793696399932460e5735ef410aaacc6f3aedabf92ac5a35ec09272f76` |

### Control Flow and Model Execution Proof
- In `m1_r5_formal.py`, every gate before and during simulation uses `add_check(...)`, which raises immediately on failure.
- The `r5_formal_m1_console.log` records exactly 388 "done" lines (40 pre-FF REF0 + 97 × 4 − 40).
- Exactly one traceback exists in the console log, located strictly in `finally: write_outputs()`, caused by standard library `json.dumps()` attempting to serialize NumPy `np.bool_`.
- All GreenLight physical simulations completed in full; no physical model gate raised.

### Model Outputs SHA256 Verification

| File | SHA256 Digest |
|---|---|
| `R5_event_responses.csv` | `0a0a427641ef96cf511bd5401ad40edd19106591fbc07fbac20636e3f2a81c49` |
| `R5_runtime_audit.csv` | `cebed7b174705590815924372ecdbbb946f7bd5913b11fd94025cff4f4d4f40a` |
| `R5_embedded_C8_event_comparison.csv` | `4bfa671caa163b8eb0f2bf9fc9bb962f383c3c7c21700d1493861687de1b04fa` |
| `R5_alignment_states.csv` | `a11efaeef73fdf8b4a47e1d267b123b703fc1615d5a85aa257a2d6859b5fbe07` |

### Numerical Gate Results Recomputed from Saved Artifacts
1. **Response rows:** 776 rows (97 events × 4 conditions × 2 horizons), all states finite. Observed effects equal frozen input exactly (maximum difference 0.0).
2. **Runtime audit:** 1552 rows, all finite; maximum initial-state error 1.14e-13; maximum flux-identity error 2.13e-14; paired initial-state and forcing errors are 0.0; all prescribed fluxes non-negative.
3. **R1 flux gate:** For all 97 events × 2 arms, `float(format(q_common_pct(...), '.12g')) == emp_flux` holds with maximum relative difference 4.93e-14. Audit REF0 `q_top` equals `emp_flux` and `q_main == 0.0` for all 388 rows.
4. **Embedded C8 gate:** 40 events, maximum |ΔT| = 3.49e-12 °C, maximum |ΔAH| = 4.39e-12 g m⁻³ (both far below 1e-6 tolerance; 40/40 <= 1e-12; 40/40 exactly equal at 12g).
5. **Dependency gate:** `evidence_R5_M1_PIP_FREEZE.txt` is byte-identical to `evidence_R4_M1_PIP_FREEZE.txt`.

### Formal Declarations
- **model outputs final; no model rerun**
- **no endpoint has been inspected for the paper**

---

## 1. Engineering Scope of Re-Aggregation (E1–E3)

- **E1. JSON serialization fix:** Recursive NumPy scalar (`np.bool_`, `np.integer`, `np.floating`, `np.generic`) to native Python conversion before `json.dumps()` in `m1_r5_formal.py` and `aggregate_r5.py`. Covered by unit test `test_json_numpy.py`.
- **E2. Independent gate reconstruction (`reconstruct_m1_gates.py`):** Reconstructs all 27 formal gates from the verified model outputs and frozen inputs, generating `R5_gate_summary_reconstructed.json`.
- **E3. Aggregation input path (`aggregate_r5.py`):** Points `M1_SUMMARY` to `R5_gate_summary_reconstructed.json`, asserting `"reconstructed_from_saved_outputs": true` and `"source_run": "39429049087"`. All scientific metrics, cohorts, signing, and bootstrap logic remain completely unchanged.
- **Workflow:** Standalone `.github/workflows/physbench_r5_formal_reagg.yml` containing only `guard` and `reaggregate` jobs.
