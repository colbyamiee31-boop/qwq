# R5 Formal Amendment 2: Stage F Engineering Repairs and D1-v2 Closeout Record

**Authority:** Claude (scientific lead for R5), 2026-10-20.

## 0. Ruling on D1-v2 (Run 39410246452, Artifact 12258842745)

### Verification via GitHub API
- **Workflow:** `physbench_r5_diag_d1_v2.yml`
- **Run ID:** `39410246452`
- **Event:** `push`
- **Attempt:** `1`
- **Jobs:** `guard` and `diag` both completed successfully (`success`).
- **Artifact:** `r5-diag-d1-v2` (ID `12258842745`, digest `sha256:c689018df07b95f24fa34ebee46dc3f8e3909ebe6e800e5567411c5997ed1025`).

### Result Pattern (Storage-Precision Criterion, 12 Significant Digits)

| Variant | Integrator | Flux Used | Difference from Frozen R4 CF0 |
|---|---|---|---|
| V0 | frozen C8 | frozen `emp_flux_*_23_23` | 0 |
| V2 | R5 (`FCF`: quad + `hfac=1.0`) | frozen `emp_flux_*_23_23` | 0 |
| V1 | frozen C8 | recomputed `q_common_pct` | 1.285848e-6 |
| V3 | R5 (`FCF`: quad + `hfac=1.0`) | recomputed `q_common_pct` | 1.285848e-6 |

### Closeout Verification Items (i)–(iv)
- **(i) V0 and V2 vs Frozen R4 CF0:** Per event, V0 and V2 are equal to each other and to frozen R4 CF0 at storage precision (12 significant digits) across all 40 events:
  - $\max |\Delta T_{V0}| = 0.0$, $\max |\Delta AH_{V0}| = 0.0$
  - $\max |\Delta T_{V2}| = 0.0$, $\max |\Delta AH_{V2}| = 0.0$
  - $V0 \equiv V2$ across all 40 events.
- **(ii) V1 vs V3:** Per event, V1 and V3 are equal to each other at storage precision across all 40 events:
  - $\max |V1 - V3|_{\Delta T} = 0.0$, $\max |V1 - V3|_{\Delta AH} = 0.0$.
  - This conclusively proves that the R5 integrator structure (quadrature output + symbolic `hfac` parameter) plays zero role in the numerical discrepancy.
- **(iii) Full-Precision Flux Bit-Identity Table:** Across all 80 arms (40 events $\times$ 2 arms: pre and post):
  - 2 arms are bit-identical between recomputed full-precision formula and frozen 12-digit stored flux.
  - 78 arms differ in their lowest-order bits due to 12g decimal truncation when written to the frozen dataset.
- **(iv) Integrator Statistics Appendix:** CVODES statistics for `CTIFL-E5-P-0158` and `CTIFL-E5-P-0165` (pre and post arms) are recorded in Section 4 below.

---

## 1. Prior Runs Record and Artifact Digests

| Stage | Run ID | Conclusion | Artifact Name | Artifact ID | SHA256 Digest |
|---|---|---|---|---|---|
| Formal v1 | `39372067352` | job `m1` skipped (CRLF/LF hash lock) | `r5-formal-prepared-inputs` | `12242138243` | `sha256:49c0d4a974b7c6cbafeea014fa2bc4fc9f6eece961a03975ba952c1e6a151859` |
| Formal v2 | `39387663913` | job `m1` failed (embedded C8 gate) | `r5-formal-m1` | `12248087110` | `sha256:f126d4ca957e8fa178e2025732bbaf852a4e21a224f8d55d72f102561571dd63` |
| Diag D1 | `39404754210` | job `diag` failed (syntax/setup) | `r5-diag-d1` | `12256086961` | `sha256:7bbec3758b2dc14c810cba7ee06497330fe7b51b3fbc065fcba035e4e7e60086` |
| Diag D1-v2 | `39410246452` | job `diag` success (conclusive) | `r5-diag-d1-v2` | `12258842745` | `sha256:c689018df07b95f24fa34ebee46dc3f8e3909ebe6e800e5567411c5997ed1025` |

**Verification of Artifact 12248087110 (Formal v2):**
In run `39387663913`, the embedded C8 gate failed at line 434 before the loop over FF0/FFN/FFH. Artifact `12248087110` contains response rows **only** for condition `REF0` and **only** for the 40 subset events (80 rows total: 40 pre and 40 post arms).

No FF0/FFN/FFH outcome exists at this commit.

---

## 2. Engineering Repairs (R1–R6)

All repairs are strictly engineering repairs. None modifies any condition, parameter, event list, baseline, endpoint, bootstrap setting, or numerical tolerance threshold.

- **R1. REF0 Flux Implementation (Classification: restores frozen R4 implementation of protocol $q_{\text{ref}}$):**
  - In `physbench/r5/m1_r5_formal.py`, `prescribed_flux`: for `REF0`, reads and returns `float(r.emp_flux_pre_23_23)` / `float(r.emp_flux_post_23_23)` directly, matching the exact C8 and R4 CF0 path.
  - The GreenLight formula `q_common_pct` is retained as an independent verification gate `r1_ref0_flux_formula_12g_equality`: checks that `float(format(q_common_pct(...), '.12g')) == emp_flux` holds exactly across all 97 events $\times$ 2 arms (194 evaluations). Pre-flight check confirms 0 mismatches, with $\max |\text{rel\_diff}| = 4.925747 \times 10^{-14}$.
- **R2. Integrator Configuration (Classification: unchanged unified integrator):**
  - All four conditions (REF0, FF0, FFN, FFH) continue to use the single unified R5 CasADi/CVODES integrator `FCF` with quadrature output and symbolic `hfac`.
  - `hfac = 1.0` for REF0, FF0, FFN; `hfac = 0.3412` for FFH.
  - As established by D1-v2, this integrator reproduces frozen R4 CF0 bit-for-bit at storage precision when supplied with the frozen flux.
- **R3. Field Flux Logging (Classification: hex audit for byte-level reproducibility):**
  - Conditions FF0, FFN, FFH continue to use `field_flux.q_field` in double precision (C10), with FFN leakage routing and FFH sensible factor per C9.
  - Every prescribed $q_{\text{top}}$ and $q_{\text{main}}$ is recorded in hex format via `float.hex()` in `R5_runtime_audit.csv`.
- **R4. Aggregate Column Handling (Classification: explicit renaming against key collisions):**
  - In `physbench/r5/aggregate_r5.py`, input metadata columns are explicitly renamed with an `_input` suffix before merging in `make_hv_canonical()` and canonical response preparation to prevent duplicate column suffixes or KeyErrors.
  - In `compute_paired()`, `pre_state` and `post_state` columns are explicitly preserved on condition frames.
  - Top-level script execution is wrapped in `if __name__ == "__main__":` to support safe importing by the test suite.
- **R5. Offline Aggregate Test (Classification: CI pre-execution software verification):**
  - Added `physbench/r5/test_aggregate_offline.py`.
  - Runs in CI during the `prepare` job before `m1` can begin. Exercises HV canonicalisation (194 rows), REF0 canonicalisation (80 rows), embedded C8 join against frozen R4 CF0 (40/40), C4 metric evaluation, and exercises the full aggregation pipeline on an in-memory fixture (40 summary rows, 48 paired difference rows).
  - Strictly software verification: writes zero files to `evidence/` and displays zero endpoint numbers.
- **R6. Static Scope and Workflow Guard (Classification: security and execution guard):**
  - Workflow `physbench_r5_formal.yml` updated to trigger exclusively on `physbench/r5/R5_FORMAL_TRIGGER_ONCE_20261020_v3.txt`.
  - Guard asserts attempt 1, v3 trigger marker, and verifies the exact 3-run history: run 39372067352 (m1 skipped), run 39387663913 (m1 failed with artifact 12248087110), and the current run.
  - Offline test step added to `prepare` job with `scipy==1.14.1` dependency.

---

## 3. Pre-Flight Verification Results

1. **Offline Aggregate Test:**
   - Ran `test_aggregate_offline.py` locally on frozen artifacts.
   - Output: `test_aggregate_offline PASS` (all 5 test suites passed).
2. **R1 Gate Dry Check:**
   - Verified across all 97 events $\times$ 2 arms: 0 mismatches at 12 significant digits.
   - Max relative difference at full double precision: $4.925747 \times 10^{-14}$.
3. **ODE Signature & Column Inspection:**
   - Verified `inspect.signature(ODE) == (x, u, d, p)`.
   - Verified all 23 columns read by `m1_r5_formal.py` and all 7 metadata columns read by `aggregate_r5.py` exist in frozen `PRIMARY_97_R2_INPUT.csv`.
4. **Git Diff Scope:**
   - The code diff from commit `2c87e08` touches strictly the 4 files specified in R6:
     - `physbench/r5/m1_r5_formal.py`
     - `physbench/r5/aggregate_r5.py`
     - `physbench/r5/test_aggregate_offline.py`
     - `.github/workflows/physbench_r5_formal.yml`
5. **Line Endings:**
   - All changed files normalized to pure LF (`i/lf`, `w/lf`).

---

## 4. Appendix: CVODES Statistics for P-0158 and P-0165 (from D1-v2 Run 39410246452)

### Event `CTIFL-E5-P-0158`

| Arm | Variant | Steps (`nsteps`) | RHS Evals (`nfevals`) | Lin Setups (`nlinsetups`) | Error Test Fails (`netfails`) | Nonlin Iters (`nniters`) |
|---|---|---|---|---|---|---|
| pre | V1 | 130 | 194 | 38 | 11 | 190 |
| pre | V2 | 130 | 194 | 38 | 11 | 190 |
| pre | V3 | 130 | 194 | 38 | 11 | 190 |
| post | V1 | 122 | 171 | 33 | 8 | 167 |
| post | V2 | 122 | 171 | 33 | 8 | 167 |
| post | V3 | 122 | 171 | 33 | 8 | 167 |

### Event `CTIFL-E5-P-0165`

| Arm | Variant | Steps (`nsteps`) | RHS Evals (`nfevals`) | Lin Setups (`nlinsetups`) | Error Test Fails (`netfails`) | Nonlin Iters (`nniters`) |
|---|---|---|---|---|---|---|
| pre | V1 | 108 | 150 | 32 | 6 | 146 |
| pre | V2 | 108 | 150 | 32 | 6 | 146 |
| pre | V3 | 108 | 150 | 32 | 6 | 146 |
| post | V1 | 110 | 154 | 28 | 6 | 150 |
| post | V2 | 110 | 154 | 28 | 6 | 150 |
| post | V3 | 110 | 154 | 28 | 6 | 150 |

*(Note: V0 executed frozen `m1_c8_primary.py` as a subprocess without CVODES internal statistics logging. V1, V2, and V3 exhibit identical CVODES iteration and step counts for both events and both arms).*
