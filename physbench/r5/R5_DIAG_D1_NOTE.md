# R5 Stage D1 Diagnostic Compute Run Note

**Authority:** Claude (scientific lead for R5) & User, 2026-10-20.  
**Execution Agent:** Google Antigravity.  
**Branch:** `r5-field-ventilation-flux`.  
**Scope:** Diagnostic compute run (Stage D1).

---

## 1. Non-Outcome Statement

**THIS IS NOT A FORMAL R5 EXPERIMENT.**
- No field-flux conditions (`FF0`, `FFN`, `FFH`) are run.
- No `q_field` or field-identified ventilation function is evaluated.
- No formal R5 outcome exists at this commit.
- The results of this run do not enter the formal R5 paper statistics and are intended exclusively as diagnostic evidence to isolate the numerical sensitivity observed in run `39387663913`.

---

## 2. D0 Summary and Corrections

1. **Pre-arm localization:** D0 confirmed that the numerical difference in `CTIFL-E5-P-0158` occurred entirely during the event-pre integration arm (`pre_T` diff $+1.2858\times 10^{-6}$ °C, `pre_AH` diff $-7.487\times 10^{-7}$ g/m³), whereas the event-post integration arm matched frozen R4 CF0 exactly ($0.0$ difference).
2. **Flux precision correction:** D0's observation of flux matching was based on CSV representations formatted to 12 significant digits. To conclusively test whether formula recalculation in float64 differs at machine precision from frozen inputs, Variant V1 is strictly retained.
3. **P-0165 error clarification:** `CTIFL-E5-P-0165` had a temperature difference of $1.00218\times 10^{-8}$ °C and humidity difference of $9.9733\times 10^{-9}$ g/m³ in run `39387663913`, which must be tracked alongside `CTIFL-E5-P-0158`.

---

## 3. Four Diagnostic Variants (40 subset events, 15 min, REF0 only)

- **V0 (C8 baseline):** Unmodified C8 integrator (`p = [d, p, qtop, qmain]`, no quad, no `hfac`) with frozen `emp_flux_*` values. Verifies whether current CI runners reproduce frozen R4 CF0 exactly ($0.0$ difference).
- **V1 (Flux test):** C8 integrator with formula-recomputed ventilation flux via `q_common_pct(...)` in full double precision. Isolates effect of flux recalculation.
- **V2 (Integrator test):** R5 integrator (`p = [d, p, qtop, qmain, hfac]`, `hfac=1.0`, `quad = qtop + qmain`) with frozen `emp_flux_*` values. Isolates effect of CasADi CVODES computational graph structure (quadrature and symbolic parameter).
- **V3 (R5 v2 reproduction):** R5 integrator with formula-recomputed ventilation flux via `q_common_pct(...)`. Tests determinism and verifies reproduction of the $1.2858\times 10^{-6}$ °C anomaly on `CTIFL-E5-P-0158`.

---

## 4. Mandatory Diagnostic Logging

1. **Full-Precision Flux Records:**
   - Evaluated `q_top` and `q_main` in `float.hex()` and `repr()`.
   - Frozen `emp_flux_*` in `float.hex()` and `repr()`.
   - Explicit boolean bit-identity check (`q_top == emp_flux`).
2. **Integrator Solver Statistics:**
   - CVODES internal steps (`nsteps`).
   - Right-hand side function evaluations (`nfevals`).
   - Error test failures (`netfails`).
   - Last step size if reported by solver (`h` / `last_step_size`).
   - Detailed focus on `CTIFL-E5-P-0158` and `CTIFL-E5-P-0165`.

---

## 5. Stop Conditions

- Failure of V0 to reproduce frozen R4 CF0 exactly ($0.0$ difference at saved precision).
- Failure of V3 to reproduce the $1.2858\times 10^{-6}$ °C difference on `CTIFL-E5-P-0158`.
- Failure of CI runner environment, dependencies, or git markers.
- If any failure occurs, no automatic rerun or v2 marker creation is permitted without explicit user and scientific lead instruction.
