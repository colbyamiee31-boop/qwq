# Stage D1 Rerun Note (D1-v2): Corrected Diagnostic Implementation

**Authority:** Claude (scientific lead for R5) & User, 2026-10-20.  
**Execution Agent:** Google Antigravity.  
**Branch:** `r5-field-ventilation-flux`.  
**Scope:** Diagnostic compute run (Stage D1-v2).

---

## 0. Review of Run 39404754210 and `m1_r5_diag_d1.py` (Head `68b6e98`)

**Prior Failed Run Evidence:**
- **Run ID:** `39404754210`
- **Head SHA:** `68b6e985`
- **Artifact:** `r5-diag-d1` (ID `12256474354`, digest `sha256:20179777373c5044ec8505eb69e84f6f53ee0d5ee15729c37f551da4333895eb`)
- **Status:** Job `diag` failed at initialization with `TypeError: ODE() takes 4 positional arguments but 5 were given`.
- **Execution fact:** **No diagnostic variant (V0–V3) was executed. GreenLight simulation never began.**

**Review Findings:**
The frozen GreenLight-Gym2 at `2d3febb` defines `ODE(x, u, d, p)` (`gl_gym/models/GreenLight/ode.py`, line 4), and C8 calls `ODE(x,u,d,p)`. The call `ODE(x, u, d, p, a)` was a plain coding error.

A static review against the frozen C8 script (`m1_c8_reproduction.py`) further showed that the original D1 script did not implement the approved design:
1. V0 was re-implemented instead of executing the unmodified `m1_c8_reproduction.py`.
2. `initial_state` deviated in state variables and column names.
3. `native_top` index was misassigned (`a[138]+a[146]` instead of `a[136]`).
4. External exchange equations lacked `ca.fabs` and misused numpy functions on CasADi symbols.
5. Storage precision comparison (`format(v, '.12g')`) was not applied for reference checking.
6. Outputs lacked `try/finally` protection.
7. Step size key was `"h"` instead of `hlast`/`hcur`.
8. Workflow lacked automated comparison of `pip freeze`.

**Correction of D1 comparison rule:**
The frozen R4 responses are stored to 12 significant digits (`%.12g`), so exact equality must be judged at storage precision `s(v) = float(format(v, '.12g'))`.

---

## 1. D1-v2 Implementation Strategy: Copy, Never Re-implement

- **V0:** Unmodified `physbench/r5/m1_c8_reproduction.py` is invoked directly as a subprocess; its output `c8_m1_responses.csv` is compared against frozen R4 CF0 using storage precision $s(v)$.
- **C8 Block:** Verbatim copy of `build_weather`, `full_action`, `initial_state`, the symbolic ODE block from `x=ca.SX.sym('x',NX)` to `FCF=ca.integrator(...)`, and `run_arm` logic from `m1_c8_reproduction.py` (with allowed rename `FCF` $\rightarrow$ `FCF_C8`).
- **R5 Block:** Verbatim copy of the symbolic ODE block from `x = ca.SX.sym("x", NX)` to `FCF = ca.integrator(...)` and parameter handling from `m1_r5_formal.py` (with allowed rename `FCF` $\rightarrow$ `FCF_R5` and `STEP_SECONDS = 900.0`).
- **Static Copy Gate:** Enforced prior to model execution in CI using Python `difflib` and block SHA256 verification.
- **Fail-Closed Stop Conditions:** If V0 fails storage-precision equality with frozen R4 CF0, V1–V3 are not run. If V3 fails to reproduce the v2 `CTIFL-E5-P-0158` difference at storage precision, execution halts fail-closed.

---

## 2. Non-Outcome Statement

**D1-v2 is a diagnostic; no FF outcome exists.**
- No field ventilation conditions (`FF0`, `FFN`, `FFH`) are executed.
- No `q_field` or field-identified ventilation function is evaluated.
- Results do not enter formal R5 paper statistics.
