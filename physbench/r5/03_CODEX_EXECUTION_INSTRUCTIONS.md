# Codex execution instructions for R5 (absolute paths)

Read in this order: `00_R5_CODEX_PROTOCOL.md` (frozen), `02_PROTOCOL_CLARIFICATIONS.md`, `01_PROVENANCE_AND_INPUTS.md`, then this file.

Hard rules:
- `E:\GreenLight_MPC` is never opened for writing, never used as a working directory and never referenced by any environment.
- No R5 outcome is computed before gate C8 passes.
- Do not interpret results. Return the outputs to Claude.

## Paths

| Item | Absolute path |
|---|---|
| Handoff (frozen) | `E:\GreenLight_R5\R5_Codex_handoff\R5_handoff` (the nesting is fine, do not move it) |
| This supplement | `E:\GreenLight_R5\R5_inputs_supplement` |
| Downloaded frozen artifacts | `E:\GreenLight_R5\frozen_artifacts\<artifact-name>\` |
| R5 working code | `E:\GreenLight_R5\r5\` |
| R5 outputs | `E:\GreenLight_R5\r5\evidence\R5\` |
| Fresh environment (Route B) | `E:\GreenLight_R5\.venv_r5` |
| Frozen model checkout (Route B) | `E:\GreenLight_R5\GreenLight-Gym2_2d3febb` |

## Step 0: fetch and verify the frozen artifacts (both routes)

Run `E:\GreenLight_R5\R5_inputs_supplement\04_fetch_frozen_artifacts.ps1` in PowerShell, with GitHub CLI logged in to an account that can read `colbyamiee31-boop/qwq`.
- The script downloads each artifact zip, checks its SHA256 against 01 §2, and unzips it only if the hash matches.
- Any mismatch or download failure → stop and report.

## Route A (recommended): run R5 as a GitHub Actions workflow in `colbyamiee31-boop/qwq`

This reproduces the R3/R4 execution environment exactly (ubuntu-24.04, Python 3.11, pinned requirements, frozen M1 commit) and needs no local environment. Step 0 is then only a local cross-check.

1. Create branch `r5-field-ventilation-flux` from `r4-aerodynamic-transport-isolation`.
2. Add `physbench/r5/`:
    - `common.py`: copy of r4 `common.py`, plus `from field_flux import q_field`.
    - `field_flux.py` and `field_ventilation_function.json`: copied byte-for-byte from the handoff `inputs/`.
    - `R5_event_list_97_v2.csv`.
    - `m1_field_flux.py`: adapted from r4 `m1_common_flux.py`. The only changes are the conditions REF0/FF0/FFN/FFH (C9, C10), the 1800-s horizon (C7), all 97 events (C5 for the subset), and gate C8.
    - `aggregate.py`: endpoints of protocol §5 with C4.
    - `test_protocol.py`: the `field_flux.py` unit tests and the input SHA checks.
3. Add `.github/workflows/physbench_r5_field_flux.yml`, modelled on the R4 workflow:
    - the `prepare` job downloads `r4-prepared-inputs` with `run-id: 37164458429`;
    - the `aggregate` job downloads `r3-m1` (`run-id: 37163085468`) and `r4-m1` (`run-id: 37164458429`) as frozen references;
    - M1 is installed by `git checkout --detach 2d3febb1ea002b24b452e32293e990beb78d3ce1`, and `pip freeze` is saved.
4. Commit the protocol files first (protocol-only commit), then the code, and run.
5. Return artifacts `r5-m1` and `r5-final` (IDs and digests) to Claude.

## Route B (only if Route A is not possible): local, fully separate from GreenLight_MPC

- **B1.** Do not reuse the copied `.venv`. Its `pyvenv.cfg`, `activate.bat` and the editable `gl_gym` install point to `E:\GreenLight_MPC`. Create a new one:
  `py -3.11 -m venv E:\GreenLight_R5\.venv_r5`
- **B2.** `git clone --no-tags https://github.com/BartvLaatum/GreenLight-Gym2.git E:\GreenLight_R5\GreenLight-Gym2_2d3febb`, then `git -C E:\GreenLight_R5\GreenLight-Gym2_2d3febb checkout --detach 2d3febb1ea002b24b452e32293e990beb78d3ce1`.
- **B3.** Install into the new venv with a non-editable install of the pinned checkout:
  `E:\GreenLight_R5\.venv_r5\Scripts\python.exe -m pip install casadi==3.8.1 cloudpickle==3.1.2 gymnasium==0.29.0 numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 E:\GreenLight_R5\GreenLight-Gym2_2d3febb`
  - Compare the resulting `pip freeze` with `frozen_artifacts\r4-m1\evidence_R4_M1_PIP_FREEZE.txt` and report every difference.
  - Verify that `gl_gym.__file__` resolves inside `.venv_r5`, and not inside `E:\GreenLight_MPC`.
  - The existing `GreenLight-Gym2-master` copy is not used.
- **B4.** Build `E:\GreenLight_R5\r5\` with the same files as Route A, step 2. Read inputs from `E:\GreenLight_R5\frozen_artifacts\r4-prepared-inputs\PRIMARY_97_R2_INPUT.csv`.
- **B5.** Run gate C8 first: reproduce R4 CF0 at 15 min on the 40-event subset and compare with `frozen_artifacts\r4-m1\evidence\R4\M1\ctifl_common_flux_responses.csv`. Windows and Linux CVODES builds may differ slightly. If the 1e-6 tolerance fails, stop and send Claude the maximum differences. Do not run FF conditions.
- **B6.** If C8 passes, run all conditions and write the outputs listed in protocol §7 to `E:\GreenLight_R5\r5\evidence\R5\`.

## Outputs to return

Return `R5_event_responses.csv`, `R5_summary.csv`, `R5_paired_differences.csv` and `R5_run_log.md`. The run log must contain:
- the route used;
- commit IDs, or the local checkout hash and pip freeze;
- the SHA256 of every input;
- the result of the C8 reproduction gate with the maximum differences;
- the FFH code location;
- the result of every runtime gate.
