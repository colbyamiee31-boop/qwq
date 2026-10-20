# R5 Formal Lock Amendment 1 (CRLF/LF Hash Correction)

**Authority:** Claude (scientific lead for R5), 2026-10-20.  
**Execution Agent:** Google Antigravity.  
**Branch:** `r5-field-ventilation-flux`.

## 0. Claude's Findings (Verified independently on 2026-10-20 through the GitHub API)

**Failed run.** Run `39372067352` (`physbench_r5_formal.yml`, head `fdbf47b4`, attempt 1):
- `guard` succeeded;
- `prepare` failed at "Validate frozen source hashes, input digest, and 97/40 event join";
- **`m1` was skipped, so GreenLight never ran**;
- `aggregate` failed only because its upstream jobs had failed.

**The committed files are the original handoff files.** The six files committed on the branch (git blobs at `fdbf47b4`) are LF and byte-identical to the files Claude handed over. Their SHA256 equals Claude's original values.

**The lock values are CRLF hashes.** The six values in `R5_FORMAL_EXECUTION_LOCK.md` are exactly the SHA256 of the same files with every LF converted to CRLF. They were computed from a Windows checkout. No content differs.

| File (physbench/r5/) | Lock value at 96c0f53 (CRLF) | Committed blob = original handoff (LF) |
|---|---|---|
| 00_R5_CODEX_PROTOCOL.md | bc0344dba73d05ce1eb4d5bd6d06d736a20b668a73ac77cf8ca30972de89f897 | 14068e7e47be99216f4d94f680fa4c2f98853a8c65b62954a11e106aea0b3c2a |
| 01_PROVENANCE_AND_INPUTS.md | 876f1fdc5d124024386095a3f1e048d48fc6b5a4af606a7ca013eb9157f7a4ec | ed9689861317d4444dc497d909615fbb48e2f939e8e687338d877b7731e0e229 |
| 02_PROTOCOL_CLARIFICATIONS.md | 94e7b450b4d0ddfa269601538c8c312ab9a812e8666962c62c2410689f108042 | 08c65534c0f2782f6bb599201e70069dfdf43039d3652f220b3ccf54aca49238 |
| field_flux.py | 4d57b85e7e4130bd5ea9340c362e5a2376effb66981d540dec27b48675dad481 | b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd |
| field_ventilation_function.json | fd90e01c82d7a8caebc544c0def08c05835879a866a74a3d3db4d6571e15778e | 4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b |
| R5_event_list_97_v2.csv | 58d475f78d3e0418339047dc7a7106162b6b3aeadf905d9620fe28e6eabc495b | 5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5 |

Ruling: the repair is approved as a cross-platform format correction with no scientific content change.

## 1. Verification Records (P2–P4)

### P2. Git blob SHA256 (LF) at commit `fdbf47b44751e2659a87208db774327a3bb8e47e`
- `physbench/r5/00_R5_CODEX_PROTOCOL.md`: `14068e7e47be99216f4d94f680fa4c2f98853a8c65b62954a11e106aea0b3c2a` (matches LF column: True)
- `physbench/r5/01_PROVENANCE_AND_INPUTS.md`: `ed9689861317d4444dc497d909615fbb48e2f939e8e687338d877b7731e0e229` (matches LF column: True)
- `physbench/r5/02_PROTOCOL_CLARIFICATIONS.md`: `08c65534c0f2782f6bb599201e70069dfdf43039d3652f220b3ccf54aca49238` (matches LF column: True)
- `physbench/r5/field_flux.py`: `b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd` (matches LF column: True)
- `physbench/r5/field_ventilation_function.json`: `4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b` (matches LF column: True)
- `physbench/r5/R5_event_list_97_v2.csv`: `5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5` (matches LF column: True)

P2 result: ALL PASS.

### P3. Git blob SHA256 with LF->CRLF conversion
- `physbench/r5/00_R5_CODEX_PROTOCOL.md`: `bc0344dba73d05ce1eb4d5bd6d06d736a20b668a73ac77cf8ca30972de89f897` (matches CRLF lock: True)
- `physbench/r5/01_PROVENANCE_AND_INPUTS.md`: `876f1fdc5d124024386095a3f1e048d48fc6b5a4af606a7ca013eb9157f7a4ec` (matches CRLF lock: True)
- `physbench/r5/02_PROTOCOL_CLARIFICATIONS.md`: `94e7b450b4d0ddfa269601538c8c312ab9a812e8666962c62c2410689f108042` (matches CRLF lock: True)
- `physbench/r5/field_flux.py`: `4d57b85e7e4130bd5ea9340c362e5a2376effb66981d540dec27b48675dad481` (matches CRLF lock: True)
- `physbench/r5/field_ventilation_function.json`: `fd90e01c82d7a8caebc544c0def08c05835879a866a74a3d3db4d6571e15778e` (matches CRLF lock: True)
- `physbench/r5/R5_event_list_97_v2.csv`: `58d475f78d3e0418339047dc7a7106162b6b3aeadf905d9620fe28e6eabc495b` (matches CRLF lock: True)

P3 result: ALL PASS. Proves that the lock values at 96c0f53 were CRLF hashes of the identical files.

### P4. Frozen R4 Input SHA256 check
- File: `PRIMARY_97_R2_INPUT.csv`
- Expected Constant: `0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b`
- Failed run `r5-formal-prepared-inputs` actual SHA256: `0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b` (Match: True)
- Frozen artifact `r4-prepared-inputs` actual SHA256: `0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b` (Match: True)

P4 result: PASS.

## 2. Failed Run Evidence Preservation

- GitHub Actions Run ID: `39372067352`
- Run URL: `https://github.com/colbyamiee31-boop/qwq/actions/runs/39372067352`
- Head SHA: `fdbf47b44751e2659a87208db774327a3bb8e47e`
- Artifacts:
  - `r5-formal-prepared-inputs`: ID `12241731605`, Digest `sha256:bb2fc581a45811ac95a9c9a1ee5a35081a855f575edbcfbbba2edda98ca81a52`
  - `r5-formal-final`: ID `12241303240`, Digest `sha256:2f3bba2acdff684e216814bac518f23d6834329c995c615504002aa68b55fe86`
- **M1 Execution Status:** Job `m1` was skipped (conclusion `skipped`, 0 executed steps). GreenLight was not run.

## 3. Scientific and Protocol Boundary Confirmation

- **No R5 FF outcome exists at this commit.**
- **Scientific content unchanged; only line-ending-dependent hash values and the trigger marker change.**
