# R5 formal execution lock (pre-outcome)

**Branch:** `r5-field-ventilation-flux`  
**Created before any formal R5 model result.**

## Frozen source hashes (SHA256)

| File | SHA256 |
|---|---|
| `physbench/r5/00_R5_CODEX_PROTOCOL.md` | `bc0344dba73d05ce1eb4d5bd6d06d736a20b668a73ac77cf8ca30972de89f897` |
| `physbench/r5/01_PROVENANCE_AND_INPUTS.md` | `876f1fdc5d124024386095a3f1e048d48fc6b5a4af606a7ca013eb9157f7a4ec` |
| `physbench/r5/02_PROTOCOL_CLARIFICATIONS.md` | `94e7b450b4d0ddfa269601538c8c312ab9a812e8666962c62c2410689f108042` |
| `physbench/r5/field_flux.py` | `4d57b85e7e4130bd5ea9340c362e5a2376effb66981d540dec27b48675dad481` |
| `physbench/r5/field_ventilation_function.json` | `fd90e01c82d7a8caebc544c0def08c05835879a866a74a3d3db4d6571e15778e` |
| `physbench/r5/R5_event_list_97_v2.csv` | `58d475f78d3e0418339047dc7a7106162b6b3aeadf905d9620fe28e6eabc495b` |

These SHA256 values were computed from the isolated R5 branch source files and matched the corresponding frozen branch file blobs before this lock was created.

## C8 gate references

- Model evidence: GitHub Actions run `39364637096`.
- Re-audit evidence: GitHub Actions run `39368915179`, artifact `r5-c8-reaudit` (ID `12240975484`).
- The audit artifact digest is `sha256:ace1173da9e8c80be83d035e14ec16a2cb994b2fb2d8e58884a4a89c9dd7699a`.

## §0 duplicated console-log confirmation

The authoritative log is `evidence/R5/C8/c8_model_console.log` in artifact `r5-c8-model-output` (run `39364637096`, artifact ID `12238792702`, artifact digest `sha256:d829156d37bcec577fdc40efe7467eefa4b435f9c14013e8ef9fac44cadd62ff`). Its SHA256 is `c1ee52f1b23162da9608fcd07f4012e4789520bfeb015d1b14904498205ff834`, 2,095 bytes.

I checked every case-insensitive basename match in `C8_reaudit_file_hashes.csv` from artifact `r5-c8-reaudit` (ID `12240975484`, file SHA256 `94905912bd61d1d5c97436ae79e78f56d1d6a02d9320f2641df37c9077864327`). All five recorded copies (one from `r5-c8-model-output`, two from `r5-c8-reproduction`, and both case variants in the archived original-run copy) have the same 2,095-byte length and the same SHA256 as the authoritative log. §0 passes.

## Execution boundary

No frozen input, event, flux parameter, evaluation rule, dependency constraint, historical branch, or original project is changed by this lock. The formal run is GreenLight M1 only. No CSGtom, refitting, or protocol adjustment is authorized.

**No R5 FF outcome exists at this commit**
