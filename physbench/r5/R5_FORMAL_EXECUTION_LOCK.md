# R5 formal execution lock (pre-outcome)

**Branch:** `r5-field-ventilation-flux`  
**Created before any formal R5 model result.**

## Frozen source hashes (SHA256)

| File | SHA256 |
|---|---|
| `physbench/r5/00_R5_CODEX_PROTOCOL.md` | `14068e7e47be99216f4d94f680fa4c2f98853a8c65b62954a11e106aea0b3c2a` |
| `physbench/r5/01_PROVENANCE_AND_INPUTS.md` | `ed9689861317d4444dc497d909615fbb48e2f939e8e687338d877b7731e0e229` |
| `physbench/r5/02_PROTOCOL_CLARIFICATIONS.md` | `08c65534c0f2782f6bb599201e70069dfdf43039d3652f220b3ccf54aca49238` |
| `physbench/r5/field_flux.py` | `b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd` |
| `physbench/r5/field_ventilation_function.json` | `4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b` |
| `physbench/r5/R5_event_list_97_v2.csv` | `5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5` |

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

Amended by R5_FORMAL_LOCK_AMENDMENT_1.md (CRLF->LF hash correction); original values preserved at commit 96c0f53 and in the amendment.
