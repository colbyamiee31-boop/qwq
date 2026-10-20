## 0. Scientific ruling (for the record)

**Model layer.** Run 39364637096 is accepted as valid at the model layer:
- jobs `prepare` and `c8` succeeded;
- `m1_c8_reproduction.py` reads only `PRIMARY_97_R2_INPUT.csv` and integrates the CF0 arms itself; it does not read R4 responses;
- 40/40 events were compared, with a maximum |ΔT| and |ΔAH| of 0.0 at the %.12g CSV precision.

**Audit layer.** The audit layer failed for two independent reasons. Both are defects of the C8 code, not of the frozen data.

1. **Hand-typed expected hashes.** `EXPECTED` in `aggregate_c8.py` holds file-level SHA256 values typed into the C8 code. They were never part of the frozen provenance, which records artifact-zip digests and run IDs. One value (`r4_summary`) has 63 hex digits, so it cannot match any file. The fix is to derive file hashes from the digest-verified artifacts, not to edit constants by hand.
2. **Path error.** `upload-artifact` stores `evidence/R5/C8/` at the artifact root. The two precheck logs are therefore at `precheck-output/<file>`, not at `precheck-output/evidence/R5/C8/<file>`.

## Amendment boundary

Original C8 run: `39364637096`. This amendment records the audit-layer repair only. The R5 protocol, clarifications C1–C10, event set, frozen inputs, field flux, `1e-6` tolerance, and all runtime thresholds from `aggregate_c8.py` remain unchanged (init ≤ `1e-8`, flux identity ≤ `1e-12`, paired errors == 0, and exact pip-freeze match). No GreenLight model is rerun; no FF0/FFN/FFH or formal R5 experiment is started. The original C8 failure report and artifacts remain untouched.