#!/usr/bin/env python3
"""
Reconstruct M1 Formal Runtime Gates from Saved Artifacts.
Authority: Claude (scientific lead for R5), 2026-10-20.

Reads saved model outputs, prepared inputs, and frozen baselines from run 39429049087,
re-verifies all mathematical, numerical, and structural gates, and outputs
R5_gate_summary_reconstructed.json using safe NumPy-compatible JSON serialization.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "physbench/r5"))
from common import q_common_pct


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [safe_json(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.generic):
        return safe_json(value.item())
    return value


def locate_file(candidates, description):
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(f"Could not locate {description}. Tried: {candidates}")


def main():
    # Resolve directories
    prep_dir = Path(os.environ.get("PREPARED_DIR", ROOT / "prepared" if (ROOT / "prepared").is_dir() else ROOT / "prepared-inputs")).resolve()
    model_dir = Path(os.environ.get("MODEL_DIR", ROOT / "model-output")).resolve()
    r4_dir = Path(os.environ.get("BASE_R4_DIR", ROOT / "baseline_r4")).resolve()
    out_dir = Path(os.environ.get("OUT_DIR", ROOT / "evidence/R5/FORMAL")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Candidate paths for files
    prep_gate_path = locate_file(
        [prep_dir / "R5_PREPARE_GATE_SUMMARY.json",
         ROOT / "prepared/R5_PREPARE_GATE_SUMMARY.json",
         ROOT / "prepared-inputs/R5_PREPARE_GATE_SUMMARY.json",
         ROOT / "physbench/r5/generated/R5_PREPARE_GATE_SUMMARY.json",
         ROOT.parent / "formal_run_v3/r5-formal-prepared-inputs/R5_PREPARE_GATE_SUMMARY.json"],
        "R5_PREPARE_GATE_SUMMARY.json"
    )
    primary_input_path = locate_file(
        [prep_dir / "PRIMARY_97_R2_INPUT.csv",
         ROOT / "prepared/PRIMARY_97_R2_INPUT.csv",
         ROOT / "prepared-inputs/PRIMARY_97_R2_INPUT.csv",
         ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv",
         ROOT.parent / "formal_run_v3/r5-formal-prepared-inputs/PRIMARY_97_R2_INPUT.csv",
         ROOT.parent / "frozen_artifacts/r2-prepared-inputs/PRIMARY_97_R2_INPUT.csv"],
        "PRIMARY_97_R2_INPUT.csv"
    )
    event_list_path = locate_file(
        [ROOT / "physbench/r5/R5_event_list_97_v2.csv"],
        "R5_event_list_97_v2.csv"
    )
    resp_path = locate_file(
        [model_dir / "evidence/R5/FORMAL/R5_event_responses.csv",
         model_dir / "R5_event_responses.csv",
         out_dir / "R5_event_responses.csv",
         ROOT.parent / "formal_run_v3/r5-formal-m1/evidence/R5/FORMAL/R5_event_responses.csv"],
        "R5_event_responses.csv"
    )
    audit_path = locate_file(
        [model_dir / "evidence/R5/FORMAL/R5_runtime_audit.csv",
         model_dir / "R5_runtime_audit.csv",
         out_dir / "R5_runtime_audit.csv",
         ROOT.parent / "formal_run_v3/r5-formal-m1/evidence/R5/FORMAL/R5_runtime_audit.csv"],
        "R5_runtime_audit.csv"
    )
    log_path = locate_file(
        [model_dir / "evidence/R5/FORMAL/r5_formal_m1_console.log",
         model_dir / "r5_formal_m1_console.log",
         out_dir / "r5_formal_m1_console.log",
         ROOT.parent / "formal_run_v3/r5-formal-m1/evidence/R5/FORMAL/r5_formal_m1_console.log"],
        "r5_formal_m1_console.log"
    )
    r5_pip_path = locate_file(
        [model_dir / "evidence_R5_M1_PIP_FREEZE.txt",
         ROOT.parent / "formal_run_v3/r5-formal-m1/evidence_R5_M1_PIP_FREEZE.txt"],
        "evidence_R5_M1_PIP_FREEZE.txt"
    )
    r4_resp_path = locate_file(
        [r4_dir / "evidence/R4/M1/ctifl_common_flux_responses.csv",
         ROOT.parent / "frozen_artifacts/r4-m1/evidence/R4/M1/ctifl_common_flux_responses.csv"],
        "ctifl_common_flux_responses.csv (R4)"
    )
    r4_audit_path = locate_file(
        [r4_dir / "evidence/R4/M1/runtime_audit.csv",
         ROOT.parent / "frozen_artifacts/r4-m1/evidence/R4/M1/runtime_audit.csv"],
        "runtime_audit.csv (R4)"
    )
    r4_pip_path = locate_file(
        [r4_dir / "evidence_R4_M1_PIP_FREEZE.txt",
         ROOT.parent / "frozen_artifacts/r4-m1/evidence_R4_M1_PIP_FREEZE.txt"],
        "evidence_R4_M1_PIP_FREEZE.txt"
    )

    checks = []
    summary = {
        "status": "PASS",
        "reconstructed_from_saved_outputs": True,
        "source_run": "39429049087",
        "greenlight_commit": "2d3febb1ea002b24b452e32293e990beb78d3ce1",
        "conditions": ["REF0", "FF0", "FFN", "FFH"],
        "horizons_min": [15, 30],
        "events_expected": 97,
        "subset_expected": 40,
        "tolerance_c8": 1e-06,
    }

    # 1. Prepare stage gates
    prep_gate = json.loads(prep_gate_path.read_text(encoding="utf-8"))
    prep_hash = sha256(prep_gate_path)
    if prep_gate.get("status") != "PASS":
        raise AssertionError(f"Prepare gate status is {prep_gate.get('status')}")
    if isinstance(prep_gate.get("checks"), dict):
        for name, passed in prep_gate["checks"].items():
            checks.append({"check": name, "passed": bool(passed), "detail": "prepare_stage"})
    elif isinstance(prep_gate.get("checks"), list):
        for c in prep_gate["checks"]:
            if isinstance(c, dict):
                checks.append(c)
            else:
                checks.append({"check": str(c), "passed": True, "detail": "prepare_stage"})
    checks.append({
        "check": "prepare_gate_summary_verified",
        "passed": True,
        "detail": json.dumps({"status": "PASS", "sha256": prep_hash, "path": str(prep_gate_path)})
    })

    # 2. Event list and anchor primary subset join
    inputs = pd.read_csv(primary_input_path)
    event_list = pd.read_csv(event_list_path)
    inputs["row_index"] = inputs.row_index.astype(int)
    event_list["row_index"] = event_list.row_index.astype(int)
    merged_events = inputs.merge(
        event_list[["row_index", "event_id", "in_R4_wind_3_6_subset"]],
        on="row_index", how="outer", suffixes=("", "_event_list"),
        indicator=True, validate="one_to_one"
    )
    assert len(merged_events) == 97 and (merged_events["_merge"] == "both").all()
    assert (merged_events.event_id.astype(str) == merged_events.event_id_event_list.astype(str)).all()
    subset = inputs[inputs.anchor_primary_3_6 == True].copy()
    assert len(subset) == 40 and subset.event_id.astype(str).nunique() == 40
    checks.append({
        "check": "event_join_and_anchor_primary_3_6",
        "passed": True,
        "detail": "97/97 row_index and event_id; subset flags agree 40/40."
    })
    summary["primary_event_count"] = len(inputs)
    summary["anchor_primary_3_6_count"] = len(subset)

    # 3. R1 formula equality gate (97 x 2 arms at 12g) and audit REF0 flux equality
    max_rel_diff = 0.0
    r1_mismatches = []
    for _, r in inputs.iterrows():
        for arm in ["pre", "post"]:
            lw = float(r.leeward_pre_pct if arm == "pre" else r.leeward_post_pct)
            ww = float(r.windward_pre_pct if arm == "pre" else r.windward_post_pct)
            wspd = float(r.event_Windsp)
            emp = float(r.emp_flux_pre_23_23 if arm == "pre" else r.emp_flux_post_23_23)
            q_calc = float(q_common_pct(lw, ww, wspd))
            s_calc = float(format(q_calc, ".12g"))
            if s_calc != emp:
                r1_mismatches.append((str(r.event_id), arm, s_calc, emp))
            scale = max(1.0, abs(emp))
            rel_diff = abs(q_calc - emp) / scale
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff
    assert len(r1_mismatches) == 0, f"R1 mismatches: {r1_mismatches}"
    checks.append({
        "check": "r1_ref0_flux_formula_12g_equality",
        "passed": True,
        "detail": json.dumps({"max_rel_diff": max_rel_diff, "mismatches": r1_mismatches})
    })
    summary["r1_max_rel_diff"] = max_rel_diff

    # 4. Audit table checks
    audit = pd.read_csv(audit_path)
    assert len(audit) == 1552, f"Expected 1552 audit rows, got {len(audit)}"
    ref0_audit = audit[audit.condition == "REF0"].copy()
    assert len(ref0_audit) == 388, f"Expected 388 REF0 audit rows, got {len(ref0_audit)}"
    merged_audit_flux = ref0_audit.merge(
        inputs[["event_id", "emp_flux_pre_23_23", "emp_flux_post_23_23"]],
        on="event_id"
    )
    for _, r in merged_audit_flux.iterrows():
        emp = float(r.emp_flux_pre_23_23 if r.arm == "pre" else r.emp_flux_post_23_23)
        assert float(r.q_top) == emp, f"REF0 audit q_top != emp for {r.event_id}"
        assert float(r.q_main) == 0.0, f"REF0 audit q_main != 0 for {r.event_id}"
    checks.append({
        "check": "r1_ref0_prescribed_flux_matches_emp_flux_audit",
        "passed": True,
        "detail": "388/388 REF0 audit rows have q_top == emp_flux and q_main == 0.0"
    })

    # 5. Embedded C8 gate (40 events against frozen R4 CF0 15-min)
    resp = pd.read_csv(resp_path)
    r4_resp = pd.read_csv(r4_resp_path)
    subset_ids = set(subset.event_id.astype(str))
    r4_cf0 = r4_resp[
        (r4_resp.model == "M1") & (r4_resp.cohort == "primary") &
        (r4_resp.stage == "CF0") & (r4_resp.horizon_min == 15)
    ].copy()
    r4_subset = r4_cf0[r4_cf0.event_id.astype(str).isin(subset_ids)].copy()
    assert len(r4_subset) == 40

    r5_ref0_15 = resp[
        (resp.condition == "REF0") & (resp.horizon_min == 15) &
        (resp.event_id.astype(str).isin(subset_ids))
    ].copy()
    assert len(r5_ref0_15) == 40

    c8_join = r4_subset[["event_id", "model_T_delta", "model_AH_delta"]].merge(
        r5_ref0_15[["event_id", "dT_model", "dAH_model"]],
        on="event_id", how="outer", indicator=True, validate="one_to_one"
    )
    assert len(c8_join) == 40 and (c8_join["_merge"] == "both").all()
    c8_join["abs_T_difference"] = np.abs(c8_join.dT_model - c8_join.model_T_delta)
    c8_join["abs_AH_difference"] = np.abs(c8_join.dAH_model - c8_join.model_AH_delta)

    max_c8_t = float(c8_join.abs_T_difference.max())
    max_c8_ah = float(c8_join.abs_AH_difference.max())
    count_t_1e12 = int((c8_join.abs_T_difference <= 1e-12).sum())
    count_ah_1e12 = int((c8_join.abs_AH_difference <= 1e-12).sum())
    count_t_12g = int((c8_join.dT_model.apply(lambda v: float(format(v, ".12g"))) == c8_join.model_T_delta).sum())
    count_ah_12g = int((c8_join.dAH_model.apply(lambda v: float(format(v, ".12g"))) == c8_join.model_AH_delta).sum())

    assert max_c8_t <= 1e-6, f"Embedded C8 max T diff exceeds 1e-6: {max_c8_t}"
    assert max_c8_ah <= 1e-6, f"Embedded C8 max AH diff exceeds 1e-6: {max_c8_ah}"

    checks.append({
        "check": "embedded_C8_REF0_40_event_reproduction",
        "passed": True,
        "detail": json.dumps({
            "matched": 40,
            "max_abs_T_difference_C": max_c8_t,
            "max_abs_AH_difference_g_m3": max_c8_ah,
            "tolerance": 1e-6,
            "count_le_1e12_T": count_t_1e12,
            "count_le_1e12_AH": count_ah_1e12,
            "count_12g_equal_T": count_t_12g,
            "count_12g_equal_AH": count_ah_12g,
        })
    })
    summary.update({
        "embedded_c8_events_matched": 40,
        "embedded_c8_max_abs_T_difference_C": max_c8_t,
        "embedded_c8_max_abs_AH_difference_g_m3": max_c8_ah,
        "embedded_c8_count_le_1e12_T": count_t_1e12,
        "embedded_c8_count_le_1e12_AH": count_ah_1e12,
        "embedded_c8_count_12g_equal_T": count_t_12g,
        "embedded_c8_count_12g_equal_AH": count_ah_12g,
    })

    # 6. Pre-FF subset runtime gates (160 rows)
    pre_ff_audit = audit[(audit.condition == "REF0") & (audit.event_id.astype(str).isin(subset_ids))].copy()
    assert len(pre_ff_audit) == 160
    assert pre_ff_audit.finite.astype(bool).all()
    assert pre_ff_audit.init_err.max() <= 1e-8
    assert pre_ff_audit.flux_identity_error.max() <= 1e-12
    assert pre_ff_audit.paired_initial_state_error.max() == 0.0
    assert pre_ff_audit.paired_forcing_error.max() == 0.0
    checks.append({
        "check": "pre_ff_subset_runtime_gates",
        "passed": True,
        "detail": json.dumps({
            "audit_rows": 160,
            "max_init_error": float(pre_ff_audit.init_err.max()),
            "max_flux_identity_error": float(pre_ff_audit.flux_identity_error.max()),
            "max_paired_initial_state_error": float(pre_ff_audit.paired_initial_state_error.max()),
            "max_paired_forcing_error": float(pre_ff_audit.paired_forcing_error.max()),
        })
    })

    # 7. Full runtime gates (776 responses, 1552 audit rows)
    assert len(resp) == 776, f"Expected 776 responses, got {len(resp)}"
    assert resp.event_id.astype(str).nunique() == 97
    assert resp.groupby(["event_id", "condition", "horizon_min"]).size().eq(1).all()
    assert audit.groupby(["event_id", "condition", "arm", "horizon_min"]).size().eq(1).all()

    finite_ok = bool(
        np.isfinite(resp[["dT_model", "dAH_model", "dT_obs", "dAH_obs"]].to_numpy(float)).all()
        and audit.finite.astype(bool).all()
        and np.isfinite(audit[["T_state", "AH_state"]].to_numpy(float)).all()
    )
    assert finite_ok, "Non-finite values found in responses or audit"

    q_vals = audit[["q_top", "q_main"]].to_numpy(float)
    q_ok = bool(np.isfinite(q_vals).all() and (q_vals >= 0).all())
    assert q_ok, "Prescribed fluxes are not finite and non-negative"

    max_init = float(audit.init_err.max())
    max_flux = float(audit.flux_identity_error.max())
    max_pair_x = float(audit.paired_initial_state_error.max())
    max_pair_d = float(audit.paired_forcing_error.max())

    assert max_init <= 1e-8, f"Max initial error {max_init} > 1e-8"
    assert max_flux <= 1e-12, f"Max flux identity error {max_flux} > 1e-12"
    assert max_pair_x == 0.0, f"Max paired initial state error {max_pair_x} != 0"
    assert max_pair_d == 0.0, f"Max paired forcing error {max_pair_d} != 0"

    checks.extend([
        {"check": "all_prescribed_fluxes_finite_nonnegative", "passed": True,
         "detail": f"min/max q across all arms: {[float(np.nanmin(q_vals)), float(np.nanmax(q_vals))]}"},
        {"check": "all_states_outputs_finite_and_initial_state_error", "passed": True,
         "detail": json.dumps({"finite": True, "max_init_error": max_init})},
        {"check": "flux_identity_all_arms_horizons", "passed": True,
         "detail": json.dumps({"audit_rows": 1552, "max_error_m3_m2": max_flux, "tolerance": 1e-12})},
        {"check": "paired_initial_state_and_forcing_identical", "passed": True,
         "detail": json.dumps({"max_initial_state_error": max_pair_x, "max_forcing_error": max_pair_d})},
        {"check": "response_row_count_97x4x2", "passed": True,
         "detail": json.dumps({"rows": 776, "events": 97, "unique_event_condition_horizon": True})},
    ])
    summary.update({
        "events_completed": 97,
        "response_rows": 776,
        "runtime_audit_rows": 1552,
        "all_states_outputs_finite": True,
        "all_prescribed_fluxes_finite_nonnegative": True,
        "max_init_error": max_init,
        "max_flux_identity_error": max_flux,
        "max_paired_initial_state_error": max_pair_x,
        "max_paired_forcing_error": max_pair_d,
        "m1_model_gates_passed": True,
    })

    # 8. Dependency gate: pip freeze byte identity to R4
    r5_pip_bytes = r5_pip_path.read_bytes()
    r4_pip_bytes = r4_pip_path.read_bytes()
    assert r5_pip_bytes == r4_pip_bytes, "PIP freeze bytes do not match frozen R4 freeze"
    checks.append({
        "check": "pip_freeze_byte_identical_to_R4",
        "passed": True,
        "detail": json.dumps({
            "R5_sha256": hashlib.sha256(r5_pip_bytes).hexdigest(),
            "R4_sha256": hashlib.sha256(r4_pip_bytes).hexdigest(),
        })
    })

    # 9. Control-flow evidence from console log
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_lines = log_text.splitlines()
    done_lines = [l for l in log_lines if l.endswith(" done") and l.startswith("R5 ")]
    assert len(done_lines) == 388, f"Expected 388 done lines, got {len(done_lines)}"
    tb_count = log_text.count("Traceback (most recent call last):")
    assert tb_count == 1, f"Expected 1 traceback, got {tb_count}"
    assert "write_outputs" in log_text
    assert "TypeError: Object of type bool_ is not JSON serializable" in log_text
    checks.append({
        "check": "m1_control_flow_and_traceback_audit",
        "passed": True,
        "detail": json.dumps({
            "done_lines_count": 388,
            "traceback_count": 1,
            "traceback_location": "write_outputs",
            "traceback_error": "TypeError: Object of type bool_ is not JSON serializable"
        })
    })

    # 10. Code location for FFH
    summary["ffh_code_location"] = {
        "file": "physbench/r5/m1_r5_formal.py",
        "function": "ODE",
        "parameter": "p[24]",
        "value_nominal": 1.0,
        "value_ffh": 0.3412
    }
    summary["checks"] = checks

    # Verify all checks passed
    assert all(c["passed"] for c in checks), "One or more reconstructed gates failed!"

    # Write output to OUT_DIR and copy to MODEL_DIR if it exists
    out_file = out_dir / "R5_gate_summary_reconstructed.json"
    cleaned_summary = safe_json(summary)
    payload_str = json.dumps(cleaned_summary, indent=2, allow_nan=False) + "\n"
    out_file.write_text(payload_str, encoding="utf-8")
    print(f"Written reconstructed gate summary to {out_file}")

    if model_dir.is_dir():
        model_formal = model_dir / "evidence/R5/FORMAL"
        model_formal.mkdir(parents=True, exist_ok=True)
        (model_formal / "R5_gate_summary_reconstructed.json").write_text(payload_str, encoding="utf-8")
        print(f"Also copied to {model_formal / 'R5_gate_summary_reconstructed.json'}")

    print("\n[PASS] reconstruct_m1_gates passed all assertions.")


if __name__ == "__main__":
    main()
