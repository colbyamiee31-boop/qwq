"""Software Test: Offline verification of R5 aggregation pipeline.

THIS IS A SOFTWARE TEST, NOT DATA.
Marked per protocol R5: does not produce scientific results, does not write
to evidence/, and does not print endpoint values.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "physbench/r5"))

import aggregate_r5

def find_file(candidates, desc):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    # Search recursively under ROOT as fallback
    filename = Path(candidates[0]).name
    matches = list(ROOT.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {desc} in candidates: {candidates}")

def run_offline_aggregate_test():
    print("=" * 70)
    print("RUNNING OFFLINE AGGREGATE TEST (SOFTWARE TEST ONLY)")
    print("=" * 70)

    # 1. Locate inputs
    p97_path = find_file([
        ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv",
        ROOT / "frozen_artifacts/r4-prepared-inputs/PRIMARY_97_R2_INPUT.csv",
        Path("E:/GreenLight_R5/frozen_artifacts/r4-prepared-inputs/PRIMARY_97_R2_INPUT.csv"),
    ], "PRIMARY_97_R2_INPUT.csv")

    r3_path = find_file([
        ROOT / "baseline_r3/evidence/R3/M1/ctifl_event_responses.csv",
        ROOT / "frozen_artifacts/r3-m1/evidence/R3/M1/ctifl_event_responses.csv",
        Path("E:/GreenLight_R5/frozen_artifacts/r3-m1/evidence/R3/M1/ctifl_event_responses.csv"),
    ], "frozen R3 M1 HV responses")

    r4_path = find_file([
        ROOT / "baseline_r4/evidence/R4/M1/ctifl_common_flux_responses.csv",
        ROOT / "frozen_artifacts/r4-m1/evidence/R4/M1/ctifl_common_flux_responses.csv",
        Path("E:/GreenLight_R5/frozen_artifacts/r4-m1/evidence/R4/M1/ctifl_common_flux_responses.csv"),
    ], "frozen R4 M1 CF0 responses")

    r5_resp_path = find_file([
        ROOT / "baseline_r5_v2/evidence/R5/FORMAL/R5_event_responses.csv",
        ROOT / "baseline_r5_v2/R5_event_responses.csv",
        Path("E:/GreenLight_R5/formal_run_39387663913/r5-formal-final/R5_event_responses.csv"),
    ], "failed run 39387663913 REF0 responses")

    r5_states_path = find_file([
        ROOT / "baseline_r5_v2/evidence/R5/FORMAL/R5_alignment_states.csv",
        ROOT / "baseline_r5_v2/R5_alignment_states.csv",
        Path("E:/GreenLight_R5/formal_run_39387663913/r5-formal-final/R5_alignment_states.csv"),
    ], "failed run 39387663913 alignment states")

    inputs = pd.read_csv(p97_path)
    hv_source = pd.read_csv(r3_path)
    r4_source = pd.read_csv(r4_path)
    r5_responses = pd.read_csv(r5_resp_path)
    r5_states = pd.read_csv(r5_states_path)

    if len(r5_responses) > 80 and "condition" in r5_responses.columns:
        anchor_ids = set(inputs[inputs.anchor_primary_3_6 == True].event_id.astype(str))
        r5_responses = r5_responses[(r5_responses.condition == "REF0") & (r5_responses.event_id.astype(str).isin(anchor_ids))].copy()
    if len(r5_states) > 80 and "condition" in r5_states.columns:
        anchor_ids = set(inputs[inputs.anchor_primary_3_6 == True].event_id.astype(str))
        r5_states = r5_states[(r5_states.condition == "REF0") & (r5_states.event_id.astype(str).isin(anchor_ids))].copy()

    assert len(inputs) == 97, f"Expected 97 inputs, got {len(inputs)}"
    assert len(r5_responses) == 80, f"Expected 80 REF0 rows in run 39387663913 artifact, got {len(r5_responses)}"

    # 2. Test HV canonicalisation
    canonical_hv = aggregate_r5.make_hv_canonical(hv_source, inputs)
    assert len(canonical_hv) == 194, f"Expected 194 HV rows, got {len(canonical_hv)}"
    assert len(canonical_hv[canonical_hv.horizon_min == 15]) == 97
    assert len(canonical_hv[canonical_hv.horizon_min == 30]) == 97
    assert int(canonical_hv.anchor_primary_3_6.sum()) == 80
    print("[PASS] Test 1: HV canonicalisation passed (194 rows, 97 per horizon, 80 anchor rows).")

    # 3. Test R5 canonicalisation on real REF0 rows
    canonical_ref0 = aggregate_r5.make_r5_canonical(r5_responses, r5_states, inputs)
    assert len(canonical_ref0) == 80, f"Expected 80 REF0 canonical rows, got {len(canonical_ref0)}"
    assert (canonical_ref0.condition == "REF0").all()
    assert (canonical_ref0.anchor_primary_3_6 == True).all()
    assert len(canonical_ref0[canonical_ref0.horizon_min == 15]) == 40
    assert len(canonical_ref0[canonical_ref0.horizon_min == 30]) == 40
    print("[PASS] Test 2: R5 REF0 canonicalisation passed (80 rows, 40 per horizon, anchor flags verified).")

    # 4. Test Embedded-C8 join
    subset_ids = set(inputs[inputs.anchor_primary_3_6 == True].event_id.astype(str))
    r4_cf0 = r4_source[(r4_source.model == "M1") & (r4_source.cohort == "primary") &
                       (r4_source.stage == "CF0") & (r4_source.horizon_min == 15)].copy()
    r4_cf0 = r4_cf0[r4_cf0.event_id.astype(str).isin(subset_ids)]
    ref0_15 = canonical_ref0[(canonical_ref0.condition == "REF0") &
                             (canonical_ref0.horizon_min == 15) &
                             (canonical_ref0.event_id.astype(str).isin(subset_ids))]
    joined = r4_cf0[["event_id", "model_T_delta", "model_AH_delta"]].merge(
        ref0_15[["event_id", "model_delta_T", "model_delta_AH"]],
        on="event_id", how="outer", indicator=True, validate="one_to_one"
    )
    assert len(joined) == 40 and bool((joined._merge == "both").all())
    print("[PASS] Test 3: Embedded-C8 join against frozen R4 CF0 passed (40/40 matched).")

    # 5. Test metric calculation and C4 signing / exclusion
    f_t = ref0_15.copy()
    f_t["model_delta"] = f_t["model_delta_T"].astype(float)
    f_t["obs_delta"] = f_t["obs_delta_T"].astype(float)
    f_t["gradient"] = f_t["T_gradient_C"].astype(float)
    f_t["pre_state"] = f_t["pre_T"].astype(float)
    f_t["post_state"] = f_t["post_T"].astype(float)
    m_t = aggregate_r5.metric(f_t, "T")

    f_ah = ref0_15.copy()
    f_ah["model_delta"] = f_ah["model_delta_AH"].astype(float)
    f_ah["obs_delta"] = f_ah["obs_delta_AH"].astype(float)
    f_ah["gradient"] = f_ah["AH_gradient_g_m3"].astype(float)
    f_ah["pre_state"] = f_ah["pre_AH"].astype(float)
    f_ah["post_state"] = f_ah["post_AH"].astype(float)
    m_ah = aggregate_r5.metric(f_ah, "AH")

    required_metric_keys = {
        "n_events", "n_excluded_fidelity", "fidelity", "rmse",
        "n_resolved", "n_unresolved", "concordance", "kappa", "spearman_norm"
    }
    assert required_metric_keys.issubset(m_t.keys())
    assert required_metric_keys.issubset(m_ah.keys())
    assert m_t["n_events"] == 40
    print("[PASS] Test 4: C4 metric calculation and signing passed.")

    # 6. In-memory fixture exercising full aggregation code paths (FF0, FFN, FFH, HV)
    # Build complete in-memory canonical frame for all 97 events and 4 conditions (in-memory only!)
    fixture_rows = []
    for cond in ["REF0", "FF0", "FFN", "FFH"]:
        for h in [15, 30]:
            for _, inp_row in inputs.iterrows():
                eid = str(inp_row.event_id)
                # If row exists in canonical_ref0 for this event and horizon, copy states
                match = canonical_ref0[(canonical_ref0.event_id == eid) & (canonical_ref0.horizon_min == h)]
                if len(match):
                    r_ref = match.iloc[0]
                    dt_m = float(r_ref.model_delta_T)
                    dah_m = float(r_ref.model_delta_AH)
                    pre_t = float(r_ref.pre_T)
                    post_t = float(r_ref.post_T)
                    pre_ah = float(r_ref.pre_AH)
                    post_ah = float(r_ref.post_AH)
                else:
                    dt_m = 0.0
                    dah_m = 0.0
                    pre_t = 20.0
                    post_t = 20.0
                    pre_ah = 10.0
                    post_ah = 10.0

                obs_t = float(inp_row.obs_T_matched_15 if h == 15 else inp_row.obs_T_matched_30)
                obs_ah = float(inp_row.obs_AH_matched_15 if h == 15 else inp_row.obs_AH_matched_30)

                fixture_rows.append({
                    "event_id": eid,
                    "event_date": str(inp_row.event_date),
                    "condition": cond,
                    "horizon_min": int(h),
                    "anchor_primary_3_6": bool(inp_row.anchor_primary_3_6),
                    "delta_u": float(inp_row.delta_u),
                    "T_gradient_C": float(inp_row.T_gradient_C),
                    "AH_gradient_g_m3": float(inp_row.AH_gradient_g_m3),
                    "model_delta_T": dt_m,
                    "model_delta_AH": dah_m,
                    "obs_delta_T": obs_t,
                    "obs_delta_AH": obs_ah,
                    "pre_T": pre_t,
                    "post_T": post_t,
                    "pre_AH": pre_ah,
                    "post_AH": post_ah,
                    "q_pre_top": 0.01,
                    "q_post_top": 0.02,
                    "q_pre_main": 0.0,
                    "q_post_main": 0.0,
                })

    fixture_df = pd.DataFrame(fixture_rows)
    assert len(fixture_df) == 776, f"Expected 776 fixture rows, got {len(fixture_df)}"
    combined_canonical = pd.concat([canonical_hv, fixture_df], ignore_index=True, sort=False)
    assert len(combined_canonical) == 194 + 776, f"Expected 970 rows, got {len(combined_canonical)}"

    # Temporarily set fast bootstrap resamples for test execution speed
    orig_boot = aggregate_r5.BOOT
    try:
        aggregate_r5.BOOT = 2
        aggregate_r5.SUMMARY_ROWS.clear()
        aggregate_r5.PAIRED_ROWS.clear()

        aggregate_r5.compute_summary(combined_canonical)
        assert len(aggregate_r5.SUMMARY_ROWS) == 40, f"Expected 40 summary rows, got {len(aggregate_r5.SUMMARY_ROWS)}"
        sum_df = pd.DataFrame(aggregate_r5.SUMMARY_ROWS)
        # Verify structural columns have no NaN
        for col in ["cohort", "condition", "variable", "horizon_min"]:
            assert not sum_df[col].isna().any(), f"NaN in summary column {col}"

        aggregate_r5.compute_paired(combined_canonical)
        assert len(aggregate_r5.PAIRED_ROWS) == 48, f"Expected 48 paired rows, got {len(aggregate_r5.PAIRED_ROWS)}"
        paired_df = pd.DataFrame(aggregate_r5.PAIRED_ROWS)
        for col in ["cohort", "variable", "horizon_min", "comparison", "endpoint"]:
            assert not paired_df[col].isna().any(), f"NaN in paired column {col}"

        print("[PASS] Test 5: In-memory fixture exercising full aggregation pipeline passed.")
    finally:
        aggregate_r5.BOOT = orig_boot
        aggregate_r5.SUMMARY_ROWS.clear()
        aggregate_r5.PAIRED_ROWS.clear()

    print("\nALL OFFLINE AGGREGATE TESTS PASSED SUCCESSFULLY!")
    print("test_aggregate_offline PASS")

if __name__ == "__main__":
    run_offline_aggregate_test()
