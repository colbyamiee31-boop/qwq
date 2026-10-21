import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/R5/FORMAL"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261019
BOOT = 2000
VARIABLES = ["T", "AH"]
CONDITIONS = ["HV", "REF0", "FF0", "FFN", "FFH"]
HORIZONS = [15, 30]
COMPARES = [("FF0", "REF0"), ("FF0", "HV"), ("FFH", "FF0")]

PREPARED = ROOT / "prepared"
MODEL = ROOT / "model-output"
BASE_R3 = ROOT / "baseline_r3"
BASE_R4 = ROOT / "baseline_r4"
INPUT_CSV = PREPARED / "PRIMARY_97_R2_INPUT.csv"
EVENT_LIST = ROOT / "physbench/r5/R5_event_list_97_v2.csv"
LOCK = ROOT / "physbench/r5/R5_FORMAL_EXECUTION_LOCK.md"
PREP_GATE = PREPARED / "R5_PREPARE_GATE_SUMMARY.json"
M1_SUMMARY = (OUT / "R5_gate_summary_reconstructed.json") if (OUT / "R5_gate_summary_reconstructed.json").is_file() else (MODEL / "evidence/R5/FORMAL/R5_gate_summary_reconstructed.json")
RESPONSES = MODEL / "evidence/R5/FORMAL/R5_event_responses.csv"
RUNTIME_AUDIT = MODEL / "evidence/R5/FORMAL/R5_runtime_audit.csv"
ALIGNMENT_STATES = MODEL / "evidence/R5/FORMAL/R5_alignment_states.csv"
PIP_R5 = MODEL / "evidence_R5_M1_PIP_FREEZE.txt"
R3_RESP = BASE_R3 / "evidence/R3/M1/ctifl_event_responses.csv"
R3_SUMMARY = BASE_R3 / "evidence/R3/M1/summary.json"
R4_RESP = BASE_R4 / "evidence/R4/M1/ctifl_common_flux_responses.csv"
R4_AUDIT = BASE_R4 / "evidence/R4/M1/runtime_audit.csv"
R4_SUMMARY = BASE_R4 / "evidence/R4/M1/summary.json"
PIP_R4 = BASE_R4 / "evidence_R4_M1_PIP_FREEZE.txt"

CHECKS = []
FAILURES = []
SUMMARY_ROWS = []
PAIRED_ROWS = []
ERROR_TEXT = None
HASH_RECORDS = []


def add(name, ok, detail):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    if not ok:
        FAILURES.append(name)


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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bool_values(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    out = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if out.isna().any():
        raise AssertionError((name, "invalid boolean", sorted(series.astype(str).unique())))
    return out.astype(bool)


def hash_inputs():
    seen = set()
    for folder in [PREPARED, MODEL, BASE_R3, BASE_R4]:
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    rel = path.relative_to(ROOT).as_posix()
                    if rel not in seen:
                        HASH_RECORDS.append({"path": rel, "sha256": sha256(path),
                                             "bytes": path.stat().st_size})
                        seen.add(rel)
    source_files = [
        "physbench/r5/00_R5_CODEX_PROTOCOL.md",
        "physbench/r5/01_PROVENANCE_AND_INPUTS.md",
        "physbench/r5/02_PROTOCOL_CLARIFICATIONS.md",
        "physbench/r5/R5_FORMAL_EXECUTION_LOCK.md",
        "physbench/r5/field_flux.py",
        "physbench/r5/field_ventilation_function.json",
        "physbench/r5/R5_event_list_97_v2.csv",
        "physbench/r5/r4-pip-constraints.txt",
        "physbench/r5/m1_r5_formal.py",
        "physbench/r5/aggregate_r5.py",
        ".github/workflows/physbench_r5_formal.yml",
    ]
    for relative in source_files:
        path = ROOT / relative
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            if rel not in seen:
                HASH_RECORDS.append({"path": rel, "sha256": sha256(path),
                                     "bytes": path.stat().st_size})
                seen.add(rel)
        else:
            add("input_file_present:" + relative, False, str(path))
    return HASH_RECORDS


def git_commit_chain():
    head = os.environ.get("GITHUB_SHA", "unknown")
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "--max-count=3", head],
            cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip().splitlines()
        return {"trigger_commit": out[0] if len(out) > 0 else "unknown",
                "code_commit": out[1] if len(out) > 1 else "unknown",
                "execution_lock_commit": out[2] if len(out) > 2 else "unknown"}
    except Exception as exc:
        return {"error": repr(exc), "trigger_commit": head}


def metric(frame, variable):
    delta_u = frame["delta_u"].to_numpy(float)
    grad = frame["gradient"].to_numpy(float)
    model = frame["model_delta"].to_numpy(float)
    obs = frame["obs_delta"].to_numpy(float)
    denom = np.abs(grad * delta_u)
    eligible = denom > 1e-12
    signs = -np.sign(delta_u) * np.sign(grad)
    n_excluded = int((~eligible).sum())
    fidelity_denominator = float(np.mean(signs[eligible] * obs[eligible])) if eligible.any() else np.nan
    fidelity = (
        float(np.mean(signs[eligible] * model[eligible]) / fidelity_denominator)
        if eligible.any() and abs(fidelity_denominator) > 0 else np.nan
    )
    rmse = float(np.sqrt(np.mean((model - obs) ** 2))) if len(frame) else np.nan
    model_sign = np.array([
        1 if d > 1e-8 * max(1.0, abs(float(hi)), abs(float(lo)))
        else (-1 if d < -1e-8 * max(1.0, abs(float(hi)), abs(float(lo))) else 0)
        for d, hi, lo in zip(model, frame["post_state"], frame["pre_state"])
    ], dtype=int)
    obs_sign = np.array([1 if x > 1e-12 else (-1 if x < -1e-12 else 0) for x in obs], dtype=int)
    resolved = (model_sign != 0) & (obs_sign != 0)
    ms = model_sign[resolved]
    osign = obs_sign[resolved]
    n_resolved = int(len(ms))
    concordance = float(np.mean(ms == osign)) if n_resolved else np.nan
    if n_resolved:
        p0 = float(np.mean(ms == osign))
        pa = float(np.mean(ms == 1))
        pb = float(np.mean(osign == 1))
        pe = pa * pb + (1.0 - pa) * (1.0 - pb)
        kappa = float((p0 - pe) / (1.0 - pe)) if abs(1.0 - pe) >= 1e-15 else np.nan
    else:
        kappa = np.nan
    f = frame.loc[resolved].copy()
    norm_denom = np.abs(f.gradient.to_numpy(float) * f.delta_u.to_numpy(float))
    good = norm_denom > 1e-12
    if int(good.sum()) >= 3:
        rho = float(spearmanr(
            f.loc[good, "model_delta"].to_numpy(float) / norm_denom[good],
            f.loc[good, "obs_delta"].to_numpy(float) / norm_denom[good]
        ).statistic)
    else:
        rho = np.nan
    return {
        "n_events": int(len(frame)), "n_excluded_fidelity": n_excluded,
        "fidelity": fidelity, "rmse": rmse, "n_resolved": n_resolved,
        "n_unresolved": int(len(frame) - n_resolved),
        "concordance": concordance, "kappa": kappa, "spearman_norm": rho,
    }


def confidence_intervals(frame, variable):
    dates = np.array(sorted(frame.event_date.astype(str).unique()))
    groups = {d: frame[frame.event_date.astype(str) == d] for d in dates}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOT):
        sample = rng.choice(dates, size=len(dates), replace=True)
        boot = pd.concat([groups[d] for d in sample], ignore_index=True)
        values.append(metric(boot, variable))
    output = {}
    for name in ["fidelity", "rmse", "concordance", "kappa", "spearman_norm"]:
        vals = np.array([v[name] for v in values], dtype=float)
        vals = vals[np.isfinite(vals)]
        output[name] = (
            [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if len(vals) else [np.nan, np.nan]
        )
    return output


def make_r5_canonical(response, states, inputs):
    response = response.copy()
    states = states.copy()
    response["event_id"] = response.event_id.astype(str)
    states["event_id"] = states.event_id.astype(str)
    response["horizon_min"] = response.horizon_min.astype(int)
    states["horizon_min"] = states.horizon_min.astype(int)
    combined = response.merge(states, on=["event_id", "condition", "horizon_min"],
                              how="outer", indicator=True, validate="one_to_one")
    if len(combined) != len(response) or not (combined["_merge"] == "both").all():
        raise AssertionError("R5 response/alignment state join is incomplete")
    meta_cols = [
        "event_id", "row_index", "event_date", "direction", "delta_u",
        "T_gradient_C", "AH_gradient_g_m3", "anchor_primary_3_6",
        "obs_T_matched_15", "obs_AH_matched_15", "obs_T_matched_30", "obs_AH_matched_30",
    ]
    metadata = inputs[meta_cols].copy()
    metadata["event_id"] = metadata.event_id.astype(str)
    metadata = metadata.rename(columns={c: c + "_input" for c in meta_cols if c != "event_id"})
    combined["event_id"] = combined.event_id.astype(str)
    combined = combined.merge(metadata, on="event_id", how="left", suffixes=("", ""),
                              indicator="input_join", validate="many_to_one")
    if not (combined.input_join == "both").all():
        raise AssertionError("R5 output contains event IDs absent from the frozen input")
    for req_col in [c + "_input" for c in meta_cols if c != "event_id"]:
        if req_col not in combined.columns:
            raise AssertionError(f"Missing column after metadata merge in make_r5_canonical: {req_col}")
    if not (combined.row_index.astype(int) == combined.row_index_input.astype(int)).all():
        raise AssertionError("R5 response row_index differs from the frozen input")
    if not (bool_values(combined.anchor_primary_3_6, "R5 response anchor").to_numpy() ==
            bool_values(combined.anchor_primary_3_6_input, "input anchor").to_numpy()).all():
        raise AssertionError("R5 response subset flag differs from the frozen input")
    if not (combined.event_date.astype(str) == combined.event_date_input.astype(str)).all():
        raise AssertionError("R5 response event_date differs from the frozen input")
    combined["anchor_primary_3_6"] = bool_values(combined.anchor_primary_3_6_input, "input anchor")
    combined["model_delta_T"] = combined.dT_model.astype(float)
    combined["model_delta_AH"] = combined.dAH_model.astype(float)
    combined["pre_T"] = combined.pre_T.astype(float)
    combined["post_T"] = combined.post_T.astype(float)
    combined["pre_AH"] = combined.pre_AH.astype(float)
    combined["post_AH"] = combined.post_AH.astype(float)
    combined["event_date"] = combined.event_date_input.astype(str)
    combined["delta_u"] = combined.delta_u_input.astype(float)
    combined["T_gradient_C"] = combined.T_gradient_C_input.astype(float)
    combined["AH_gradient_g_m3"] = combined.AH_gradient_g_m3_input.astype(float)
    combined["dT_obs_expected"] = combined.apply(
        lambda r: float(r.obs_T_matched_15_input if r.horizon_min == 15 else r.obs_T_matched_30_input), axis=1)
    combined["dAH_obs_expected"] = combined.apply(
        lambda r: float(r.obs_AH_matched_15_input if r.horizon_min == 15 else r.obs_AH_matched_30_input), axis=1)
    if (np.abs(combined.dT_obs.astype(float) - combined.dT_obs_expected) > 1e-10).any():
        raise AssertionError("R5 dT_obs differs from the frozen observed matched effect")
    if (np.abs(combined.dAH_obs.astype(float) - combined.dAH_obs_expected) > 1e-10).any():
        raise AssertionError("R5 dAH_obs differs from the frozen observed matched effect")
    combined["obs_delta_T"] = combined.dT_obs_expected
    combined["obs_delta_AH"] = combined.dAH_obs_expected
    return combined


def make_hv_canonical(hv_source, inputs):
    hv = hv_source[
        (hv_source.model == "M1") & (hv_source.stage == "HV")
        & (hv_source.cohort == "primary") & (hv_source.horizon_min.isin(HORIZONS))
    ].copy()
    if len(hv) != 194 or any(len(hv[hv.horizon_min == h]) != 97 for h in HORIZONS):
        raise AssertionError(("frozen_HV_rows_not_97_per_horizon", len(hv)))
    meta_cols = [
        "event_id", "row_index", "event_date", "direction", "delta_u",
        "T_gradient_C", "AH_gradient_g_m3", "anchor_primary_3_6",
        "obs_T_matched_15", "obs_AH_matched_15", "obs_T_matched_30", "obs_AH_matched_30",
    ]
    input_meta = inputs[meta_cols].copy()
    input_meta["event_id"] = input_meta.event_id.astype(str)
    input_meta = input_meta.rename(columns={c: c + "_input" for c in meta_cols if c != "event_id"})
    hv["event_id"] = hv.event_id.astype(str)
    hv = hv.merge(input_meta, on="event_id", how="outer", suffixes=("", ""),
                  indicator=True, validate="many_to_one")
    if not (hv._merge == "both").all() or len(hv) != 194:
        raise AssertionError("HV reference does not align with all frozen primary events")
    for req_col in [c + "_input" for c in meta_cols if c != "event_id"]:
        if req_col not in hv.columns:
            raise AssertionError(f"Missing column after metadata merge in make_hv_canonical: {req_col}")
    if not (hv.event_date.astype(str) == hv.event_date_input.astype(str)).all():
        raise AssertionError("HV event_date differs from frozen input")
    for r3_name, input_name in [
        ("delta_u", "delta_u_input"),
        ("T_gradient_C", "T_gradient_C_input"),
        ("AH_gradient_g_m3", "AH_gradient_g_m3_input"),
    ]:
        if (np.abs(hv[r3_name].astype(float) - hv[input_name].astype(float)) > 1e-10).any():
            raise AssertionError("HV metadata differs from frozen input: " + r3_name)
    hv["condition"] = "HV"
    hv["event_date"] = hv.event_date_input.astype(str)
    hv["anchor_primary_3_6"] = bool_values(hv.anchor_primary_3_6_input, "input anchor")
    hv["delta_u"] = hv.delta_u_input.astype(float)
    hv["T_gradient_C"] = hv.T_gradient_C_input.astype(float)
    hv["AH_gradient_g_m3"] = hv.AH_gradient_g_m3_input.astype(float)
    hv["horizon_min"] = hv.horizon_min.astype(int)
    hv["model_delta_T"] = hv.model_T_delta.astype(float)
    hv["model_delta_AH"] = hv.model_AH_delta.astype(float)
    hv["pre_T"] = hv.pre_T.astype(float)
    hv["post_T"] = hv.post_T.astype(float)
    hv["pre_AH"] = hv.pre_AH.astype(float)
    hv["post_AH"] = hv.post_AH.astype(float)
    hv["obs_delta_T"] = hv.apply(
        lambda r: float(r.obs_T_matched_15_input if r.horizon_min == 15 else r.obs_T_matched_30_input), axis=1)
    hv["obs_delta_AH"] = hv.apply(
        lambda r: float(r.obs_AH_matched_15_input if r.horizon_min == 15 else r.obs_AH_matched_30_input), axis=1)
    for column in ["q_pre_top", "q_post_top", "q_pre_main", "q_post_main"]:
        hv[column] = np.nan
    hv["q_note"] = "Frozen R3 HV response artifact has no arm-level q columns; HV was reused without rerun."
    return hv


def compute_summary(canonical):
    for cohort in ["primary", "anchor_primary_3_6"]:
        for condition in CONDITIONS:
            for horizon in HORIZONS:
                for variable in VARIABLES:
                    frame = canonical[(canonical.condition == condition) &
                                      (canonical.horizon_min == horizon)].copy()
                    if cohort == "anchor_primary_3_6":
                        frame = frame[frame.anchor_primary_3_6.astype(bool)].copy()
                    expected_n = 97 if cohort == "primary" else 40
                    if len(frame) != expected_n or frame.event_id.astype(str).nunique() != expected_n:
                        raise AssertionError(("summary_cohort_count", cohort, condition, horizon, len(frame)))
                    frame["model_delta"] = frame["model_delta_" + variable].astype(float)
                    frame["obs_delta"] = frame["obs_delta_" + variable].astype(float)
                    frame["gradient"] = frame["T_gradient_C" if variable == "T" else "AH_gradient_g_m3"].astype(float)
                    frame["pre_state"] = frame["pre_" + variable].astype(float)
                    frame["post_state"] = frame["post_" + variable].astype(float)
                    point = metric(frame, variable)
                    ci = confidence_intervals(frame, variable)
                    q_available = condition != "HV"
                    if q_available:
                        qstats = {
                            "q_pre_top_mean": float(frame.q_pre_top.mean()),
                            "q_post_top_mean": float(frame.q_post_top.mean()),
                            "q_delta_top_mean": float((frame.q_post_top - frame.q_pre_top).mean()),
                            "q_pre_main_mean": float(frame.q_pre_main.mean()),
                            "q_post_main_mean": float(frame.q_post_main.mean()),
                            "q_delta_main_mean": float((frame.q_post_main - frame.q_pre_main).mean()),
                            "q_note": "Prescribed R5 flux summarized from the pre/post arms.",
                        }
                    else:
                        qstats = {
                            "q_pre_top_mean": np.nan, "q_post_top_mean": np.nan,
                            "q_delta_top_mean": np.nan, "q_pre_main_mean": np.nan,
                            "q_post_main_mean": np.nan, "q_delta_main_mean": np.nan,
                            "q_note": "Frozen R3 HV has no arm-level q columns; no HV model rerun was performed.",
                        }
                    SUMMARY_ROWS.append({
                        "cohort": cohort, "condition": condition, "variable": variable,
                        "horizon_min": horizon, **point,
                        "fidelity_ci_low": ci["fidelity"][0], "fidelity_ci_high": ci["fidelity"][1],
                        "rmse_ci_low": ci["rmse"][0], "rmse_ci_high": ci["rmse"][1],
                        "concordance_ci_low": ci["concordance"][0],
                        "concordance_ci_high": ci["concordance"][1],
                        "kappa_ci_low": ci["kappa"][0], "kappa_ci_high": ci["kappa"][1],
                        "spearman_norm_ci_low": ci["spearman_norm"][0],
                        "spearman_norm_ci_high": ci["spearman_norm"][1],
                        "bootstrap_resamples": BOOT, "bootstrap_seed": SEED,
                        "q_available": q_available, **qstats,
                    })


def paired_bootstrap(frame_a, frame_b, variable, endpoint):
    dates = np.array(sorted(frame_a.event_date.astype(str).unique()))
    groups_a = {d: frame_a[frame_a.event_date.astype(str) == d] for d in dates}
    groups_b = {d: frame_b[frame_b.event_date.astype(str) == d] for d in dates}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOT):
        sample = rng.choice(dates, size=len(dates), replace=True)
        a = pd.concat([groups_a[d] for d in sample], ignore_index=True)
        b = pd.concat([groups_b[d] for d in sample], ignore_index=True)
        sa = metric(a, variable)
        sb = metric(b, variable)
        if endpoint == "abs_fidelity_error":
            values.append(abs(sa["fidelity"] - 1.0) - abs(sb["fidelity"] - 1.0))
        else:
            values.append(sa["rmse"] - sb["rmse"])
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [np.nan, np.nan]


def compute_paired(canonical):
    for cohort in ["primary", "anchor_primary_3_6"]:
        for variable in VARIABLES:
            for horizon in HORIZONS:
                frames = {}
                for condition in CONDITIONS:
                    f = canonical[(canonical.condition == condition) &
                                  (canonical.horizon_min == horizon)].copy()
                    if cohort == "anchor_primary_3_6":
                        f = f[f.anchor_primary_3_6.astype(bool)].copy()
                    f["model_delta"] = f["model_delta_" + variable].astype(float)
                    f["obs_delta"] = f["obs_delta_" + variable].astype(float)
                    f["gradient"] = f["T_gradient_C" if variable == "T" else "AH_gradient_g_m3"].astype(float)
                    f["pre_state"] = f["pre_" + variable].astype(float)
                    f["post_state"] = f["post_" + variable].astype(float)
                    frames[condition] = f
                for a_name, b_name in COMPARES:
                    fa, fb = frames[a_name], frames[b_name]
                    ia, ib = set(fa.event_id.astype(str)), set(fb.event_id.astype(str))
                    paired_ok = ia == ib and len(ia) == len(fa) == len(fb)
                    add("paired_event_alignment:" + cohort + ":" + variable + ":" +
                        str(horizon) + ":" + a_name + "_minus_" + b_name,
                        paired_ok, "paired event IDs=" + str(len(ia)) + "; both sets equal=" + str(ia == ib))
                    if not paired_ok:
                        continue
                    for endpoint in ["abs_fidelity_error", "rmse"]:
                        sa, sb = metric(fa, variable), metric(fb, variable)
                        if endpoint == "abs_fidelity_error":
                            estimate = abs(sa["fidelity"] - 1.0) - abs(sb["fidelity"] - 1.0)
                        else:
                            estimate = sa["rmse"] - sb["rmse"]
                        ci = paired_bootstrap(fa, fb, variable, endpoint)
                        PAIRED_ROWS.append({
                            "cohort": cohort, "variable": variable, "horizon_min": horizon,
                            "comparison": a_name + " - " + b_name, "endpoint": endpoint,
                            "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1],
                            "n_events": len(fa), "bootstrap_resamples": BOOT,
                            "bootstrap_seed": SEED,
                        })


def write_placeholder_outputs():
    if not (OUT / "R5_summary.csv").exists():
        pd.DataFrame(columns=["cohort", "condition", "variable", "horizon_min"]).to_csv(OUT / "R5_summary.csv", index=False)
    if not (OUT / "R5_paired_differences.csv").exists():
        pd.DataFrame(columns=["cohort", "variable", "horizon_min", "comparison", "endpoint"]).to_csv(OUT / "R5_paired_differences.csv", index=False)


def main():
    global ERROR_TEXT
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    repo = os.environ.get("GITHUB_REPOSITORY", "unknown")
    run_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + repo + "/actions/runs/" + run_id
    commit_info = git_commit_chain()
    identities = json.loads(os.environ.get("ARTIFACT_IDENTITIES_JSON", "{}"))
    prepare_result = os.environ.get("PREPARE_RESULT", "unknown")
    m1_result = os.environ.get("M1_RESULT", "unknown")
    guard_result = os.environ.get("GUARD_RESULT", "unknown")

    prepare_gate = read_json(PREP_GATE)
    m1_summary = read_json(M1_SUMMARY)
    if not m1_summary.get("reconstructed_from_saved_outputs") or m1_summary.get("source_run") != "39429049087":
        raise AssertionError("M1_SUMMARY must be reconstructed from run 39429049087 outputs")
    add("one_time_guard_succeeded", guard_result == "success", guard_result)
    add("prepare_job_succeeded", prepare_result == "success", prepare_result)
    add("M1_job_succeeded", m1_result == "success", m1_result)
    add("prepare_unit_test_and_input_hash_gates", prepare_gate.get("status") == "PASS",
        json.dumps({"status": prepare_gate.get("status"), "failed_checks": prepare_gate.get("failed_checks", [])}))
    add("M1_runtime_gates", m1_summary.get("m1_model_gates_passed") is True and m1_summary.get("status") == "PASS",
        json.dumps({"status": m1_summary.get("status"), "events": m1_summary.get("events_completed"),
                    "response_rows": m1_summary.get("response_rows"),
                    "failed_checks": [x for x in m1_summary.get("checks", []) if not x.get("passed")]}))

    artifact_checks = identities.get("checks", [])
    add("frozen_and_current_artifact_digests_verified",
        bool(identities.get("status") == "PASS" and artifact_checks and all(x.get("passed") for x in artifact_checks)),
        json.dumps({"status": identities.get("status"), "checks": artifact_checks}))

    source_hashes = hash_inputs()
    add("required_input_hash_records_complete",
        all(item.get("sha256") and item.get("bytes") is not None for item in source_hashes),
        "file hash records=" + str(len(source_hashes)))

    inputs = pd.read_csv(INPUT_CSV)
    inputs["event_id"] = inputs.event_id.astype(str)
    inputs["anchor_primary_3_6"] = bool_values(inputs.anchor_primary_3_6, "prepared input anchor")
    event_list = pd.read_csv(EVENT_LIST)
    event_list["event_id"] = event_list.event_id.astype(str)
    event_list["row_index"] = event_list.row_index.astype(int)
    inputs["row_index"] = inputs.row_index.astype(int)
    if len(inputs) != 97 or inputs.event_id.nunique() != 97 or inputs.row_index.nunique() != 97:
        raise AssertionError(("primary_input_not_97_unique", len(inputs)))
    flags = inputs.merge(event_list[["row_index", "event_id", "in_R4_wind_3_6_subset"]],
                         on="row_index", suffixes=("", "_event_list"),
                         validate="one_to_one", indicator=True)
    if len(flags) != 97 or not (flags._merge == "both").all():
        raise AssertionError("event list row_index join failed")
    if not (flags.event_id.astype(str) == flags.event_id_event_list.astype(str)).all():
        raise AssertionError("event_id mismatch against v2 event list")
    list_flags = bool_values(flags.in_R4_wind_3_6_subset, "event-list flag")
    if not (flags.anchor_primary_3_6.to_numpy() == list_flags.to_numpy()).all():
        raise AssertionError("40-event flag mismatch against event-list v2")
    subset_ids = set(inputs.loc[inputs.anchor_primary_3_6, "event_id"].astype(str))
    add("frozen_97_events_and_anchor_primary_3_6_40",
        len(subset_ids) == 40 and len(event_list) == 97,
        json.dumps({"primary": len(inputs), "subset": len(subset_ids),
                    "row_index_join": 97, "flags_agree": True}))

    response = pd.read_csv(RESPONSES)
    audit = pd.read_csv(RUNTIME_AUDIT)
    states = pd.read_csv(ALIGNMENT_STATES)
    response["event_id"] = response.event_id.astype(str)
    response["condition"] = response.condition.astype(str)
    response["horizon_min"] = response.horizon_min.astype(int)
    schema = {
        "event_id", "row_index", "event_date", "direction", "anchor_primary_3_6",
        "condition", "horizon_min", "dT_model", "dAH_model", "dT_obs", "dAH_obs",
        "q_pre_top", "q_post_top", "q_pre_main", "q_post_main", "hfac",
    }
    add("R5_response_schema", schema.issubset(response.columns), ",".join(response.columns))
    response_count_ok = (
        len(response) == 776 and response.event_id.astype(str).nunique() == 97
        and set(response.condition.astype(str)) == {"REF0", "FF0", "FFN", "FFH"}
        and set(response.horizon_min.astype(int)) == {15, 30}
        and not response.duplicated(["event_id", "condition", "horizon_min"]).any()
    )
    add("R5_response_count_776", response_count_ok,
        json.dumps({"rows": len(response), "events": int(response.event_id.astype(str).nunique())}))

    canonical_r5 = make_r5_canonical(response, states, inputs)
    hv_source = pd.read_csv(R3_RESP)
    hv_used = hv_source[
        (hv_source.model == "M1") & (hv_source.stage == "HV")
        & (hv_source.cohort == "primary") & (hv_source.horizon_min.isin(HORIZONS))
    ].copy()
    hv_used.to_csv(OUT / "R5_HV_reference_used.csv", index=False, float_format="%.12g")
    canonical_hv = make_hv_canonical(hv_source, inputs)
    canonical = pd.concat([canonical_hv, canonical_r5], ignore_index=True, sort=False)
    add("frozen_R3_HV_reference_97_each_horizon",
        len(hv_used) == 194 and all(len(hv_used[hv_used.horizon_min == h]) == 97 for h in HORIZONS),
        json.dumps({"rows": len(hv_used), "horizon_counts": {str(h): int(len(hv_used[hv_used.horizon_min == h])) for h in HORIZONS}}))

    r4 = pd.read_csv(R4_RESP)
    r4["event_id"] = r4.event_id.astype(str)
    r4_summary = read_json(R4_SUMMARY)
    r4_audit = pd.read_csv(R4_AUDIT)
    r4_cf0 = r4[(r4.model == "M1") & (r4.cohort == "primary") &
                (r4.stage == "CF0") & (r4.horizon_min == 15)].copy()
    r4_cf0 = r4_cf0[r4_cf0.event_id.astype(str).isin(subset_ids)]
    ref0_15 = canonical_r5[(canonical_r5.condition == "REF0") &
                           (canonical_r5.horizon_min == 15) &
                           (canonical_r5.event_id.astype(str).isin(subset_ids))]
    joined = r4_cf0[["event_id", "model_T_delta", "model_AH_delta"]].merge(
        ref0_15[["event_id", "model_delta_T", "model_delta_AH"]],
        on="event_id", how="outer", indicator=True, validate="one_to_one")
    joined["abs_T_difference"] = np.abs(joined.model_delta_T - joined.model_T_delta)
    joined["abs_AH_difference"] = np.abs(joined.model_delta_AH - joined.model_AH_delta)
    joined.to_csv(OUT / "R5_embedded_C8_event_comparison.csv", index=False, float_format="%.12g")
    max_c8_t = float(joined.abs_T_difference.max()) if len(joined) else float("inf")
    max_c8_ah = float(joined.abs_AH_difference.max()) if len(joined) else float("inf")
    c8_ok = len(joined) == 40 and bool((joined._merge == "both").all()) and max_c8_t <= 1e-6 and max_c8_ah <= 1e-6
    r4_audit_cf0 = r4_audit[(r4_audit.stage == "CF0") & (r4_audit.cohort == "primary")]
    r4_runtime_ok = (
        r4_summary.get("gate_pass") is True and len(r4_audit_cf0) == 80
        and bool(r4_audit_cf0.finite.astype(bool).all())
        and float(r4_audit_cf0.init_err.max()) <= 1e-8
        and float(r4_audit_cf0.flux_identity_error.max()) <= 1e-12
    )
    add("frozen_R4_CF0_runtime_audits", r4_runtime_ok,
        json.dumps({"summary_gate_pass": r4_summary.get("gate_pass"), "rows": len(r4_audit_cf0),
                    "max_init_error": float(r4_audit_cf0.init_err.max()) if len(r4_audit_cf0) else None,
                    "max_flux_identity_error": float(r4_audit_cf0.flux_identity_error.max()) if len(r4_audit_cf0) else None}))
    add("embedded_C8_REF0_40_event_reproduction", c8_ok,
        json.dumps({"matched": int((joined._merge == "both").sum()),
                    "max_abs_T_difference_C": max_c8_t,
                    "max_abs_AH_difference_g_m3": max_c8_ah, "tolerance": 1e-6}))

    runtime_checks = {
        "all_states_and_outputs_finite": bool(audit.finite.astype(bool).all()) and bool(np.isfinite(audit[["T_state", "AH_state"]].to_numpy(float)).all()),
        "fluxes_finite_nonnegative": bool(np.isfinite(audit[["q_top", "q_main"]].to_numpy(float)).all() and (audit[["q_top", "q_main"]].to_numpy(float) >= 0).all()),
        "flux_identity": len(audit) == 1552 and float(audit.flux_identity_error.max()) <= 1e-12,
        "initial_state": len(audit) == 1552 and float(audit.init_err.max()) <= 1e-8,
        "paired_initial_state": len(audit) == 1552 and float(audit.paired_initial_state_error.max()) == 0.0,
        "paired_forcing": len(audit) == 1552 and float(audit.paired_forcing_error.max()) == 0.0,
    }
    for name, ok in runtime_checks.items():
        add("M1_runtime_" + name, ok,
            json.dumps({"audit_rows": len(audit),
                        "max_flux_identity_error": float(audit.flux_identity_error.max()) if len(audit) else None,
                        "max_init_error": float(audit.init_err.max()) if len(audit) else None,
                        "max_paired_initial_state_error": float(audit.paired_initial_state_error.max()) if len(audit) else None,
                        "max_paired_forcing_error": float(audit.paired_forcing_error.max()) if len(audit) else None}))

    pip_match = PIP_R5.is_file() and PIP_R4.is_file() and PIP_R5.read_bytes() == PIP_R4.read_bytes()
    add("pip_freeze_byte_identical_to_R4", pip_match,
        json.dumps({"R4_sha256": sha256(PIP_R4) if PIP_R4.is_file() else None,
                    "R5_sha256": sha256(PIP_R5) if PIP_R5.is_file() else None}))

    compute_summary(canonical)
    compute_paired(canonical)
    pd.DataFrame(SUMMARY_ROWS).to_csv(OUT / "R5_summary.csv", index=False, float_format="%.12g", na_rep="")
    pd.DataFrame(PAIRED_ROWS).to_csv(OUT / "R5_paired_differences.csv", index=False,
                                     float_format="%.12g", na_rep="")
    response.to_csv(OUT / "R5_event_responses.csv", index=False, float_format="%.12g")
    audit.to_csv(OUT / "R5_runtime_audit.csv", index=False, float_format="%.12g")

    overall_pass = not FAILURES
    gate_summary = {
        "status": "PASS" if overall_pass else "FAIL",
        "passed": overall_pass,
        "workflow_run_id": run_id, "workflow_run_url": run_url,
        "branch": "r5-field-ventilation-flux", "head_sha": os.environ.get("GITHUB_SHA"),
        "commit_chain": commit_info,
        "greenlight_commit": "2d3febb1ea002b24b452e32293e990beb78d3ce1",
        "events_completed": int(response.event_id.astype(str).nunique()),
        "response_rows": len(response), "conditions": ["REF0", "FF0", "FFN", "FFH"],
        "horizons_min": [15, 30],
        "c8_max_abs_T_difference_C": max_c8_t,
        "c8_max_abs_AH_difference_g_m3": max_c8_ah,
        "c8_tolerance": 1e-6,
        "bootstrap_resamples": BOOT, "bootstrap_seed": SEED,
        "failed_checks": FAILURES, "checks": CHECKS,
        "artifact_identities": identities,
        "input_hashes": HASH_RECORDS,
    }
    (OUT / "R5_gate_summary.json").write_text(
        json.dumps(safe_json(gate_summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    ffh_line = m1_summary.get("ffh_code_location", {})
    trigger_audit = os.environ.get("TRIGGER_AUDIT", "Not provided by guard.")
    lock_text = LOCK.read_text(encoding="utf-8") if LOCK.is_file() else ""
    c8_record = next((line for line in lock_text.splitlines() if "All five recorded copies" in line), "§0 confirmation missing.")
    lines = [
        "# R5 formal run log", "",
        "- Status: " + ("PASS" if overall_pass else "FAIL"),
        "- GitHub Actions run: " + run_url,
        "- Run ID: " + run_id,
        "- Branch: r5-field-ventilation-flux",
        "- Commit 1 (execution lock): " + commit_info.get("execution_lock_commit", "unknown"),
        "- Commit 2 (formal code and workflow): " + commit_info.get("code_commit", "unknown"),
        "- Commit 3 (one-time trigger only): " + commit_info.get("trigger_commit", "unknown"),
        "- GreenLight-Gym2 commit: 2d3febb1ea002b24b452e32293e990beb78d3ce1",
        "- Conditions: REF0, FF0, FFN, FFH; horizons 15 and 30 minutes.",
        "- Events: " + str(int(response.event_id.astype(str).nunique())) + " unique events; response rows " + str(len(response)) + ".",
        "- Bootstrap: 2,000 date-cluster resamples, seed 20261019.",
        "- Embedded C8: " + ("PASS" if c8_ok else "FAIL") +
        "; max |dT|=" + str(max_c8_t) + ", max |dAH|=" + str(max_c8_ah) + "; tolerance 1e-6.",
        "- Frozen R4 runtime audits: " + ("PASS" if r4_runtime_ok else "FAIL"),
        "- pip freeze byte match to R4: " + str(pip_match),
        "- FFH heat-factor code location: " + str(ffh_line.get("file")) + ":" + str(ffh_line.get("line")),
        "- §0 log-copy confirmation: " + c8_record,
        "- Trigger audit: " + trigger_audit,
        "",
        "## Artifact identities", "",
        "| Name | ID | Producer run | SHA256 digest | Status |",
        "|---|---:|---:|---|---|",
    ]
    for item in identities.get("artifacts", []):
        lines.append("| " + str(item.get("name")) + " | " + str(item.get("id")) + " | " +
                     str(item.get("run_id")) + " | " + str(item.get("digest")) + " | " +
                     str(item.get("status")) + " |")
    lines += ["", "## Runtime gates", "", "| Gate | Result | Detail |", "|---|---|---|"]
    for item in CHECKS:
        lines.append("| " + item["check"] + " | " + ("PASS" if item["passed"] else "FAIL") +
                     " | " + item["detail"].replace("|", "\\|").replace("\n", " ") + " |")
    lines += ["", "## Input file SHA256 records", "", "| Path | SHA256 | Bytes |", "|---|---|---:|"]
    for item in HASH_RECORDS:
        lines.append("| " + item["path"] + " | " + item["sha256"] + " | " + str(item["bytes"]) + " |")
    lines += [
        "", "## Logs and outputs", "",
        "- R5_event_responses.csv",
        "- R5_summary.csv",
        "- R5_paired_differences.csv",
        "- R5_HV_reference_used.csv",
        "- R5_runtime_audit.csv",
        "- R5_alignment_states.csv",
        "- R5_embedded_C8_event_comparison.csv",
        "- logs/field_flux_unit_tests.log",
        "- logs/prepare_validation.log",
        "- logs/r5_formal_m1_console.log",
        "- logs/aggregate_console.log",
        "",
        "The r5-formal-final artifact digest is recorded in GitHub Actions artifact metadata after upload.",
    ]
    (OUT / "R5_run_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not overall_pass:
        raise SystemExit("R5 formal gate failure; see R5_gate_summary.json")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        ERROR_TEXT = traceback.format_exc()
        add("aggregate_execution", False, ERROR_TEXT)
        write_placeholder_outputs()
        failure_summary = {
            "status": "FAIL", "passed": False,
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "failed_checks": FAILURES, "checks": CHECKS,
            "error": ERROR_TEXT, "input_hashes": HASH_RECORDS,
        }
        (OUT / "R5_gate_summary.json").write_text(
            json.dumps(safe_json(failure_summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (OUT / "R5_run_log.md").write_text(
            "# R5 formal run log\n\nStatus: FAIL\n\n" + ERROR_TEXT + "\n", encoding="utf-8")
        raise SystemExit("R5 aggregate failed; see failure report")

