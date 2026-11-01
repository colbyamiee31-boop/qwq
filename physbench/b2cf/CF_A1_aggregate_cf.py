import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cf-a1-main-output"
SEED = 20261019
BOOT = 2000
VARIABLES = ["T", "AH"]
CONDITIONS = ["HV", "REF0", "FF0", "FFN", "FFH", "FF0-CF", "FFH-CF"]
HORIZONS = [15, 30]
COMPARES = [("FF0", "REF0"), ("FF0", "HV"), ("FFH", "FF0"),
            ("FF0-CF", "REF0"), ("FFH-CF", "FF0-CF"),
            ("FF0-CF", "HV"), ("FF0-CF", "FF0")]
CHECKS = []
FAILURES = []
SUMMARY_ROWS = []
PAIRED_ROWS = []
NATURE = 'B2 按 v3.2 冻结协议执行时，G4 未通过，已依该协议终止（GitHub Actions run 41308725059）。B2-A1 是在这次失败之后另立的修订协议，按它得到的结果是"修订协议下的结果"，**不是** v3.2 的预注册验证。'

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

def main():
    input_path = ROOT / "cf-a1-input/PRIMARY_97_R2_INPUT.csv"
    r5_response = ROOT / "cf-a1-r5/evidence/R5/FORMAL/R5_event_responses.csv"
    r5_states = ROOT / "cf-a1-r5/evidence/R5/FORMAL/R5_alignment_states.csv"
    r3_response = ROOT / "cf-a1-r3/evidence/R3/M1/ctifl_event_responses.csv"
    locked = {
        input_path: "0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b",
        r5_response: "0a0a427641ef96cf511bd5401ad40edd19106591fbc07fbac20636e3f2a81c49",
        r5_states: "a11efaeef73fdf8b4a47e1d267b123b703fc1615d5a85aa257a2d6859b5fbe07",
        r3_response: "f102765b460b1646e82017a6f6c9761f4b072b7eab027ce6b23238557c6a8ffe",
    }
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in locked}
    if not all(sha256(p) == value for p, value in locked.items()):
        raise RuntimeError("frozen historical input hash mismatch")
    gate = read_json(OUT / "CF_A1_gate_summary.json")
    if not (gate.get("status") == "PASS" and gate.get("m1_model_gates_passed") is True
            and gate.get("response_rows") == 388 and gate.get("runtime_audit_rows") == 776
            and gate.get("C8_environment_event_count") == 40
            and all(x.get("passed") for x in gate.get("checks", []))):
        raise RuntimeError("main run audit is not PASS")
    inputs = pd.read_csv(input_path)
    if len(inputs) != 97 or inputs.event_id.nunique() != 97 or inputs.row_index.nunique() != 97:
        raise RuntimeError("frozen input is not 97 unique events")
    historical = pd.read_csv(r5_response)
    historical_states = pd.read_csv(r5_states)
    current = pd.read_csv(OUT / "CF_A1_event_responses.csv")
    current_states = pd.read_csv(OUT / "CF_A1_alignment_states.csv")
    def exact_keys(df, conditions, count):
        return (len(df) == count and df.event_id.astype(str).nunique() == 97
                and set(df.condition.astype(str)) == set(conditions)
                and set(df.horizon_min.astype(int)) == {15, 30}
                and not df.duplicated(["event_id", "condition", "horizon_min"]).any())
    if not (exact_keys(historical, ["REF0", "FF0", "FFN", "FFH"], 776)
            and exact_keys(historical_states, ["REF0", "FF0", "FFN", "FFH"], 776)
            and exact_keys(current, ["FF0-CF", "FFH-CF"], 388)
            and exact_keys(current_states, ["FF0-CF", "FFH-CF"], 388)):
        raise RuntimeError("input response or state identities/counts invalid")
    canon_r5 = make_r5_canonical(historical, historical_states, inputs)
    canon_cf = make_r5_canonical(current, current_states, inputs)
    canon_hv = make_hv_canonical(pd.read_csv(r3_response), inputs)
    canonical = pd.concat([canon_hv, canon_r5, canon_cf], ignore_index=True, sort=False)
    compute_summary(canonical)
    compute_paired(canonical)
    if FAILURES:
        raise RuntimeError("paired alignment failure: " + repr(FAILURES))
    endpoints = pd.DataFrame(SUMMARY_ROWS)
    paired = pd.DataFrame(PAIRED_ROWS)
    endpoints.to_csv(OUT / "CF_A1_endpoints.csv", index=False, float_format="%.12g", na_rep="")
    paired.to_csv(OUT / "CF_A1_paired_differences.csv", index=False, float_format="%.12g", na_rep="")
    rows = ["# B2-A1 预设判读", "", "**性质声明：** " + NATURE, ""]
    for variable in VARIABLES:
        x = paired[(paired.cohort == "primary") & (paired.variable == variable)
                   & (paired.horizon_min == 15) & (paired.comparison == "FF0-CF - REF0")
                   & (paired.endpoint == "abs_fidelity_error")]
        if len(x) != 1:
            raise RuntimeError("missing or duplicate prespecified primary comparison")
        v = x.iloc[0]
        if not all(np.isfinite(float(v[k])) for k in ("estimate", "ci_low", "ci_high")):
            text = "区间不可用；协议 §C4 未提供此情形的预设措辞，需报告作者审查。"
        elif v.ci_high < 0:
            text = "改善在日期折外仍然成立"
        elif v.ci_low <= 0 <= v.ci_high:
            text = "日期折外未能确认改善"
        else:
            text = "区间完全高于 0；协议 §C4 未提供此情形的预设措辞，需报告作者审查。"
        rows.extend([f"- {variable}：{text}；Δ|fidelity − 1| = {v.estimate:.8g}，95% 区间 [{v.ci_low:.8g}, {v.ci_high:.8g}]。"])
    rows.extend(["", "FF0-CF − HV：两者在通路和渗漏上也不同。", ""])
    (OUT / "CF_A1_interpretation.md").write_text("\n".join(rows), encoding="utf-8")
    report = {"nature_statement": NATURE, "status": "PASS", "seed": SEED, "bootstrap": BOOT,
              "conditions": CONDITIONS, "comparisons": COMPARES, "checks": CHECKS,
              "historical_input_sha256": hashes,
              "main_output_sha256": {x: sha256(OUT / x) for x in ("CF_A1_event_responses.csv", "CF_A1_alignment_states.csv", "CF_A1_runtime_audit.csv", "CF_A1_gate_summary.json")}}
    (OUT / "CF_A1_aggregate_summary.json").write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stable = ["CF_A1_event_responses.csv", "CF_A1_alignment_states.csv", "CF_A1_runtime_audit.csv",
              "CF_A1_gate_summary.json", "CF_A1_C8_REF0_responses.csv", "CF_A1_C8_runtime_audit.csv",
              "CF_A1_endpoints.csv", "CF_A1_paired_differences.csv", "CF_A1_interpretation.md",
              "CF_A1_aggregate_summary.json", "CF_A1_pip_freeze.txt", "CF_A1_pip_freeze_diff.txt"]
    log = ["# B2-A1 主运行记录", "", "**性质声明：** " + NATURE, "",
           "- GitHub Actions run ID: " + os.environ.get("GITHUB_RUN_ID", "unknown"),
           "- GitHub commit: " + os.environ.get("GITHUB_SHA", "unknown"),
           "- Model: 2d3febb1ea002b24b452e32293e990beb78d3ce1",
           "- Main: 97 events, FF0-CF / FFH-CF, two arms, 15 and 30 minutes.",
           "- Environment-only C8: 40 events, REF0, two arms; separately archived.",
           "- Date bootstrap: 2,000 replicates, seed 20261019.",
           "- R5 and R3 references are read-only; FF0-CF versus HV also differs in routing and leakage.",
           "", "## SHA-256: frozen reference inputs", ""]
    log += ["- `" + name + "`: `" + digest + "`" for name, digest in hashes.items()]
    log += ["", "## SHA-256: stable outputs", ""]
    log += ["- `" + name + "`: `" + sha256(OUT / name) + "`" for name in stable]
    (OUT / "CF_A1_run_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        detail = {"nature_statement": NATURE, "status": "FAIL", "error": traceback.format_exc(),
                  "checks": CHECKS, "failed_checks": FAILURES}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "CF_A1_aggregate_failure.json").write_text(json.dumps(safe_json(detail), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
