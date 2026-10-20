import json
import sys
import traceback
from pathlib import Path

import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np
import pandas as pd
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "physbench/r5"))
from common import *  # noqa: E402,F403
from field_flux import q_field  # noqa: E402

CONDITIONS = ['REF0', 'FF0', 'FFN', 'FFH']
HORIZONS = [15, 30]
STEP_SECONDS = 900.0
INPUT_PATH = ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv"
EVENT_LIST_PATH = ROOT / "physbench/r5/R5_event_list_97_v2.csv"
LOCK_PATH = ROOT / "physbench/r5/R5_FORMAL_EXECUTION_LOCK.md"
PREPARE_GATE_PATH = ROOT / "physbench/r5/generated/R5_PREPARE_GATE_SUMMARY.json"
R4_REFERENCE_PATH = ROOT / "baseline_c8/evidence/R4/M1/ctifl_common_flux_responses.csv"
R4_AUDIT_PATH = ROOT / "baseline_c8/evidence/R4/M1/runtime_audit.csv"
R4_SUMMARY_PATH = ROOT / "baseline_c8/evidence/R4/M1/summary.json"
OUT = ROOT / "evidence/R5/FORMAL"
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_COLUMNS = [
    "event_id", "row_index", "event_date", "direction", "anchor_primary_3_6",
    "condition", "horizon_min", "dT_model", "dAH_model", "dT_obs", "dAH_obs",
    "q_pre_top", "q_post_top", "q_pre_main", "q_post_main", "hfac",
]
AUDIT_COLUMNS = [
    "event_id", "row_index", "event_date", "condition", "arm", "horizon_min",
    "q_top", "q_main", "flow_volume_integrated", "flow_volume_expected",
    "flux_identity_error", "init_err", "finite", "paired_initial_state_error",
    "paired_forcing_error", "T_state", "AH_state",
]
STATE_COLUMNS = [
    "event_id", "condition", "horizon_min", "pre_T", "post_T", "pre_AH", "post_AH",
]

response_rows = []
audit_rows = []
state_rows = []
checks = []
failure_text = None
summary = {
    "status": "RUNNING",
    "greenlight_commit": "2d3febb1ea002b24b452e32293e990beb78d3ce1",
    "conditions": CONDITIONS,
    "horizons_min": HORIZONS,
    "events_expected": 97,
    "subset_expected": 40,
    "tolerance_c8": 1e-6,
}


def add_check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    if not passed:
        raise RuntimeError("gate failed: " + name + ": " + str(detail))


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bool_values(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise AssertionError((name, "non-boolean value", sorted(series.astype(str).unique())))
    return mapped.astype(bool)


def locked_hashes():
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(LOCK_PATH)
    found = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name = parts[1].strip()
        digest = parts[2].strip()
        if len(name) > 2 and len(digest) > 2:
            name, digest = name[1:-1], digest[1:-1]
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower()):
                found[name] = digest.lower()
    required = [
        "physbench/r5/00_R5_CODEX_PROTOCOL.md",
        "physbench/r5/01_PROVENANCE_AND_INPUTS.md",
        "physbench/r5/02_PROTOCOL_CLARIFICATIONS.md",
        "physbench/r5/field_flux.py",
        "physbench/r5/field_ventilation_function.json",
        "physbench/r5/R5_event_list_97_v2.csv",
    ]
    missing = [name for name in required if name not in found]
    if missing:
        raise AssertionError(("execution lock missing hashes", missing))
    return {name: found[name] for name in required}


def verify_source_hashes():
    expected = locked_hashes()
    actual = {}
    for relative, want in expected.items():
        path = ROOT / relative
        got = sha256(path)
        actual[relative] = got
        if got != want:
            raise AssertionError(("execution lock hash mismatch", relative, want, got))
    return actual


def build_weather(r):
    d = np.zeros(10, dtype=float)
    d[0] = float(r.event_Iglob)
    d[1] = float(r.event_Tout)
    d[2] = float(r.out_vp_pa)
    d[3] = float(ppm_to_mg_m3(float(r.event_Tout), float(r.outdoor_co2_ppm_fixed)))
    d[4] = float(r.event_Windsp)
    d[5] = float(r.sky_temperature_c_proxy)
    d[6] = float(r.soil_boundary_temperature_c_fixed)
    d[7] = float(r.event_Iglob) * STEP_SECONDS / 1e6
    d[8] = 1.0 if float(r.event_Iglob) > 0 else 0.0
    d[9] = d[8]
    return d


def full_action(value):
    return np.array([0, 0, 0, float(value), 0, 0], dtype=float)


def initial_state(r):
    xx = X_NATIVE.copy()
    xx[2] = float(r.event_Tair)
    xx[15] = float(r.in_vp_pa)
    xx[0] = float(ppm_to_mg_m3(float(r.event_Tair), float(r.CO2_pre_ppm)))
    xx[4] = float(getattr(r, "canopy_temperature_c_closure", r.event_Tair))
    return xx


def prescribed_flux(r, arm, condition):
    suffix = "pre" if arm == "pre" else "post"
    leeward = float(getattr(r, f"leeward_{suffix}_pct"))
    windward = float(getattr(r, f"windward_{suffix}_pct"))
    wind = float(r.event_Windsp)
    q_ref = float(q_common_pct(leeward, windward, wind))
    q_field = float(q_field_fn(leeward / 100.0, windward / 100.0, wind,
                               float(r.event_Tair), float(r.event_Tout)))
    if condition == "REF0":
        return q_ref, 0.0, 1.0
    if condition == "FF0":
        return q_field, 0.0, 1.0
    if condition == "FFN":
        leak = float(m1_native_leakage(P_HV, wind))
        return q_field + float(P_HV[204]) * leak, (1.0 - float(P_HV[204])) * leak, 1.0
    if condition == "FFH":
        return q_field, 0.0, 0.3412
    raise ValueError(condition)


q_field_fn = q_field


def run_arm(r, arm, condition, paired_x_error, paired_d_error):
    action = float(r.u_pre if arm == "pre" else r.u_post)
    xx = initial_state(r)
    dd = build_weather(r)
    uu = full_action(action)
    qt, qm, hfac = prescribed_flux(r, arm, condition)
    if not (np.isfinite(qt) and np.isfinite(qm) and qt >= 0 and qm >= 0):
        raise AssertionError(("prescribed_flux_invalid", r.event_id, condition, arm, qt, qm))
    if not np.isfinite(hfac) or hfac <= 0:
        raise AssertionError(("heat_factor_invalid", r.event_id, condition, hfac))
    pvec = np.concatenate([dd, P_HV, [qt, qm, hfac]])
    rr = FCF(x0=xx, u=uu, p=pvec)
    x900 = np.asarray(rr["xf"].full()).reshape(-1)
    dv900 = float(np.asarray(rr["qf"].full()).reshape(-1)[0])
    rr = FCF(x0=x900, u=uu, p=pvec)
    x1800 = np.asarray(rr["xf"].full()).reshape(-1)
    dv_step_2 = float(np.asarray(rr["qf"].full()).reshape(-1)[0])
    finite = bool(np.all(np.isfinite(xx)) and np.all(np.isfinite(x900)) and np.all(np.isfinite(x1800)))
    if not finite:
        raise RuntimeError(("nonfinite_state", r.event_id, condition, arm))
    states = {}
    arm_audit = []
    init_err = max(
        abs(float(xx[2]) - float(r.event_Tair)),
        abs(float(xx[15]) - float(r.in_vp_pa)),
        abs(float(mg_m3_to_ppm(xx[2], xx[0])) - float(r.CO2_pre_ppm)),
        abs(float(xx[4]) - float(getattr(r, "canopy_temperature_c_closure", r.event_Tair))),
    )
    for horizon, xf in [(15, x900), (30, x1800)]:
        T = float(xf[2])
        AH = float(ah_from_t_vp(T, float(xf[15])))
        seconds = horizon * 60.0
        volume = dv900 if horizon == 15 else dv900 + dv_step_2
        expected = float((qt + qm) * seconds)
        arm_audit.append({
            "event_id": str(r.event_id), "row_index": int(r.row_index),
            "event_date": str(r.event_date), "condition": condition, "arm": arm,
            "horizon_min": horizon, "q_top": float(qt), "q_main": float(qm),
            "flow_volume_integrated": volume, "flow_volume_expected": expected,
            "flux_identity_error": abs(volume - expected), "init_err": float(init_err),
            "finite": finite, "paired_initial_state_error": float(paired_x_error),
            "paired_forcing_error": float(paired_d_error), "T_state": T, "AH_state": AH,
        })
        states[horizon] = {"T": T, "AH": AH}
    return {"states": states, "audit": arm_audit, "q_top": float(qt),
            "q_main": float(qm), "hfac": float(hfac), "finite": finite}


def response_rows_for(r, condition, pre, post):
    rows = []
    for horizon in HORIZONS:
        p = pre["states"][horizon]
        q = post["states"][horizon]
        obs_t = float(getattr(r, f"obs_T_matched_{horizon}"))
        obs_ah = float(getattr(r, f"obs_AH_matched_{horizon}"))
        rows.append({
            "event_id": str(r.event_id), "row_index": int(r.row_index),
            "event_date": str(r.event_date), "direction": str(r.direction),
            "anchor_primary_3_6": bool(r.anchor_primary_3_6),
            "condition": condition, "horizon_min": horizon,
            "dT_model": q["T"] - p["T"], "dAH_model": q["AH"] - p["AH"],
            "dT_obs": obs_t, "dAH_obs": obs_ah,
            "q_pre_top": pre["q_top"], "q_post_top": post["q_top"],
            "q_pre_main": pre["q_main"], "q_post_main": post["q_main"],
            "hfac": pre["hfac"],
        })
        state_rows.append({
            "event_id": str(r.event_id), "condition": condition, "horizon_min": horizon,
            "pre_T": p["T"], "post_T": q["T"], "pre_AH": p["AH"], "post_AH": q["AH"],
        })
    audit_rows.extend(pre["audit"])
    audit_rows.extend(post["audit"])
    return rows


def write_outputs():
    pd.DataFrame(response_rows, columns=RESPONSE_COLUMNS).to_csv(
        OUT / "R5_event_responses.csv", index=False, float_format="%.12g")
    pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS).to_csv(
        OUT / "R5_runtime_audit.csv", index=False, float_format="%.12g")
    pd.DataFrame(state_rows, columns=STATE_COLUMNS).to_csv(
        OUT / "R5_alignment_states.csv", index=False, float_format="%.12g")
    summary["checks"] = checks
    summary["status"] = "FAIL" if failure_text or any(not c["passed"] for c in checks) else "PASS"
    if failure_text:
        summary["error"] = failure_text
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value
    safe_summary = finite_json(summary)
    (OUT / "R5_gate_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(safe_summary, indent=2, allow_nan=False), flush=True)


def main():
    global q_field_fn, X_NATIVE, P_NATIVE, NX, NU, ND, P_HV, FCF
    prep_gate = json.loads(PREPARE_GATE_PATH.read_text(encoding="utf-8"))
    add_check("prepare_unit_test_and_lock_hash_gates",
              prep_gate.get("status") == "PASS",
              json.dumps({"status": prep_gate.get("status"),
                          "failed_checks": prep_gate.get("failed_checks", [])}))
    source_hashes = verify_source_hashes()
    summary["source_sha256"] = source_hashes
    add_check("execution_lock_source_hashes",
              all(source_hashes.values()),
              json.dumps(source_hashes, sort_keys=True))
    E = pd.read_csv(INPUT_PATH)
    event_list = pd.read_csv(EVENT_LIST_PATH)
    if len(E) != 97 or E.event_id.astype(str).nunique() != 97 or E.row_index.nunique() != 97:
        raise AssertionError(("primary_input_count_or_duplicates", len(E),
                              E.event_id.astype(str).nunique(), E.row_index.nunique()))
    if len(event_list) != 97 or event_list.event_id.astype(str).nunique() != 97 or event_list.row_index.nunique() != 97:
        raise AssertionError(("event_list_count_or_duplicates", len(event_list)))
    E["row_index"] = E.row_index.astype(int)
    event_list["row_index"] = event_list.row_index.astype(int)
    merged = E.merge(event_list[["row_index", "event_id", "in_R4_wind_3_6_subset"]],
                     on="row_index", how="outer", suffixes=("", "_event_list"),
                     indicator=True, validate="one_to_one")
    if len(merged) != 97 or not (merged["_merge"] == "both").all():
        raise AssertionError(("row_index_join_not_97_of_97", len(merged),
                              merged.loc[merged["_merge"] != "both", "row_index"].tolist()))
    if not (merged.event_id.astype(str) == merged.event_id_event_list.astype(str)).all():
        raise AssertionError("event_id_mismatch_after_row_index_join")
    E["anchor_primary_3_6"] = bool_values(E.anchor_primary_3_6, "input.anchor_primary_3_6")
    event_flag = bool_values(merged.in_R4_wind_3_6_subset, "event_list.in_R4_wind_3_6_subset")
    if not (E.anchor_primary_3_6.to_numpy() == event_flag.to_numpy()).all():
        raise AssertionError("anchor_primary_3_6 flag mismatch against event-list v2")
    subset = E[E.anchor_primary_3_6].copy()
    if len(subset) != 40 or subset.event_id.astype(str).nunique() != 40:
        raise AssertionError(("anchor_primary_3_6_not_40", len(subset)))
    checks.append({"check": "event_join_and_anchor_primary_3_6",
                   "passed": True, "detail": "97/97 row_index and event_id; subset flags agree 40/40."})
    summary["primary_event_count"] = len(E)
    summary["anchor_primary_3_6_count"] = len(subset)

    r4 = pd.read_csv(R4_REFERENCE_PATH)
    r4_summary = json.loads(R4_SUMMARY_PATH.read_text(encoding="utf-8"))
    r4_audit = pd.read_csv(R4_AUDIT_PATH)
    if r4_summary.get("gate_pass") is not True:
        raise AssertionError(("frozen_R4_M1_runtime_gate_not_pass", r4_summary))
    r4_cf0 = r4[(r4.model == "M1") & (r4.cohort == "primary") &
                (r4.stage == "CF0") & (r4.horizon_min == 15)].copy()
    r4_subset = r4_cf0[r4_cf0.event_id.astype(str).isin(set(subset.event_id.astype(str)))]
    if len(r4_cf0) != 40 or len(r4_subset) != 40 or set(r4_subset.event_id.astype(str)) != set(subset.event_id.astype(str)):
        raise AssertionError(("frozen_R4_CF0_reference_not_exact_40", len(r4_cf0), len(r4_subset)))
    r4_audit_cf0 = r4_audit[(r4_audit.stage == "CF0") & (r4_audit.cohort == "primary")]
    r4_runtime_ok = (
        len(r4_audit_cf0) == 80 and bool(r4_audit_cf0.finite.astype(bool).all())
        and float(r4_audit_cf0.init_err.max()) <= 1e-8
        and float(r4_audit_cf0.flux_identity_error.max()) <= 1e-12
    )
    checks.append({"check": "frozen_R4_CF0_runtime_audits",
                   "passed": r4_runtime_ok,
                   "detail": json.dumps({"rows": len(r4_audit_cf0),
                       "max_init_error": float(r4_audit_cf0.init_err.max()) if len(r4_audit_cf0) else None,
                       "max_flux_identity_error": float(r4_audit_cf0.flux_identity_error.max()) if len(r4_audit_cf0) else None})})
    if not r4_runtime_ok:
        raise AssertionError("frozen R4 CF0 runtime audits failed")

    env = gym.make(
        "gl_gym/GreenLightTomato-v0",
        controlled_inputs=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"],
        normalize_actions=False, parameter_provider="fixed")
    env.reset(seed=3407, options={"scenario": {"location": "Amsterdam",
                                               "growth_year": 2010, "start_day": 244}})
    e = env.unwrapped
    X_NATIVE = e.x.copy()
    P_NATIVE = np.asarray(e.p, dtype=float).copy()
    NX, NU, ND = e.nx, e.nu, e.nd
    env.close()
    P_HV = harmonise_m1_hv(P_NATIVE)
    assert abs(P_HV[49] - H_TARGET) <= 1e-12
    assert abs(P_HV[55] / P_HV[46] - VENT_PROJECTED_RATIO) <= 1e-12

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    d = ca.SX.sym("d", ND)
    p = ca.SX.sym("p", len(P_NATIVE))
    qtop = ca.SX.sym("qtop")
    qmain = ca.SX.sym("qmain")
    hfac = ca.SX.sym("hfac")
    a = aux_states.update(x, u, d, p)
    dx = ODE(x, u, d, p)

    def sensible(hec, t1, t2):
        return ca.fabs(hec) * (t1 - t2)

    def air_mv(flow, vp1, vp2, t1, t2):
        return 0.002165 * ca.fabs(flow) * (vp1 / (t1 + 273.15) - vp2 / (t2 + 273.15))

    def air_mc(flow, c1, c2):
        return ca.fabs(flow) * (c1 - c2)

    native_top = a[136]
    native_main = a[137] + a[145]
    dx[0] += (air_mc(native_main, x[0], d[3]) - air_mc(qmain, x[0], d[3])) / p[122]
    dx[1] += (air_mc(native_top, x[1], d[3]) - air_mc(qtop, x[1], d[3])) / p[123]
    cp_rho = p[111] * p[23]
    dx[2] += (sensible(cp_rho * native_main, x[2], d[1]) -
              sensible(cp_rho * qmain, x[2], d[1])) / p[112]
    dx[3] += (sensible(cp_rho * native_top, x[3], d[1]) -
              sensible(hfac * cp_rho * qtop, x[3], d[1])) / p[120]
    dx[15] += (air_mv(native_main, x[15], d[2], x[2], d[1]) -
               air_mv(qmain, x[15], d[2], x[2], d[1])) / a[35]
    dx[16] += (air_mv(native_top, x[16], d[2], x[3], d[1]) -
               air_mv(qtop, x[16], d[2], x[3], d[1])) / a[36]
    FCF = ca.integrator(
        "R5_M1_CF900", "cvodes",
        {"x": x, "u": u, "p": ca.vertcat(d, p, qtop, qmain, hfac),
         "ode": dx, "quad": qtop + qmain},
        0.0, STEP_SECONDS,
        {"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 90000})
    summary["ffh_code_location"] = {
        "file": "physbench/r5/m1_r5_formal.py",
        "line": next(i for i, line in enumerate(Path(__file__).read_text(encoding="utf-8").splitlines(), 1)
                     if "sensible(hfac * cp_rho * qtop" in line),
    }

    for _, r in subset.iterrows():
        xx0, xx1 = initial_state(r), initial_state(r)
        dd0, dd1 = build_weather(r), build_weather(r)
        pair_x_error = float(np.max(np.abs(xx0 - xx1)))
        pair_d_error = float(np.max(np.abs(dd0 - dd1)))
        pre = run_arm(r, "pre", "REF0", pair_x_error, pair_d_error)
        post = run_arm(r, "post", "REF0", pair_x_error, pair_d_error)
        response_rows.extend(response_rows_for(r, "REF0", pre, post))
        print("R5 pre-FF REF0 reproduction", str(r.event_id), "done", flush=True)
    pre_gate_audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    pre_gate_responses = pd.DataFrame(response_rows, columns=RESPONSE_COLUMNS)
    c8_15 = pre_gate_responses[pre_gate_responses.horizon_min == 15].copy()
    c8_join = r4_subset[["event_id", "model_T_delta", "model_AH_delta"]].merge(
        c8_15[["event_id", "dT_model", "dAH_model"]],
        on="event_id", how="outer", indicator=True, validate="one_to_one")
    c8_join["abs_T_difference"] = np.abs(c8_join.dT_model - c8_join.model_T_delta)
    c8_join["abs_AH_difference"] = np.abs(c8_join.dAH_model - c8_join.model_AH_delta)
    c8_join.to_csv(OUT / "R5_embedded_C8_event_comparison.csv", index=False, float_format="%.12g")
    max_c8_t = float(c8_join.abs_T_difference.max()) if len(c8_join) else float("inf")
    max_c8_ah = float(c8_join.abs_AH_difference.max()) if len(c8_join) else float("inf")
    c8_alignment = len(c8_join) == 40 and bool((c8_join["_merge"] == "both").all())
    c8_early_audit = (
        len(pre_gate_audit) == 160
        and bool(pre_gate_audit.finite.astype(bool).all())
        and float(pre_gate_audit.init_err.max()) <= 1e-8
        and float(pre_gate_audit.flux_identity_error.max()) <= 1e-12
        and float(pre_gate_audit.paired_initial_state_error.max()) == 0.0
        and float(pre_gate_audit.paired_forcing_error.max()) == 0.0
    )
    summary.update({
        "embedded_c8_events_matched": int((c8_join["_merge"] == "both").sum()),
        "embedded_c8_max_abs_T_difference_C": max_c8_t,
        "embedded_c8_max_abs_AH_difference_g_m3": max_c8_ah,
    })
    add_check("embedded_C8_REF0_40_event_reproduction",
              c8_alignment and max_c8_t <= 1e-6 and max_c8_ah <= 1e-6,
              json.dumps({"matched": int((c8_join["_merge"] == "both").sum()),
                          "max_abs_T_difference_C": max_c8_t,
                          "max_abs_AH_difference_g_m3": max_c8_ah,
                          "tolerance": 1e-6}))
    add_check("pre_FF_subset_runtime_gates", c8_early_audit,
              json.dumps({"audit_rows": len(pre_gate_audit),
                          "max_init_error": float(pre_gate_audit.init_err.max()) if len(pre_gate_audit) else None,
                          "max_flux_identity_error": float(pre_gate_audit.flux_identity_error.max()) if len(pre_gate_audit) else None,
                          "max_paired_initial_state_error": float(pre_gate_audit.paired_initial_state_error.max()) if len(pre_gate_audit) else None,
                          "max_paired_forcing_error": float(pre_gate_audit.paired_forcing_error.max()) if len(pre_gate_audit) else None}))
    summary["pre_ff_c8_gate_passed"] = True

    subset_ids = set(subset.event_id.astype(str))
    for _, r in E.iterrows():
        for condition in CONDITIONS:
            if condition == "REF0" and str(r.event_id) in subset_ids:
                continue
            xx0, xx1 = initial_state(r), initial_state(r)
            dd0, dd1 = build_weather(r), build_weather(r)
            pair_x_error = float(np.max(np.abs(xx0 - xx1)))
            pair_d_error = float(np.max(np.abs(dd0 - dd1)))
            pre = run_arm(r, "pre", condition, pair_x_error, pair_d_error)
            post = run_arm(r, "post", condition, pair_x_error, pair_d_error)
            response_rows.extend(response_rows_for(r, condition, pre, post))
            print("R5", condition, str(r.event_id), "done", flush=True)

    responses = pd.DataFrame(response_rows, columns=RESPONSE_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    finite_ok = (
        len(responses) == 776
        and np.isfinite(responses[["dT_model", "dAH_model", "dT_obs", "dAH_obs"]].to_numpy(float)).all()
        and bool(audit.finite.astype(bool).all())
        and np.isfinite(audit[["T_state", "AH_state"]].to_numpy(float)).all()
    )
    q_values = audit[["q_top", "q_main"]].to_numpy(float) if len(audit) else np.array([[np.nan]])
    q_ok = bool(np.isfinite(q_values).all() and (q_values >= 0).all())
    max_init = float(audit.init_err.max()) if len(audit) else float("inf")
    max_flux = float(audit.flux_identity_error.max()) if len(audit) else float("inf")
    max_pair_x = float(audit.paired_initial_state_error.max()) if len(audit) else float("inf")
    max_pair_d = float(audit.paired_forcing_error.max()) if len(audit) else float("inf")
    event_count_ok = (
        len(responses) == 776
        and responses.event_id.astype(str).nunique() == 97
        and responses.groupby(["event_id", "condition", "horizon_min"]).size().eq(1).all()
        and audit.groupby(["event_id", "condition", "arm", "horizon_min"]).size().eq(1).all()
    )
    checks.extend([
        {"check": "all_prescribed_fluxes_finite_nonnegative", "passed": q_ok,
         "detail": "min/max q across all arms: " + (str([float(np.nanmin(q_values)), float(np.nanmax(q_values))]) if len(q_values) else "unavailable")},
        {"check": "all_states_outputs_finite_and_initial_state_error", "passed": finite_ok and max_init <= 1e-8,
         "detail": json.dumps({"finite": bool(finite_ok), "max_init_error": max_init})},
        {"check": "flux_identity_all_arms_horizons", "passed": max_flux <= 1e-12,
         "detail": json.dumps({"audit_rows": len(audit), "max_error_m3_m2": max_flux, "tolerance": 1e-12})},
        {"check": "paired_initial_state_and_forcing_identical", "passed": max_pair_x == 0.0 and max_pair_d == 0.0,
         "detail": json.dumps({"max_initial_state_error": max_pair_x, "max_forcing_error": max_pair_d})},
        {"check": "response_row_count_97x4x2", "passed": len(responses) == 776 and event_count_ok,
         "detail": json.dumps({"rows": len(responses), "events": int(responses.event_id.astype(str).nunique()),
                               "unique_event_condition_horizon": bool(event_count_ok)})},
    ])
    summary.update({
        "events_completed": int(responses.event_id.astype(str).nunique()),
        "response_rows": len(responses), "runtime_audit_rows": len(audit),
        "all_states_outputs_finite": bool(finite_ok),
        "all_prescribed_fluxes_finite_nonnegative": bool(q_ok),
        "max_init_error": max_init, "max_flux_identity_error": max_flux,
        "max_paired_initial_state_error": max_pair_x,
        "max_paired_forcing_error": max_pair_d,
        "m1_model_gates_passed": all(c["passed"] for c in checks),
    })
    if not all(c["passed"] for c in checks):
        raise RuntimeError("one or more M1 runtime gates failed")


try:
    main()
except BaseException:
    failure_text = traceback.format_exc()
    print(failure_text, flush=True)
finally:
    write_outputs()

if failure_text is not None or not summary.get("m1_model_gates_passed", False):
    raise SystemExit("R5 M1 runtime gate failed")

