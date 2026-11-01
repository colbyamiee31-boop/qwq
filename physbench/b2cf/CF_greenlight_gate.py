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

CONDITIONS = ['REF0', 'FF0']
HORIZONS = [15, 30]
STEP_SECONDS = 900.0
INPUT_PATH = ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv"
EVENT_LIST_PATH = ROOT / "physbench/r5/R5_event_list_97_v2.csv"
LOCK_PATH = ROOT / "physbench/r5/R5_FORMAL_EXECUTION_LOCK.md"
PREPARE_GATE_PATH = ROOT / "physbench/r5/generated/R5_PREPARE_GATE_SUMMARY.json"
R4_REFERENCE_PATH = ROOT / "baseline_c8/evidence/R4/M1/ctifl_common_flux_responses.csv"
R4_AUDIT_PATH = ROOT / "baseline_c8/evidence/R4/M1/runtime_audit.csv"
R4_SUMMARY_PATH = ROOT / "baseline_c8/evidence/R4/M1/summary.json"
OUT = ROOT / "cf-output"
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_COLUMNS = [
    "event_id", "row_index", "event_date", "direction", "anchor_primary_3_6",
    "condition", "horizon_min", "dT_model", "dAH_model", "dT_obs", "dAH_obs",
    "q_pre_top", "q_post_top", "q_pre_main", "q_post_main", "hfac",
]
AUDIT_COLUMNS = [
    "event_id", "row_index", "event_date", "condition", "arm", "horizon_min",
    "q_top", "q_main", "q_top_hex", "q_main_hex", "flow_volume_integrated", "flow_volume_expected",
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
    q_field = float(q_field_fn(leeward / 100.0, windward / 100.0, wind,
                               float(r.event_Tair), float(r.event_Tout)))
    if condition in ("FF0-CF", "FFH-CF"):
        fold = CF_FOLDS[int(r.row_index)]
        params = CF_PARAMS[f"drop_fold_{fold}"]
        def phi(v):
            import math
            return [1/(1+math.exp(-params["slope"]*(v-c)))-1/(1+math.exp(params["slope"]*c)) for c in params["centres"]]
        import math
        vl=leeward/100.0; vw=windward/100.0
        flow=wind*(sum(a*x for a,x in zip(params["aL"],phi(vl)))+sum(a*x for a,x in zip(params["aW"],phi(vw))))+params["b"]*(vl+vw)*math.sqrt(max(float(r.event_Tair)-float(r.event_Tout),0))
        return flow, 0.0, params["heat_factor_s_T"] if condition == "FFH-CF" else 1.0
    if condition == "REF0":
        emp_flux = float(getattr(r, f"emp_flux_{suffix}_23_23"))
        return emp_flux, 0.0, 1.0
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
            "q_top_hex": float(qt).hex(), "q_main_hex": float(qm).hex(),
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
            return {str(k): finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, np.ndarray):
            return [finite_json(v) for v in value.tolist()]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.generic):
            return finite_json(value.item())
        return value
    safe_summary = finite_json(summary)
    (OUT / "R5_gate_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(safe_summary, indent=2, allow_nan=False), flush=True)


def run_gates():
    global q_field_fn, X_NATIVE, P_NATIVE, NX, NU, ND, P_HV, FCF
    E = pd.read_csv(ROOT / "cf-prepared/PRIMARY_97_R2_INPUT.csv")
    E["anchor_primary_3_6"] = bool_values(E.anchor_primary_3_6, "anchor")
    subset = E[E.anchor_primary_3_6].copy()
    if len(E) != 97 or len(subset) != 40 or subset.event_id.nunique() != 40: raise RuntimeError("input 97/40 mismatch")
    baseline = pd.read_csv(ROOT / "cf-baseline/evidence/R5/FORMAL/R5_event_responses.csv")
    global CF_FOLDS, CF_PARAMS
    fold_data = pd.read_csv(ROOT / "physbench/b2cf/CF_event_folds.csv")
    CF_FOLDS = dict(zip(fold_data.row_index.astype(int),fold_data.fold.astype(int)))
    CF_PARAMS = json.loads((ROOT / "physbench/b2cf/CF_fold_params.json").read_text())
    P = json.loads((ROOT / "physbench/b2cf/CF_full_reconstructed.json").read_text())
    import math
    def reconstructed(vL, vW, U, Tin, Tout):
        def phi(v): return [1/(1+math.exp(-P["slope"]*(v-c)))-1/(1+math.exp(P["slope"]*c)) for c in P["centres"]]
        gl=sum(a*x for a,x in zip(P["aL"],phi(vL)));gw=sum(a*x for a,x in zip(P["aW"],phi(vW)))
        return U*(gl+gw)+P["b"]*(vL+vW)*math.sqrt(max(Tin-Tout,0))
    q_field_fn = reconstructed
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
    for condition in CONDITIONS:
        start = len(response_rows)
        for _, r in subset.iterrows():
            pair_x_error=float(np.max(np.abs(initial_state(r)-initial_state(r))))
            pair_d_error=float(np.max(np.abs(build_weather(r)-build_weather(r))))
            pre=run_arm(r,"pre",condition,pair_x_error,pair_d_error)
            post=run_arm(r,"post",condition,pair_x_error,pair_d_error)
            response_rows.extend(response_rows_for(r,condition,pre,post))
            print(condition,str(r.event_id),flush=True)
        got=pd.DataFrame(response_rows[start:],columns=RESPONSE_COLUMNS)
        ref=baseline[(baseline.condition==condition)&baseline.event_id.isin(subset.event_id)].copy()
        joined=got.merge(ref,on=["event_id","horizon_min"],suffixes=("_actual","_ref"),validate="one_to_one")
        if len(joined)!=80: raise RuntimeError("baseline rows mismatch")
        rows=[];stats=[]
        for h in HORIZONS:
            for tracer,col in [("T","dT_model"),("AH","dAH_model")]:
                t=joined[joined.horizon_min==h];reference=t[col+"_ref"].to_numpy(float);actual=t[col+"_actual"].to_numpy(float);S=float(np.median(abs(reference)));diff=abs(actual-reference);floor=(abs(reference)<.1*S)&(abs(actual)<.1*S);passmask=diff<=1e-6*np.maximum(np.maximum(abs(reference),abs(actual)),.1*S);valid=abs(reference)>=.1*S
                for eid,a,b,d,fl,ok in zip(t.event_id,actual,reference,diff,floor,passmask):rows.append(dict(event_id=eid,condition=condition,horizon_min=h,tracer=tracer,actual=a,reference=b,abs_diff=d,S=S,floor_used=bool(fl),passed=bool(ok)))
                stats.append(dict(condition=condition,horizon_min=h,tracer=tracer,n=len(t),S=S,max_abs_diff=float(diff.max()),max_rel_diff=float((diff[valid]/abs(reference[valid])).max()) if valid.any() else None,floor_pass_count=int((floor&passmask).sum()),failures=int((~passmask).sum())))
        pd.DataFrame(rows).to_csv(OUT / ("CF_"+condition+"_comparisons.csv"),index=False)
        (OUT / ("CF_"+condition+"_summary.json")).write_text(json.dumps(stats,indent=2))
        if any(x["failures"] for x in stats): raise RuntimeError(condition+" gate failed; stop")
    pd.DataFrame(response_rows,columns=RESPONSE_COLUMNS).to_csv(OUT/"CF_event_responses.csv",index=False)
    pd.DataFrame(audit_rows,columns=AUDIT_COLUMNS).to_csv(OUT/"CF_runtime_audit.csv",index=False)
if __name__ == "__main__": run_gates()

