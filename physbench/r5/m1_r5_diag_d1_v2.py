import hashlib
import inspect
import json
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "physbench/r5"))
from common import (
    H_TARGET,
    VENT_PROJECTED_RATIO,
    ppm_to_mg_m3,
    mg_m3_to_ppm,
    ah_from_t_vp,
    q_common_pct,
    harmonise_m1_hv,
)

P97_PATH = ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv"
OUT_DIR = ROOT / "evidence/R5/DIAG_D1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
C8_SCRIPT_PATH = ROOT / "physbench/r5/m1_c8_reproduction.py"
C8_RESPONSES_PATH = ROOT / "evidence/R5/C8/c8_m1_responses.csv"

# Potential paths for R4 CF0 and R5 run 39387663913 references
R4_CF0_PATH = ROOT / "baseline_r4/evidence/R4/M1/ctifl_common_flux_responses.csv"
if not R4_CF0_PATH.exists():
    # Fallback search
    candidates = list(ROOT.glob("**/ctifl_common_flux_responses.csv"))
    if candidates:
        R4_CF0_PATH = candidates[0]

R5_V2_COMP_PATH = ROOT / "baseline_r5_formal_m1/evidence/R5/FORMAL/R5_embedded_C8_event_comparison.csv"
if not R5_V2_COMP_PATH.exists():
    candidates = list(ROOT.glob("**/R5_embedded_C8_event_comparison.csv"))
    if candidates:
        R5_V2_COMP_PATH = candidates[0]

STEP_SECONDS = 900.0

def s(v: float) -> float:
    return float(format(float(v), ".12g"))

def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value

# >>> COPY_BLOCK: c8_build_weather
def build_weather(r):
    d=np.zeros(10,dtype=float)
    d[0]=float(r.event_Iglob)
    d[1]=float(r.event_Tout)
    d[2]=float(r.out_vp_pa)
    d[3]=float(ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed)))
    d[4]=float(r.event_Windsp)
    d[5]=float(r.sky_temperature_c_proxy)
    d[6]=float(r.soil_boundary_temperature_c_fixed)
    d[7]=float(r.event_Iglob)*900.0/1e6
    d[8]=1.0 if float(r.event_Iglob)>0 else 0.0
    d[9]=d[8]
    return d
# <<< COPY_BLOCK: c8_build_weather

# >>> COPY_BLOCK: c8_full_action
def full_action(v):
    return np.array([0,0,0,float(v),0,0],dtype=float)
# <<< COPY_BLOCK: c8_full_action

# >>> COPY_BLOCK: c8_env
env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
X_NATIVE=e.x.copy()
P_NATIVE=np.asarray(e.p,dtype=float).copy()
NX=e.nx; NU=e.nu; ND=e.nd
env.close()

P_HV=harmonise_m1_hv(P_NATIVE)
assert abs(P_HV[49]-H_TARGET)<=1e-12
assert abs(P_HV[55]/P_HV[46]-VENT_PROJECTED_RATIO)<=1e-12
# <<< COPY_BLOCK: c8_env

# >>> COPY_BLOCK: c8_initial_state
def initial_state(r):
    xx=X_NATIVE.copy()
    xx[2]=float(r.event_Tair)
    xx[15]=float(r.in_vp_pa)
    xx[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    xx[4]=float(getattr(r,'canopy_temperature_c_closure',r.event_Tair))
    return xx
# <<< COPY_BLOCK: c8_initial_state

# >>> COPY_BLOCK: c8_symbolic
x=ca.SX.sym('x',NX)
u=ca.SX.sym('u',NU)
d=ca.SX.sym('d',ND)
p=ca.SX.sym('p',len(P_NATIVE))
qtop=ca.SX.sym('qtop')
qmain=ca.SX.sym('qmain')
a=aux_states.update(x,u,d,p)
dx=ODE(x,u,d,p)

def sensible(hec,t1,t2):
    return ca.fabs(hec)*(t1-t2)
def air_mv(f,vp1,vp2,t1,t2):
    return 0.002165*ca.fabs(f)*(vp1/(t1+273.15)-vp2/(t2+273.15))
def air_mc(f,c1,c2):
    return ca.fabs(f)*(c1-c2)

native_top=a[136]
native_main=a[137]+a[145]

# CO2 external exchange replacement
dx[0] += (air_mc(native_main,x[0],d[3])-air_mc(qmain,x[0],d[3]))/p[122]
dx[1] += (air_mc(native_top,x[1],d[3])-air_mc(qtop,x[1],d[3]))/p[123]
# sensible-heat external exchange replacement
cp_rho=p[111]*p[23]
dx[2] += (sensible(cp_rho*native_main,x[2],d[1])-sensible(cp_rho*qmain,x[2],d[1]))/p[112]
dx[3] += (sensible(cp_rho*native_top,x[3],d[1])-sensible(cp_rho*qtop,x[3],d[1]))/p[120]
# vapour external exchange replacement
dx[15] += (air_mv(native_main,x[15],d[2],x[2],d[1])-air_mv(qmain,x[15],d[2],x[2],d[1]))/a[35]
dx[16] += (air_mv(native_top,x[16],d[2],x[3],d[1])-air_mv(qtop,x[16],d[2],x[3],d[1]))/a[36]

FCF_C8=ca.integrator(
    'R4_M1_CF900','cvodes',
    {'x':x,'u':u,'p':ca.vertcat(d,p,qtop,qmain),'ode':dx},
    0.0,900.0,
    {'abstol':1e-5,'reltol':1e-5,'max_num_steps':90000}
)
# <<< COPY_BLOCK: c8_symbolic

# >>> COPY_BLOCK: r5_symbolic
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
FCF_R5 = ca.integrator(
    "R5_M1_CF900", "cvodes",
    {"x": x, "u": u, "p": ca.vertcat(d, p, qtop, qmain, hfac),
     "ode": dx, "quad": qtop + qmain},
    0.0, STEP_SECONDS,
    {"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 90000})
# <<< COPY_BLOCK: r5_symbolic

def execute_r5_step(xx, uu, dd, qt, qm, hfac):
    # >>> COPY_BLOCK: r5_pvec_call
    pvec = np.concatenate([dd, P_HV, [qt, qm, hfac]])
    rr = FCF_R5(x0=xx, u=uu, p=pvec)
    x900 = np.asarray(rr["xf"].full()).reshape(-1)
    dv900 = float(np.asarray(rr["qf"].full()).reshape(-1)[0])
    # <<< COPY_BLOCK: r5_pvec_call
    stats = FCF_R5.stats()
    return x900, dv900, stats

def execute_c8_step(xx, uu, dd, qt, qm):
    rr = FCF_C8(x0=xx, u=uu, p=np.concatenate([dd, P_HV, [qt, qm]]))
    stats = FCF_C8.stats()
    xf = np.asarray(rr["xf"].full()).reshape(-1)
    return xf, stats

def extract_stat_scalars(stats_dict):
    keys = ["nsteps", "nfevals", "netfails", "nlinsetups", "hlast", "hcur"]
    res = {}
    for k in keys:
        v = stats_dict.get(k)
        if isinstance(v, (float, np.floating)):
            res[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, (int, np.integer)):
            res[k] = int(v)
        else:
            res[k] = None
    res["last_step_size"] = res["hlast"] if res.get("hlast") is not None else res.get("hcur")
    return res

def run_preflight_checks():
    print("=" * 70)
    print("PRE-FLIGHT CHECKS FOR STAGE D1-V2")
    print("=" * 70)
    preflight = {}

    # Check 1: Signatures
    sig_ode = inspect.signature(ODE)
    params_ode = list(sig_ode.parameters.keys())
    assert params_ode == ["x", "u", "d", "p"], f"ODE signature mismatch: {params_ode}"
    sig_aux = inspect.signature(aux_states.update)
    params_aux = list(sig_aux.parameters.keys())
    assert params_aux == ["x", "u", "d", "p"], f"aux_states signature mismatch: {params_aux}"
    preflight["signatures"] = {"ode": str(sig_ode), "aux": str(sig_aux), "passed": True}
    print("[PASS] Item 1: ODE and aux_states signatures match exactly (x, u, d, p).")

    # Check 2: Static copy gate
    import check_d1_v2_copy_gate
    copy_hashes = check_d1_v2_copy_gate.run_copy_gate()
    preflight["copy_gate"] = {"hashes": copy_hashes, "passed": True}
    print("[PASS] Item 2: Static copy gate verified 100% byte-identical against frozen sources.")

    # Check 3: Required columns
    p97_cols = [
        "event_id", "row_index", "anchor_primary_3_6", "u_pre", "u_post",
        "leeward_pre_pct", "leeward_post_pct", "windward_pre_pct", "windward_post_pct",
        "event_Windsp", "event_Tair", "in_vp_pa", "CO2_pre_ppm", "canopy_temperature_c_closure",
        "event_Iglob", "event_Tout", "out_vp_pa", "outdoor_co2_ppm_fixed",
        "sky_temperature_c_proxy", "soil_boundary_temperature_c_fixed",
        "emp_flux_pre_23_23", "emp_flux_post_23_23"
    ]
    df_p97 = pd.read_csv(P97_PATH)
    missing_p97 = [c for c in p97_cols if c not in df_p97.columns]
    assert not missing_p97, f"Missing columns in PRIMARY_97_R2_INPUT: {missing_p97}"

    r4_cols = [
        "stage", "cohort", "horizon_min", "model", "event_id",
        "model_T_delta", "model_AH_delta", "pre_T", "post_T", "pre_AH", "post_AH"
    ]
    assert R4_CF0_PATH.exists(), f"Missing R4 CF0 baseline at {R4_CF0_PATH}"
    df_r4 = pd.read_csv(R4_CF0_PATH)
    missing_r4 = [c for c in r4_cols if c not in df_r4.columns]
    assert not missing_r4, f"Missing columns in R4 CF0: {missing_r4}"
    preflight["required_columns"] = {"missing_p97": missing_p97, "missing_r4": missing_r4, "passed": True}
    print("[PASS] Item 3: All required input columns present in P97 and R4 CF0.")

    # Check 4: Integrator construct & dummy call
    c8_in, c8_out = FCF_C8.n_in(), FCF_C8.n_out()
    r5_in, r5_out = FCF_R5.n_in(), FCF_R5.n_out()
    print(f"FCF_C8 n_in: {c8_in}, n_out: {c8_out}")
    print(f"FCF_R5 n_in: {r5_in}, n_out: {r5_out}")
    # Dummy call FCF_C8 with baseline vectors to check that call raises no exception
    dummy_p = np.concatenate([np.zeros(ND, dtype=float), P_HV, [0.0, 0.0]])
    dummy_x = X_NATIVE.copy()
    dummy_u = np.zeros(NU, dtype=float)
    _ = FCF_C8(x0=dummy_x, u=dummy_u, p=dummy_p)
    preflight["integrators"] = {
        "c8_n_in": c8_in, "c8_n_out": c8_out,
        "r5_n_in": r5_in, "r5_n_out": r5_out,
        "dummy_call_passed": True
    }
    print("[PASS] Item 4: Both integrators constructed and dummy call executed without error.")

    # Check 5: Input SHA256
    p97_sha = hashlib.sha256(P97_PATH.read_bytes()).hexdigest()
    expected_p97_sha = "0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b"
    assert p97_sha == expected_p97_sha, f"P97 SHA mismatch: {p97_sha} vs {expected_p97_sha}"
    preflight["input_sha256"] = {"sha256": p97_sha, "passed": True}
    print(f"[PASS] Item 5: PRIMARY_97_R2_INPUT.csv SHA256 verified ({p97_sha}).")

    return preflight, df_p97, df_r4

def main():
    preflight, df_p97, df_r4 = run_preflight_checks()

    # Filter 40 events
    subset = df_p97[df_p97.anchor_primary_3_6 == True].sort_values("row_index").copy()
    subset_ids = list(subset.event_id)
    assert len(subset) == 40, f"Expected 40 subset events, got {len(subset)}"

    r4_cf0_15 = df_r4[
        (df_r4.stage == "CF0") &
        (df_r4.horizon_min == 15) &
        (df_r4.event_id.isin(subset_ids))
    ].sort_values("event_id").copy()
    assert len(r4_cf0_15) == 40, f"Expected 40 R4 CF0 events, got {len(r4_cf0_15)}"

    # Check existence of R5 run 39387663913 comparison file
    df_r5_v2 = None
    if R5_V2_COMP_PATH.exists():
        df_r5_v2 = pd.read_csv(R5_V2_COMP_PATH)
        assert len(df_r5_v2) == 40, f"Expected 40 events in R5 v2 comparison file, got {len(df_r5_v2)}"

    summary = {
        "stage": "D1-v2",
        "preflight": preflight,
        "v0": {},
        "v1": {},
        "v2": {},
        "v3": {},
        "status": "RUNNING"
    }

    comparison_records = {eid: {"event_id": eid} for eid in subset_ids}
    for _, r4_row in r4_cf0_15.iterrows():
        eid = str(r4_row.event_id)
        comparison_records[eid]["dT_R4"] = float(r4_row.model_T_delta)
        comparison_records[eid]["dAH_R4"] = float(r4_row.model_AH_delta)

    flux_records = []
    stat_records = []
    stat_full_json = {}

    def write_all_outputs():
        print("\nWriting all D1 diagnostic output artifacts...")
        comp_df = pd.DataFrame(list(comparison_records.values()))
        comp_df.to_csv(OUT_DIR / "variant_comparison_table.csv", index=False)
        pd.DataFrame(flux_records).to_csv(OUT_DIR / "full_precision_flux_table.csv", index=False)
        pd.DataFrame(stat_records).to_csv(OUT_DIR / "integrator_statistics.csv", index=False)
        (OUT_DIR / "integrator_statistics_full.json").write_text(
            json.dumps(finite_json(stat_full_json), indent=2) + "\n", encoding="utf-8"
        )
        (OUT_DIR / "diag_d1_summary.json").write_text(
            json.dumps(finite_json(summary), indent=2) + "\n", encoding="utf-8"
        )
        print("Output artifacts written successfully to:", OUT_DIR)

    try:
        # =====================================================================
        # VARIANT V0: Unmodified C8 subprocess
        # =====================================================================
        print("\n" + "=" * 70)
        print("EXECUTING VARIANT V0 (UNMODIFIED C8 SUBPROCESS)")
        print("=" * 70)

        cmd = [sys.executable, str(C8_SCRIPT_PATH)]
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            print("C8 subprocess failed with error:")
            print(res.stderr)
            raise RuntimeError(f"C8 subprocess exited with code {res.returncode}")

        if not C8_RESPONSES_PATH.exists():
            raise FileNotFoundError(f"C8 responses not found at {C8_RESPONSES_PATH}")

        c8_resp = pd.read_csv(C8_RESPONSES_PATH)
        c8_subset = c8_resp[
            (c8_resp.stage == "CF0") &
            (c8_resp.horizon_min == 15) &
            (c8_resp.event_id.isin(subset_ids))
        ].sort_values("event_id").copy()
        assert len(c8_subset) == 40, f"Expected 40 C8 subset events, got {len(c8_subset)}"

        max_v0_diff_T = 0.0
        max_v0_diff_AH = 0.0
        v0_passed = True

        for _, c8_row in c8_subset.iterrows():
            eid = str(c8_row.event_id)
            r4_row = r4_cf0_15[r4_cf0_15.event_id == eid].iloc[0]

            dt_v0_raw = float(c8_row.model_T_delta)
            dah_v0_raw = float(c8_row.model_AH_delta)
            dt_r4 = float(r4_row.model_T_delta)
            dah_r4 = float(r4_row.model_AH_delta)

            s_dt_v0 = s(dt_v0_raw)
            s_dah_v0 = s(dah_v0_raw)

            diff_T = abs(s_dt_v0 - dt_r4)
            diff_AH = abs(s_dah_v0 - dah_r4)

            comparison_records[eid]["dT_V0_raw"] = repr(dt_v0_raw)
            comparison_records[eid]["dAH_V0_raw"] = repr(dah_v0_raw)
            comparison_records[eid]["s_dT_V0"] = s_dt_v0
            comparison_records[eid]["s_dAH_V0"] = s_dah_v0
            comparison_records[eid]["diff_T_V0"] = diff_T
            comparison_records[eid]["diff_AH_V0"] = diff_AH

            if diff_T > max_v0_diff_T:
                max_v0_diff_T = diff_T
            if diff_AH > max_v0_diff_AH:
                max_v0_diff_AH = diff_AH

            if diff_T != 0.0 or diff_AH != 0.0:
                v0_passed = False

        summary["v0"] = {
            "max_diff_T_storage": max_v0_diff_T,
            "max_diff_AH_storage": max_v0_diff_AH,
            "passed": v0_passed
        }
        print(f"V0 max diff T: {max_v0_diff_T}, max diff AH: {max_v0_diff_AH}")
        if not v0_passed:
            summary["status"] = "FAIL_V0_STOP_CONDITION"
            raise RuntimeError(
                f"STOP CONDITION: V0 failed storage-precision equality with frozen R4 CF0 "
                f"(max_diff_T={max_v0_diff_T}, max_diff_AH={max_v0_diff_AH}). Halting fail-closed."
            )
        print("[PASS] Variant V0 exactly matched frozen R4 CF0 at storage precision (0.0).")

        # =====================================================================
        # VARIANTS V1, V2, V3
        # =====================================================================
        variants = ["V1", "V2", "V3"]
        stat_full_json["P-0158"] = {}
        stat_full_json["P-0165"] = {}

        for var in variants:
            print("\n" + "=" * 70)
            print(f"EXECUTING VARIANT {var}")
            print("=" * 70)

            max_diff_T = 0.0
            max_diff_AH = 0.0
            diff_T_exceed_1e_12 = []
            diff_T_exceed_1e_9 = []
            diff_T_exceed_1e_6 = []

            for _, r in subset.iterrows():
                eid = str(r.event_id)
                r4_row = r4_cf0_15[r4_cf0_15.event_id == eid].iloc[0]
                dt_r4 = float(r4_row.model_T_delta)
                dah_r4 = float(r4_row.model_AH_delta)

                # Pre arm
                xx_pre = initial_state(r)
                dd_pre = build_weather(r)
                uu_pre = full_action(float(r.u_pre))
                emp_flux_pre = float(r.emp_flux_pre_23_23)

                if var in ["V1", "V3"]:
                    qt_pre = float(q_common_pct(r.leeward_pre_pct, r.windward_pre_pct, r.event_Windsp))
                else:  # V2
                    qt_pre = emp_flux_pre
                qm_pre = 0.0

                flux_records.append({
                    "variant": var, "event_id": eid, "arm": "pre",
                    "q_top_repr": repr(qt_pre), "q_top_hex": float(qt_pre).hex(),
                    "q_main_repr": repr(qm_pre), "q_main_hex": float(qm_pre).hex(),
                    "emp_flux_repr": repr(emp_flux_pre), "emp_flux_hex": float(emp_flux_pre).hex(),
                    "bit_identical": (float(qt_pre).hex() == float(emp_flux_pre).hex())
                })

                if var == "V1":
                    xf_pre, stats_pre = execute_c8_step(xx_pre, uu_pre, dd_pre, qt_pre, qm_pre)
                else:  # V2, V3
                    xf_pre, dv_pre, stats_pre = execute_r5_step(xx_pre, uu_pre, dd_pre, qt_pre, qm_pre, 1.0)

                t_pre = float(xf_pre[2])
                ah_pre = float(ah_from_t_vp(t_pre, float(xf_pre[15])))

                scal_pre = extract_stat_scalars(stats_pre)
                stat_records.append({
                    "variant": var, "event_id": eid, "arm": "pre",
                    **scal_pre
                })

                # Post arm
                xx_post = initial_state(r)
                dd_post = build_weather(r)
                uu_post = full_action(float(r.u_post))
                emp_flux_post = float(r.emp_flux_post_23_23)

                if var in ["V1", "V3"]:
                    qt_post = float(q_common_pct(r.leeward_post_pct, r.windward_post_pct, r.event_Windsp))
                else:  # V2
                    qt_post = emp_flux_post
                qm_post = 0.0

                flux_records.append({
                    "variant": var, "event_id": eid, "arm": "post",
                    "q_top_repr": repr(qt_post), "q_top_hex": float(qt_post).hex(),
                    "q_main_repr": repr(qm_post), "q_main_hex": float(qm_post).hex(),
                    "emp_flux_repr": repr(emp_flux_post), "emp_flux_hex": float(emp_flux_post).hex(),
                    "bit_identical": (float(qt_post).hex() == float(emp_flux_post).hex())
                })

                if var == "V1":
                    xf_post, stats_post = execute_c8_step(xx_post, uu_post, dd_post, qt_post, qm_post)
                else:  # V2, V3
                    xf_post, dv_post, stats_post = execute_r5_step(xx_post, uu_post, dd_post, qt_post, qm_post, 1.0)

                t_post = float(xf_post[2])
                ah_post = float(ah_from_t_vp(t_post, float(xf_post[15])))

                scal_post = extract_stat_scalars(stats_post)
                stat_records.append({
                    "variant": var, "event_id": eid, "arm": "post",
                    **scal_post
                })

                # Special full stats recording for P-0158 and P-0165
                for target_key in ["P-0158", "P-0165"]:
                    if target_key in eid:
                        stat_full_json[target_key][f"{var}_pre"] = stats_pre
                        stat_full_json[target_key][f"{var}_post"] = stats_post

                dt_raw = t_post - t_pre
                dah_raw = ah_post - ah_pre
                s_dt = s(dt_raw)
                s_dah = s(dah_raw)

                diff_T = abs(s_dt - dt_r4)
                diff_AH = abs(s_dah - dah_r4)

                comparison_records[eid][f"dT_{var}_raw"] = repr(dt_raw)
                comparison_records[eid][f"dAH_{var}_raw"] = repr(dah_raw)
                comparison_records[eid][f"s_dT_{var}"] = s_dt
                comparison_records[eid][f"s_dAH_{var}"] = s_dah
                comparison_records[eid][f"diff_T_{var}"] = diff_T
                comparison_records[eid][f"diff_AH_{var}"] = diff_AH

                if diff_T > max_diff_T:
                    max_diff_T = diff_T
                if diff_AH > max_diff_AH:
                    max_diff_AH = diff_AH

                if diff_T > 1e-12:
                    diff_T_exceed_1e_12.append(eid)
                if diff_T > 1e-9:
                    diff_T_exceed_1e_9.append(eid)
                if diff_T > 1e-6:
                    diff_T_exceed_1e_6.append(eid)

            summary[var.lower()] = {
                "max_diff_T_storage": max_diff_T,
                "max_diff_AH_storage": max_diff_AH,
                "count_exceed_1e_12": len(diff_T_exceed_1e_12),
                "count_exceed_1e_9": len(diff_T_exceed_1e_9),
                "count_exceed_1e_6": len(diff_T_exceed_1e_6),
                "events_exceed_1e_12": diff_T_exceed_1e_12,
                "events_exceed_1e_9": diff_T_exceed_1e_9,
                "events_exceed_1e_6": diff_T_exceed_1e_6,
            }
            print(f"{var} max diff T vs R4: {max_diff_T}, max diff AH vs R4: {max_diff_AH}")
            print(f"{var} events exceeding 1e-12: {len(diff_T_exceed_1e_12)}")
            print(f"{var} events exceeding 1e-9:  {len(diff_T_exceed_1e_9)}")
            print(f"{var} events exceeding 1e-6:  {len(diff_T_exceed_1e_6)}")

            # Check V3 reproducibility against run 39387663913
            if var == "V3" and df_r5_v2 is not None:
                v3_reproduced = True
                max_v3_ref_diff_T = 0.0
                for _, r5_ref_row in df_r5_v2.iterrows():
                    veid = str(r5_ref_row.event_id)
                    cur_s_dt = comparison_records[veid]["s_dT_V3"]
                    cur_s_dah = comparison_records[veid]["s_dAH_V3"]
                    ref_dt = float(r5_ref_row.dT_model)
                    ref_dah = float(r5_ref_row.dAH_model)

                    diff_ref_T = abs(cur_s_dt - ref_dt)
                    diff_ref_AH = abs(cur_s_dah - ref_dah)
                    if diff_ref_T > max_v3_ref_diff_T:
                        max_v3_ref_diff_T = diff_ref_T

                    if diff_ref_T != 0.0 or diff_ref_AH != 0.0:
                        v3_reproduced = False

                summary["v3"]["reproduced_run_39387663913"] = v3_reproduced
                summary["v3"]["max_ref_diff_T"] = max_v3_ref_diff_T

                p0158_diff = comparison_records.get("CTIFL-E5-P-0158", {}).get("diff_T_V3", 0.0)
                summary["v3"]["p0158_diff_T"] = p0158_diff
                summary["v3"]["p0158_target_expected"] = 1.2858481068043614e-06
                print(f"V3 reproduced run 39387663913: {v3_reproduced} (max ref diff T: {max_v3_ref_diff_T})")
                print(f"V3 P-0158 diff T: {p0158_diff} (expected ~1.285848e-6)")
                if not v3_reproduced:
                    summary["status"] = "FAIL_V3_REPRODUCIBILITY"
                    raise RuntimeError("STOP CONDITION: V3 failed to reproduce formal run 39387663913 results.")

        summary["status"] = "PASS"
        print("\n" + "=" * 70)
        print("STAGE D1-V2 DIAGNOSTIC EXPERIMENT COMPLETED SUCCESSFULLY (PASS)")
        print("=" * 70)

    except Exception as e:
        print(f"\nEXCEPTION DURING RUN: {e}")
        traceback.print_exc()
        if summary["status"] == "RUNNING":
            summary["status"] = "ERROR"
            summary["error"] = str(e)
        raise
    finally:
        write_all_outputs()

if __name__ == "__main__":
    main()
