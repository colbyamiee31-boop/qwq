import json, sys, os, hashlib, traceback
from pathlib import Path
import numpy as np, pandas as pd
import casadi as ca
import gymnasium as gym
import gl_gym
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/R5/DIAG_D1"
OUT.mkdir(parents=True, exist_ok=True)

H_TARGET = 7.375
VENT_PROJECTED_RATIO = 0.17474999631859142
CTIFL_FLOOR_AREA = 1036.8
VENT_L = 4.05
VENT_H = 1.40
BANK_L = 23
BANK_W = 23
R = 8.3144598
K = 273.15
MCO2 = 44.01e-3
P = 101325.0

def ppm_to_mg_m3(t, ppm):
    return P * np.asarray(ppm, dtype=float) * MCO2 / (R * (np.asarray(t, dtype=float) + K))

def mg_m3_to_ppm(t, mg):
    return R * (np.asarray(t, dtype=float) + K) * np.asarray(mg, dtype=float) / (P * MCO2)

def sat_vp_pa(t):
    t = np.asarray(t, dtype=float)
    return 610.78 * np.exp(17.2694 * t / (t + 238.3))

def ah_from_t_vp(t, vp):
    return 216.7 * (np.asarray(vp, dtype=float) / 100.0) / (np.asarray(t, dtype=float) + 273.15)

def gl_fun(delta_deg):
    d = np.asarray(delta_deg, dtype=float)
    return 2.46e-2 * (1.0 - np.exp(-d / 14.5))

def gw_fun(delta_deg):
    d = np.asarray(delta_deg, dtype=float)
    return -1.89e-5 * d * d + 2.23e-3 * d

def q_common_pct(leeward_pct, windward_pct, wind_speed):
    vl = np.asarray(leeward_pct, dtype=float) / 100.0
    vw = np.asarray(windward_pct, dtype=float) / 100.0
    u = np.asarray(wind_speed, dtype=float)
    dl = 44.0 * vl
    dw = 44.0 * vw
    q = u * VENT_L * VENT_H / CTIFL_FLOOR_AREA * (BANK_L * gl_fun(dl) + BANK_W * gw_fun(dw))
    return q

def harmonise_m1_hv(p):
    q = np.asarray(p, dtype=float).copy()
    q[48] = 6.97
    q[49] = H_TARGET
    q[112] = q[48] * q[111] * q[23]
    q[120] = (q[49] - q[48]) * q[111] * q[23]
    q[122] = q[48]
    q[123] = q[49] - q[48]
    q[55] = VENT_PROJECTED_RATIO * q[46]
    return q

def build_weather(r):
    d = np.zeros(10, dtype=float)
    d[0] = float(r.event_Iglob)
    d[1] = float(r.event_Tout)
    d[2] = float(r.out_vp_pa)
    d[3] = float(ppm_to_mg_m3(float(r.event_Tout), float(r.outdoor_co2_ppm_fixed)))
    d[4] = float(r.event_Windsp)
    d[5] = float(r.sky_temperature_c_proxy)
    d[6] = float(r.soil_boundary_temperature_c_fixed)
    d[7] = float(r.event_Iglob) * 900.0 / 1e6
    d[8] = 1.0 if float(r.event_Iglob) > 0 else 0.0
    d[9] = d[8]
    return d

def full_action(v):
    return np.array([0, 0, 0, float(v), 0, 0], dtype=float)

def find_file(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(f"None of candidates exist: {candidates}")

def main():
    print("=== PhysBench Stage D1 Diagnostic Run Starting ===")
    
    # Locate inputs
    p_inp = find_file([
        ROOT / "physbench/r5/generated/PRIMARY_97_R2_INPUT.csv",
        ROOT / "physbench/r5/PRIMARY_97_R2_INPUT.csv",
        ROOT / "evidence/r4-prepared-inputs/PRIMARY_97_R2_INPUT.csv",
    ])
    p_r4_resp = find_file([
        ROOT / "baseline_r4/evidence/R4/M1/ctifl_common_flux_responses.csv",
        ROOT / "baseline_c8/evidence/R4/M1/ctifl_common_flux_responses.csv",
        ROOT / "evidence/R4/M1/ctifl_common_flux_responses.csv",
        ROOT / "frozen_artifacts/r4-m1/evidence/R4/M1/ctifl_common_flux_responses.csv",
    ])
    
    print(f"Using primary input: {p_inp}")
    print(f"Using frozen R4 reference: {p_r4_resp}")
    
    # Verify primary input hash
    inp_bytes = p_inp.read_bytes()
    inp_hash = hashlib.sha256(inp_bytes).hexdigest()
    assert inp_hash == "0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b", f"Input hash mismatch: {inp_hash}"
    print("Primary input SHA256 verified.")

    inp_df = pd.read_csv(p_inp)
    r4_df = pd.read_csv(p_r4_resp)
    r4_cf0 = r4_df[(r4_df["stage"] == "CF0") & (r4_df["cohort"] == "primary") & (r4_df["horizon_min"] == 15)].copy()
    
    # Filter 40 events
    def as_bool(s):
        if pd.api.types.is_bool_dtype(s): return s.astype(bool)
        return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    
    subset_mask = as_bool(inp_df["anchor_primary_3_6"])
    events_40 = inp_df[subset_mask].copy().sort_values("row_index").reset_index(drop=True)
    assert len(events_40) == 40, f"Expected 40 subset events, got {len(events_40)}"
    print(f"Verified 40 subset events for D1 diagnosis.")

    # GreenLight Environment & ODE setup
    env = gym.make("gl_gym/GreenLightTomato-v0",
                   controlled_inputs=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"],
                   normalize_actions=False, parameter_provider="fixed")
    env.reset(seed=3407, options={"scenario": {"location": "Amsterdam", "growth_year": 2010, "start_day": 244}})
    e = env.unwrapped
    X_NATIVE = e.x.copy()
    P_NATIVE = np.asarray(e.p, dtype=float).copy()
    NX = e.nx; NU = e.nu; ND = e.nd
    env.close()

    P_HV = harmonise_m1_hv(P_NATIVE)
    assert abs(P_HV[49] - H_TARGET) <= 1e-12
    assert abs(P_HV[55] / P_HV[46] - VENT_PROJECTED_RATIO) <= 1e-12

    def initial_state(r):
        xx = X_NATIVE.copy()
        xx[2] = float(r.event_Tair)
        xx[3] = float(r.event_Tair)
        vp = float(r.in_vp_pa)
        xx[15] = float(ah_from_t_vp(xx[2], vp))
        xx[16] = float(ah_from_t_vp(xx[3], vp))
        xx[0] = float(ppm_to_mg_m3(xx[2], float(r.event_CO2air)))
        xx[1] = float(ppm_to_mg_m3(xx[3], float(r.event_CO2air)))
        return xx

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    d = ca.SX.sym("d", ND)
    p = ca.SX.sym("p", len(P_NATIVE))
    qtop = ca.SX.sym("qtop")
    qmain = ca.SX.sym("qmain")
    hfac = ca.SX.sym("hfac")

    a = aux_states.update(x, u, d, p)
    dx_base = ODE(x, u, d, p, a)
    native_main = a[137] + a[145]
    native_top = a[138] + a[146]
    air_mc = lambda q, c_in, c_out: q * (c_in - c_out)
    cp_rho = p[111] * p[23]
    sensible = lambda cp_q, t_in, t_out: cp_q * (t_in - t_out)
    air_mv = lambda q, vp_in, vp_out, t_in, t_out: q * (ah_from_t_vp(t_in, vp_in) - ah_from_t_vp(t_out, vp_out))

    # Integrator C8 (used for V0 and V1): no quad, no hfac
    dx_c8 = dx_base.copy()
    dx_c8[0] += (air_mc(native_main, x[0], d[3]) - air_mc(qmain, x[0], d[3])) / p[122]
    dx_c8[1] += (air_mc(native_top, x[1], d[3]) - air_mc(qtop, x[1], d[3])) / p[123]
    dx_c8[2] += (sensible(cp_rho * native_main, x[2], d[1]) - sensible(cp_rho * qmain, x[2], d[1])) / p[112]
    dx_c8[3] += (sensible(cp_rho * native_top, x[3], d[1]) - sensible(cp_rho * qtop, x[3], d[1])) / p[120]
    dx_c8[15] += (air_mv(native_main, x[15], d[2], x[2], d[1]) - air_mv(qmain, x[15], d[2], x[2], d[1])) / a[35]
    dx_c8[16] += (air_mv(native_top, x[16], d[2], x[3], d[1]) - air_mv(qtop, x[16], d[2], x[3], d[1])) / a[36]

    FCF_C8 = ca.integrator("FCF_C8", "cvodes",
                           {"x": x, "u": u, "p": ca.vertcat(d, p, qtop, qmain), "ode": dx_c8},
                           0.0, 900.0,
                           {"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 90000})

    # Integrator R5 (used for V2 and V3): with quad and hfac
    dx_r5 = dx_base.copy()
    dx_r5[0] += (air_mc(native_main, x[0], d[3]) - air_mc(qmain, x[0], d[3])) / p[122]
    dx_r5[1] += (air_mc(native_top, x[1], d[3]) - air_mc(qtop, x[1], d[3])) / p[123]
    dx_r5[2] += (sensible(cp_rho * native_main, x[2], d[1]) - sensible(cp_rho * qmain, x[2], d[1])) / p[112]
    dx_r5[3] += (sensible(cp_rho * native_top, x[3], d[1]) - sensible(hfac * cp_rho * qtop, x[3], d[1])) / p[120]
    dx_r5[15] += (air_mv(native_main, x[15], d[2], x[2], d[1]) - air_mv(qmain, x[15], d[2], x[2], d[1])) / a[35]
    dx_r5[16] += (air_mv(native_top, x[16], d[2], x[3], d[1]) - air_mv(qtop, x[16], d[2], x[3], d[1])) / a[36]

    FCF_R5 = ca.integrator("FCF_R5", "cvodes",
                           {"x": x, "u": u, "p": ca.vertcat(d, p, qtop, qmain, hfac),
                            "ode": dx_r5, "quad": qtop + qmain},
                           0.0, 900.0,
                           {"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 90000})

    print("Both C8 and R5 integrator computational graphs constructed successfully.")

    def run_integration(integrator, is_r5, xx, dd, uu, qtop_val, qmain_val):
        if is_r5:
            p_val = np.concatenate([dd, P_HV, [qtop_val, qmain_val, 1.0]])
        else:
            p_val = np.concatenate([dd, P_HV, [qtop_val, qmain_val]])
        res = integrator(x0=xx, u=uu, p=p_val)
        raw_stats = integrator.stats()
        stats_clean = {}
        for k, v in raw_stats.items():
            if isinstance(v, (int, float, str, bool)):
                stats_clean[k] = v
            else:
                stats_clean[k] = str(v)
        xf = np.asarray(res["xf"].full()).reshape(-1)
        T_out = float(xf[2])
        VP_out = float(xf[15])
        AH_out = float(ah_from_t_vp(T_out, VP_out))
        return T_out, AH_out, stats_clean

    variants = [
        {"name": "V0", "use_r5": False, "use_recomputed_flux": False,
         "desc": "C8 integrator + frozen emp_flux"},
        {"name": "V1", "use_r5": False, "use_recomputed_flux": True,
         "desc": "C8 integrator + recomputed q_common_pct flux"},
        {"name": "V2", "use_r5": True,  "use_recomputed_flux": False,
         "desc": "R5 integrator + frozen emp_flux"},
        {"name": "V3", "use_r5": True,  "use_recomputed_flux": True,
         "desc": "R5 integrator + recomputed q_common_pct flux"},
    ]

    flux_records = []
    stats_records = []
    comparison_records = []
    all_stats_dict = {}

    for var in variants:
        v_name = var["name"]
        use_r5 = var["use_r5"]
        use_recomp = var["use_recomputed_flux"]
        integrator = FCF_R5 if use_r5 else FCF_C8
        print(f"\n--- Running Variant {v_name}: {var['desc']} ---")

        all_stats_dict[v_name] = {}

        for idx, r in events_40.iterrows():
            eid = str(r.event_id)
            row_idx = int(r.row_index)
            wind = float(r.event_Windsp)

            # Frozen fluxes
            emp_pre = float(r.emp_flux_pre_23_23)
            emp_post = float(r.emp_flux_post_23_23)

            # Recomputed fluxes
            recomp_pre = float(q_common_pct(float(r.leeward_pre_pct), float(r.windward_pre_pct), wind))
            recomp_post = float(q_common_pct(float(r.leeward_post_pct), float(r.windward_post_pct), wind))

            q_pre = recomp_pre if use_recomp else emp_pre
            q_post = recomp_post if use_recomp else emp_post

            # Record full-precision flux details
            for arm, q_val, emp_val in [("pre", q_pre, emp_pre), ("post", q_post, emp_post)]:
                is_bit_id = (q_val == emp_val)
                abs_d = abs(q_val - emp_val)
                rel_d = abs_d / max(1e-12, abs(emp_val))
                flux_records.append({
                    "variant": v_name,
                    "event_id": eid,
                    "row_index": row_idx,
                    "arm": arm,
                    "q_top": q_val,
                    "q_top_hex": q_val.hex(),
                    "q_top_repr": repr(q_val),
                    "q_main": 0.0,
                    "q_main_hex": (0.0).hex(),
                    "emp_flux": emp_val,
                    "emp_flux_hex": emp_val.hex(),
                    "emp_flux_repr": repr(emp_val),
                    "bit_identical_to_emp_flux": is_bit_id,
                    "abs_diff_to_emp_flux": abs_d,
                    "rel_diff_to_emp_flux": rel_d,
                })

            xx = initial_state(r)
            dd = build_weather(r)
            uu_pre = full_action(r.u_pre)
            uu_post = full_action(r.u_post)

            # Pre arm integration
            T_pre, AH_pre, s_pre = run_integration(integrator, use_r5, xx, dd, uu_pre, q_pre, 0.0)
            # Post arm integration
            T_post, AH_post, s_post = run_integration(integrator, use_r5, xx, dd, uu_post, q_post, 0.0)

            all_stats_dict[v_name][f"{eid}_pre"] = s_pre
            all_stats_dict[v_name][f"{eid}_post"] = s_post

            for arm, st in [("pre", s_pre), ("post", s_post)]:
                stats_records.append({
                    "variant": v_name,
                    "event_id": eid,
                    "arm": arm,
                    "nsteps": st.get("nsteps", "missing"),
                    "nfevals": st.get("nfevals", "missing"),
                    "netfails": st.get("netfails", "missing"),
                    "h_last": st.get("h", st.get("last_step", "missing")),
                    "all_stats_json": json.dumps(st),
                })

            dT_mod = T_post - T_pre
            dAH_mod = AH_post - AH_pre

            # Compare against frozen R4 CF0
            r4_row = r4_cf0[r4_cf0["event_id"] == eid].iloc[0]
            r4_dT = float(r4_row["model_T_delta"])
            r4_dAH = float(r4_row["model_AH_delta"])
            r4_pre_T = float(r4_row["pre_T"])
            r4_post_T = float(r4_row["post_T"])
            r4_pre_AH = float(r4_row["pre_AH"])
            r4_post_AH = float(r4_row["post_AH"])

            abs_T_diff = abs(dT_mod - r4_dT)
            abs_AH_diff = abs(dAH_mod - r4_dAH)

            comparison_records.append({
                "variant": v_name,
                "event_id": eid,
                "row_index": row_idx,
                "horizon_min": 15,
                "pre_T": T_pre,
                "post_T": T_post,
                "pre_AH": AH_pre,
                "post_AH": AH_post,
                "dT_model": dT_mod,
                "dAH_model": dAH_mod,
                "pre_T_R4": r4_pre_T,
                "post_T_R4": r4_post_T,
                "pre_AH_R4": r4_pre_AH,
                "post_AH_R4": r4_post_AH,
                "dT_R4": r4_dT,
                "dAH_R4": r4_dAH,
                "abs_T_difference": abs_T_diff,
                "abs_AH_difference": abs_AH_diff,
                "pre_T_diff": abs(T_pre - r4_pre_T),
                "post_T_diff": abs(T_post - r4_post_T),
            })

    # Convert to DataFrames and save CSVs
    df_flux = pd.DataFrame(flux_records)
    df_stats = pd.DataFrame(stats_records)
    df_comp = pd.DataFrame(comparison_records)

    df_flux.to_csv(OUT / "R5_DIAG_D1_flux_comparison.csv", index=False)
    df_stats.to_csv(OUT / "R5_DIAG_D1_integrator_stats.csv", index=False)
    df_comp.to_csv(OUT / "R5_DIAG_D1_variant_comparison.csv", index=False)

    with open(OUT / "R5_DIAG_D1_integrator_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats_dict, f, indent=2)

    print("\n=== Summary of D1 Results by Variant ===")
    summary = {}
    for var in variants:
        v_name = var["name"]
        sub = df_comp[df_comp["variant"] == v_name]
        max_T = float(sub["abs_T_difference"].max())
        max_T_eid = str(sub.loc[sub["abs_T_difference"].idxmax(), "event_id"])
        max_AH = float(sub["abs_AH_difference"].max())
        max_AH_eid = str(sub.loc[sub["abs_AH_difference"].idxmax(), "event_id"])
        
        p0158_row = sub[sub["event_id"] == "CTIFL-E5-P-0158"].iloc[0]
        p0165_row = sub[sub["event_id"] == "CTIFL-E5-P-0165"].iloc[0]

        summary[v_name] = {
            "description": var["desc"],
            "max_abs_T_diff": max_T,
            "max_abs_T_event": max_T_eid,
            "max_abs_AH_diff": max_AH,
            "max_abs_AH_event": max_AH_eid,
            "count_exact_zero_T": int((sub["abs_T_difference"] == 0.0).sum()),
            "count_le_1e12_T": int((sub["abs_T_difference"] <= 1e-12).sum()),
            "count_le_1e9_T": int((sub["abs_T_difference"] <= 1e-9).sum()),
            "CTIFL-E5-P-0158": {
                "abs_T_diff": float(p0158_row["abs_T_difference"]),
                "abs_AH_diff": float(p0158_row["abs_AH_difference"]),
                "pre_T_diff": float(p0158_row["pre_T_diff"]),
                "post_T_diff": float(p0158_row["post_T_diff"]),
            },
            "CTIFL-E5-P-0165": {
                "abs_T_diff": float(p0165_row["abs_T_difference"]),
                "abs_AH_diff": float(p0165_row["abs_AH_difference"]),
                "pre_T_diff": float(p0165_row["pre_T_diff"]),
                "post_T_diff": float(p0165_row["post_T_diff"]),
            }
        }
        print(f"Variant {v_name}: Max |dT| = {max_T:.16e} ({max_T_eid}), Max |dAH| = {max_AH:.16e} ({max_AH_eid})")
        print(f"  P-0158: |dT| = {p0158_row['abs_T_difference']:.16e}, pre_T_diff = {p0158_row['pre_T_diff']:.16e}, post_T_diff = {p0158_row['post_T_diff']:.16e}")

    # Check Stop Conditions
    v0_pass = (summary["V0"]["max_abs_T_diff"] == 0.0 and summary["V0"]["max_abs_AH_diff"] == 0.0)
    summary["v0_exact_match_r4"] = v0_pass
    print(f"\nV0 Exact Match R4 (0.0): {v0_pass}")

    v3_p0158_expected = 1.2858481068043614e-06
    v3_p0158_actual = summary["V3"]["CTIFL-E5-P-0158"]["abs_T_diff"]
    v3_reproduced = (abs(v3_p0158_actual - v3_p0158_expected) < 1e-10)
    summary["v3_reproduced_v2_p0158"] = v3_reproduced
    print(f"V3 Reproduced v2 P-0158 error ({v3_p0158_actual:.16e} vs expected {v3_p0158_expected:.16e}): {v3_reproduced}")

    with open(OUT / "R5_DIAG_D1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown Report
    report_md = f"""# Stage D1 Diagnostic Compute Report

**Date:** 2026-10-20  
**Cohort:** 40 events of anchor_primary_3_6 (15 min horizon)

## 1. Summary of Variants Against Frozen R4 CF0

| Variant | Integrator | Ventilation Flux | Max |ΔT| (°C) | Max Event (T) | Max |ΔAH| (g/m³) | P-0158 |ΔT| | P-0165 |ΔT| |
|---|---|---|---|---|---|---|---|
| **V0** | C8 (p=[d,p,qtop,qmain]) | Frozen emp_flux | {summary['V0']['max_abs_T_diff']:.10e} | {summary['V0']['max_abs_T_event']} | {summary['V0']['max_abs_AH_diff']:.10e} | {summary['V0']['CTIFL-E5-P-0158']['abs_T_diff']:.10e} | {summary['V0']['CTIFL-E5-P-0165']['abs_T_diff']:.10e} |
| **V1** | C8 (p=[d,p,qtop,qmain]) | Recomputed q_common_pct | {summary['V1']['max_abs_T_diff']:.10e} | {summary['V1']['max_abs_T_event']} | {summary['V1']['max_abs_AH_diff']:.10e} | {summary['V1']['CTIFL-E5-P-0158']['abs_T_diff']:.10e} | {summary['V1']['CTIFL-E5-P-0165']['abs_T_diff']:.10e} |
| **V2** | R5 (quad, hfac=1) | Frozen emp_flux | {summary['V2']['max_abs_T_diff']:.10e} | {summary['V2']['max_abs_T_event']} | {summary['V2']['max_abs_AH_diff']:.10e} | {summary['V2']['CTIFL-E5-P-0158']['abs_T_diff']:.10e} | {summary['V2']['CTIFL-E5-P-0165']['abs_T_diff']:.10e} |
| **V3** | R5 (quad, hfac=1) | Recomputed q_common_pct | {summary['V3']['max_abs_T_diff']:.10e} | {summary['V3']['max_abs_T_event']} | {summary['V3']['max_abs_AH_diff']:.10e} | {summary['V3']['CTIFL-E5-P-0158']['abs_T_diff']:.10e} | {summary['V3']['CTIFL-E5-P-0165']['abs_T_diff']:.10e} |

## 2. Gate Verification & Stop Conditions

- **V0 exact reproduction of R4 CF0:** {'PASS' if v0_pass else 'FAIL'}
- **V3 exact reproduction of v2 P-0158:** {'PASS' if v3_reproduced else 'FAIL'}
"""
    (OUT / "R5_DIAG_D1_report.md").write_text(report_md, encoding="utf-8")
    print("\nSaved all D1 diagnostics to evidence/R5/DIAG_D1.")

    if not v0_pass:
        raise SystemExit("STOP CONDITION: V0 failed to reproduce frozen R4 CF0 exactly (0.0).")
    if not v3_reproduced:
        raise SystemExit("STOP CONDITION: V3 failed to reproduce the expected v2 P-0158 error.")

    print("=== PhysBench Stage D1 Diagnostic Completed Successfully ===")

if __name__ == "__main__":
    main()
