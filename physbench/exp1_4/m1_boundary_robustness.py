import json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca

from gl_gym.models.GreenLight.ode import ODE

ROOT = Path(__file__).resolve().parents[2]
FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT_JSON = ROOT / 'EXP1_4_M1_BOUNDARY_ROBUSTNESS.json'
OUT_CELLS = ROOT / 'EXP1_4_M1_CELL_SUMMARY.csv'
OUT_SCAN = ROOT / 'EXP1_4_M1_RESPONSE_SCAN.csv'
OUT_ROOTS = ROOT / 'EXP1_4_M1_REFINED_ROOTS.csv'
OUT_FACTORS = ROOT / 'EXP1_4_M1_FACTOR_SUMMARY.csv'

R = 8.3144598
K = 273.15
MCO2 = 44.01e-3
P = 101325.0
LOW = 0.1
HIGH = 0.9
HORIZONS = [300, 900, 1800]
FORCING_ROWS = [0, 6, 10]
SCAN_GRID = np.arange(250.0, 600.0 + 0.1, 25.0)
ROOT_WIDTH_TOL = 1e-4
RESPONSE_TOL = 1e-7
ROOT_DEDUP_TOL = 1e-3
EXP1_3_ROOT = 438.5370684042573
EXP1_3_DIRECT_EQ = 423.5522926326636
BASELINE_REPRO_TOL = 0.02


def sat_vp(t):
    t = np.asarray(t, dtype=float)
    return 610.78 * np.exp(17.2694 * t / (t + 238.3))


def ppm_to_mg_m3(t, ppm):
    return P * np.asarray(ppm) * MCO2 / (R * (np.asarray(t) + K))


def mg_m3_to_ppm(t, mg):
    return R * (np.asarray(t) + K) * np.asarray(mg) / (P * MCO2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_action(v):
    return np.array([0, 0, 0, v, 0, 0], dtype=float)


forcing = pd.read_csv(FORCING)
init = json.loads(INIT.read_text())

STATE_CASES = [
    {
        'state_id': 'S_COOL_HUMID',
        'air_temperature_c': 20.0,
        'air_rh_pct': 80.0,
        'air_vapor_pressure_pa': float(0.80 * sat_vp(20.0)),
        'canopy_temperature_c': 20.5,
    },
    {
        'state_id': 'S_NOMINAL',
        'air_temperature_c': 24.0,
        'air_rh_pct': 70.0,
        'air_vapor_pressure_pa': float(init['air_vapor_pressure_pa']),
        'canopy_temperature_c': 24.5,
    },
    {
        'state_id': 'S_WARM_DRY',
        'air_temperature_c': 30.0,
        'air_rh_pct': 55.0,
        'air_vapor_pressure_pa': float(0.55 * sat_vp(30.0)),
        'canopy_temperature_c': 30.5,
    },
]
STATE_MAP = {s['state_id']: s for s in STATE_CASES}

# Frozen native model setup inherited from EXP1.3.
env = gym.make(
    'gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil', 'uCO2', 'uThScr', 'uVent', 'uLamp', 'uBlScr'],
    normalize_actions=False,
    parameter_provider='fixed',
)
env.reset(seed=3407, options={'scenario': {'location': 'Amsterdam', 'growth_year': 2010, 'start_day': 244}})
e = env.unwrapped
native_reset_x = e.x.copy()
p_native = np.asarray(e.p, dtype=float).copy()

# Build the unchanged native ODE once, then horizon-specific CVODES wrappers.
x_sym = ca.SX.sym('x', e.nx)
u_sym = ca.SX.sym('u', e.nu)
d_sym = ca.SX.sym('d', e.nd)
p_sym = ca.SX.sym('p', len(p_native))
dx_sym = ODE(x_sym, u_sym, d_sym, p_sym)
F_BY_H = {}
for h in HORIZONS:
    F_BY_H[h] = ca.integrator(
        f'EXP1_4_M1_H{h}',
        'cvodes',
        {'x': x_sym, 'u': u_sym, 'p': ca.vertcat(d_sym, p_sym), 'ode': dx_sym},
        0.0,
        float(h),
        {'abstol': 1e-4, 'reltol': 1e-4, 'max_num_steps': int(1.5e5)},
    )


def disturbance_for_row(row_idx):
    row = forcing.iloc[int(row_idx)]
    d0 = np.zeros(10, dtype=float)
    d0[0] = row['global_radiation_w_m2']
    d0[1] = row['outdoor_temperature_c']
    d0[2] = row['outdoor_vapor_pressure_pa']
    d0[3] = ppm_to_mg_m3(row['outdoor_temperature_c'], row['outdoor_co2_ppm'])
    d0[4] = row['wind_speed_m_s']
    d0[5] = row['sky_temperature_c']
    d0[6] = row['soil_boundary_temperature_c']
    # Native GreenLight extras exactly follow EXP0.5 semantics.
    d0[7] = float((forcing.loc[:int(row_idx), 'global_radiation_w_m2'] * 900.0).sum() / 1e6)
    d0[8] = 1.0 if row['global_radiation_w_m2'] > 0 else 0.0
    d0[9] = d0[8]
    return d0


def x0_for(state_id, c0):
    s = STATE_MAP[state_id]
    z = native_reset_x.copy()
    z[2] = s['air_temperature_c']
    z[15] = s['air_vapor_pressure_pa']
    z[4] = s['canopy_temperature_c']
    z[0] = ppm_to_mg_m3(s['air_temperature_c'], float(c0))
    return z


arm_cache = {}
runtime_errors = []


def run_arm(horizon_s, state_id, forcing_row, c0, vent):
    key = (int(horizon_s), state_id, int(forcing_row), round(float(c0), 10), float(vent))
    if key in arm_cache:
        return arm_cache[key]
    try:
        z = x0_for(state_id, c0)
        d0 = disturbance_for_row(forcing_row)
        combined_p = np.concatenate([d0, p_native])
        res = F_BY_H[int(horizon_s)](x0=z, u=full_action(vent), p=combined_p)
        xf = np.asarray(res['xf'].full()).reshape(-1)
        finite = bool(np.all(np.isfinite(xf)))
        final_ppm = float(mg_m3_to_ppm(xf[2], xf[0])) if finite else float('nan')
        out = {'final_co2_ppm': final_ppm, 'finite': finite, 'error': None}
    except Exception as exc:
        out = {'final_co2_ppm': float('nan'), 'finite': False, 'error': repr(exc)}
        runtime_errors.append({'key': list(key), 'error': repr(exc)})
    arm_cache[key] = out
    return out


def response(horizon_s, state_id, forcing_row, c0):
    lo = run_arm(horizon_s, state_id, forcing_row, c0, LOW)
    hi = run_arm(horizon_s, state_id, forcing_row, c0, HIGH)
    if not (lo['finite'] and hi['finite']):
        return float('nan')
    return float(hi['final_co2_ppm'] - lo['final_co2_ppm'])


def refine_bracket(horizon_s, state_id, forcing_row, lo, hi, flo, fhi):
    initial_lo, initial_hi = float(lo), float(hi)
    if abs(flo) <= RESPONSE_TOL:
        return {'root_ppm': float(lo), 'response_ppm': float(flo), 'iterations': 0,
                'final_bracket_width_ppm': 0.0, 'initial_bracket_low_ppm': initial_lo,
                'initial_bracket_high_ppm': initial_hi, 'converged': True}
    if abs(fhi) <= RESPONSE_TOL:
        return {'root_ppm': float(hi), 'response_ppm': float(fhi), 'iterations': 0,
                'final_bracket_width_ppm': 0.0, 'initial_bracket_low_ppm': initial_lo,
                'initial_bracket_high_ppm': initial_hi, 'converged': True}
    if (not np.isfinite(flo)) or (not np.isfinite(fhi)) or flo * fhi > 0:
        raise AssertionError('Invalid sign-changing bracket')
    for it in range(1, 51):
        mid = 0.5 * (lo + hi)
        fm = response(horizon_s, state_id, forcing_row, mid)
        if not np.isfinite(fm):
            return {'root_ppm': float('nan'), 'response_ppm': float('nan'), 'iterations': it,
                    'final_bracket_width_ppm': float(hi-lo), 'initial_bracket_low_ppm': initial_lo,
                    'initial_bracket_high_ppm': initial_hi, 'converged': False}
        if abs(fm) <= RESPONSE_TOL or (hi - lo) <= ROOT_WIDTH_TOL:
            return {'root_ppm': float(mid), 'response_ppm': float(fm), 'iterations': it,
                    'final_bracket_width_ppm': float(hi-lo), 'initial_bracket_low_ppm': initial_lo,
                    'initial_bracket_high_ppm': initial_hi, 'converged': True}
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    mid = 0.5 * (lo + hi)
    fm = response(horizon_s, state_id, forcing_row, mid)
    return {'root_ppm': float(mid), 'response_ppm': float(fm), 'iterations': 50,
            'final_bracket_width_ppm': float(hi-lo), 'initial_bracket_low_ppm': initial_lo,
            'initial_bracket_high_ppm': initial_hi,
            'converged': bool((hi-lo) <= ROOT_WIDTH_TOL or abs(fm) <= RESPONSE_TOL)}


def dedup_roots(roots):
    good = [r for r in roots if np.isfinite(r['root_ppm'])]
    good.sort(key=lambda r: r['root_ppm'])
    out = []
    for r in good:
        if not out or abs(r['root_ppm'] - out[-1]['root_ppm']) > ROOT_DEDUP_TOL:
            out.append(r)
        elif abs(r['response_ppm']) < abs(out[-1]['response_ppm']):
            out[-1] = r
    return out


cell_rows = []
scan_rows = []
root_rows = []
cell_objects = []

for h in HORIZONS:
    for s in STATE_CASES:
        for fr in FORCING_ROWS:
            sid = s['state_id']
            frow = forcing.iloc[fr]
            condition_id = f'H{h}_{sid}_FROW{fr}'
            ys = []
            for c0 in SCAN_GRID:
                y = response(h, sid, fr, float(c0))
                ys.append(y)
                scan_rows.append({
                    'model_id': 'M1', 'condition_id': condition_id,
                    'horizon_s': h, 'state_id': sid, 'forcing_row': fr,
                    'initial_co2_ppm': float(c0), 'high_minus_low_final_co2_ppm': y,
                })

            candidates = []
            for i, y in enumerate(ys):
                if np.isfinite(y) and abs(y) <= RESPONSE_TOL:
                    candidates.append(refine_bracket(h, sid, fr, SCAN_GRID[i], SCAN_GRID[i], y, y))
            for i in range(len(SCAN_GRID)-1):
                y0, y1 = ys[i], ys[i+1]
                if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 < 0:
                    candidates.append(refine_bracket(h, sid, fr, SCAN_GRID[i], SCAN_GRID[i+1], y0, y1))
            roots = dedup_roots(candidates)
            unique_root = roots[0]['root_ppm'] if len(roots) == 1 else None

            d0 = disturbance_for_row(fr)
            direct_eq = float(mg_m3_to_ppm(s['air_temperature_c'], d0[3]))
            shift = float(unique_root - direct_eq) if unique_root is not None else None
            finite_scan = bool(np.all(np.isfinite(np.asarray(ys, dtype=float))))
            all_roots_converged = bool(all(r['converged'] for r in roots))
            all_roots_inside = bool(all(
                r['initial_bracket_low_ppm'] - 1e-12 <= r['root_ppm'] <= r['initial_bracket_high_ppm'] + 1e-12
                for r in roots
            ))

            cell = {
                'model_id': 'M1', 'model': 'GreenLight-Gym2', 'condition_id': condition_id,
                'horizon_s': int(h), 'state': s, 'forcing_row': int(fr),
                'forcing_time_s': float(frow['time_s']),
                'forcing': {k: float(frow[k]) for k in forcing.columns if k != 'time_s'},
                'direct_transport_equilibrium_ppm': direct_eq,
                'root_count': len(roots), 'roots': roots,
                'primary_boundary_ppm': unique_root, 'boundary_shift_ppm': shift,
                'finite_scan': finite_scan, 'all_roots_converged': all_roots_converged,
                'all_roots_inside_detected_brackets': all_roots_inside,
            }
            cell_objects.append(cell)
            cell_rows.append({
                'model_id': 'M1', 'condition_id': condition_id, 'horizon_s': h,
                'state_id': sid, 'forcing_row': fr, 'forcing_time_s': float(frow['time_s']),
                'initial_air_temperature_c': s['air_temperature_c'],
                'initial_air_rh_pct': s['air_rh_pct'],
                'global_radiation_w_m2': float(frow['global_radiation_w_m2']),
                'outdoor_temperature_c': float(frow['outdoor_temperature_c']),
                'outdoor_rh_pct': float(frow['outdoor_rh_pct']),
                'wind_speed_m_s': float(frow['wind_speed_m_s']),
                'direct_transport_equilibrium_ppm': direct_eq,
                'root_count': len(roots),
                'all_roots_ppm': ';'.join(f"{r['root_ppm']:.9f}" for r in roots),
                'primary_boundary_ppm': unique_root,
                'boundary_shift_ppm': shift,
                'finite_scan': finite_scan,
            })
            for j, r in enumerate(roots):
                root_rows.append({
                    'model_id': 'M1', 'condition_id': condition_id, 'horizon_s': h,
                    'state_id': sid, 'forcing_row': fr, 'root_index': j,
                    **r,
                })
            print(condition_id, 'roots=', [r['root_ppm'] for r in roots], 'direct=', direct_eq, flush=True)

cells_df = pd.DataFrame(cell_rows)
scan_df = pd.DataFrame(scan_rows)
roots_df = pd.DataFrame(root_rows)


def factor_summary(df):
    records = []
    for factor in ['horizon_s', 'state_id', 'forcing_row']:
        for level, g in df.groupby(factor, dropna=False):
            u = g[g['root_count'] == 1]['primary_boundary_ppm'].dropna().astype(float)
            sh = g[g['root_count'] == 1]['boundary_shift_ppm'].dropna().astype(float)
            records.append({
                'model_id': 'M1', 'factor': factor, 'level': str(level),
                'n_cells': int(len(g)), 'n_unique_root_cells': int(len(u)),
                'unique_root_fraction': float(len(u)/len(g)),
                'boundary_min_ppm': float(u.min()) if len(u) else np.nan,
                'boundary_q25_ppm': float(u.quantile(0.25)) if len(u) else np.nan,
                'boundary_median_ppm': float(u.median()) if len(u) else np.nan,
                'boundary_q75_ppm': float(u.quantile(0.75)) if len(u) else np.nan,
                'boundary_max_ppm': float(u.max()) if len(u) else np.nan,
                'boundary_range_ppm': float(u.max()-u.min()) if len(u) else np.nan,
                'shift_median_ppm': float(sh.median()) if len(sh) else np.nan,
            })
    return pd.DataFrame(records)

factor_df = factor_summary(cells_df)

baseline = cells_df[(cells_df['horizon_s'] == 900) &
                    (cells_df['state_id'] == 'S_NOMINAL') &
                    (cells_df['forcing_row'] == 0)].iloc[0]
baseline_root_error = abs(float(baseline['primary_boundary_ppm']) - EXP1_3_ROOT) if baseline['root_count'] == 1 else float('inf')
baseline_direct_error = abs(float(baseline['direct_transport_equilibrium_ppm']) - EXP1_3_DIRECT_EQ)
all_finite = bool(cells_df['finite_scan'].all())
all_converged = bool(all(c['all_roots_converged'] for c in cell_objects))
all_inside = bool(all(c['all_roots_inside_detected_brackets'] for c in cell_objects))

result = {
    'experiment': 'PhysBench-GH EXP1.4',
    'model_id': 'M1', 'model': 'GreenLight-Gym2',
    'frozen_commit': '2d3febb1ea002b24b452e32293e990beb78d3ce1',
    'benchmark_forcing_sha256': sha(FORCING),
    'matched_initial_state_sha256': sha(INIT),
    'design': {
        'horizons_s': HORIZONS,
        'state_cases': STATE_CASES,
        'forcing_rows': FORCING_ROWS,
        'co2_scan_ppm': SCAN_GRID.tolist(),
        'low_ventilation_fraction': LOW,
        'high_ventilation_fraction': HIGH,
        'root_width_tolerance_ppm': ROOT_WIDTH_TOL,
        'response_tolerance_ppm': RESPONSE_TOL,
    },
    'cells': cell_objects,
    'runtime_audit': {
        'all_scan_responses_finite': all_finite,
        'all_refined_roots_converged': all_converged,
        'all_refined_roots_inside_detected_brackets': all_inside,
        'runtime_errors': runtime_errors,
        'cached_native_arm_evaluations': len(arm_cache),
        'baseline_exp1_3_root_abs_error_ppm': float(baseline_root_error),
        'baseline_exp1_3_direct_equilibrium_abs_error_ppm': float(baseline_direct_error),
    },
    'interpretation_limit': (
        'Controlled model-internal robustness map. No equation, flux, solver tolerance, '
        'controller mapping, or model-specific parameter was repaired or retuned.'
    ),
}
result['run_pass'] = bool(
    all_finite and all_converged and all_inside and len(runtime_errors) == 0
    and baseline_root_error <= BASELINE_REPRO_TOL
    and baseline_direct_error <= BASELINE_REPRO_TOL
)

cells_df.to_csv(OUT_CELLS, index=False)
scan_df.to_csv(OUT_SCAN, index=False)
roots_df.to_csv(OUT_ROOTS, index=False)
factor_df.to_csv(OUT_FACTORS, index=False)
OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'run_pass': result['run_pass'], 'runtime_audit': result['runtime_audit']}, indent=2))
env.close()
if not result['run_pass']:
    raise SystemExit('M1 EXP1.4 runtime/reproducibility gate failed')
