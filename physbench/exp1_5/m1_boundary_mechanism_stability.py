import json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca

from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT = Path(__file__).resolve().parents[2]
FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT_JSON = ROOT / 'EXP1_5_M1_BOUNDARY_MECHANISM_STABILITY.json'
OUT_CELLS = ROOT / 'EXP1_5_M1_CELL_SUMMARY.csv'
OUT_ROOTS = ROOT / 'EXP1_5_M1_ROOT_MECHANISMS.csv'
OUT_EDGES = ROOT / 'EXP1_5_M1_EDGE_MECHANISMS.csv'
OUT_TERMS = ROOT / 'EXP1_5_M1_TERM_CONTRIBUTIONS.csv'
OUT_SCAN = ROOT / 'EXP1_5_M1_SURFACE_REGRESSION_SCAN.csv'

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
ANCHOR_CONDITION = 'H900_S_NOMINAL_FROW0'
EXP1_4_ANCHOR = 438.5370684042573
ANCHOR_TOL = 0.02
EXPECTED_UNIQUE = 21
EXPECTED_NO_ROOT = 6
EXPECTED_MULTIPLE = 0
NAMED_TOTAL_CLOSURE_TOL = 1e-10
PAIR_DENSITY_CLOSURE_TOL = 0.005
PAIR_PPM_CLOSURE_TOL = 0.005
ACTIVE_SHARE = 0.10

TERM_NAMES = ['canopy_net','main_to_top','main_to_outside','co2_injection','blower_source','pad_source']
VECTOR_NAMES = TERM_NAMES + ['temperature_conversion']
GROUP_NAMES = ['direct_exchange','canopy_biology','intercompartment','external_source','representation']


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
    return np.array([0,0,0,v,0,0], dtype=float)


forcing = pd.read_csv(FORCING)
init = json.loads(INIT.read_text())
STATE_CASES = [
    {'state_id':'S_COOL_HUMID','air_temperature_c':20.0,'air_rh_pct':80.0,
     'air_vapor_pressure_pa':float(0.80*sat_vp(20.0)),'canopy_temperature_c':20.5},
    {'state_id':'S_NOMINAL','air_temperature_c':24.0,'air_rh_pct':70.0,
     'air_vapor_pressure_pa':float(init['air_vapor_pressure_pa']),'canopy_temperature_c':24.5},
    {'state_id':'S_WARM_DRY','air_temperature_c':30.0,'air_rh_pct':55.0,
     'air_vapor_pressure_pa':float(0.55*sat_vp(30.0)),'canopy_temperature_c':30.5},
]
STATE_MAP = {s['state_id']:s for s in STATE_CASES}

# Frozen GreenLight model.
env = gym.make(
    'gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,
    parameter_provider='fixed',
)
env.reset(seed=3407, options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e = env.unwrapped
native_reset_x = e.x.copy()
p_native = np.asarray(e.p, dtype=float).copy()

# Native ODE and read-only quadratures inherited from EXP1.3.
x = ca.SX.sym('x', e.nx)
u = ca.SX.sym('u', e.nu)
d = ca.SX.sym('d', e.nd)
p = ca.SX.sym('p', len(p_native))
a = aux_states.update(x,u,d,p)
dx = ODE(x,u,d,p)
q_canopy = -a[216] / p[122]
q_main_top = -a[217] / p[122]
q_main_out = -a[219] / p[122]
q_injection = a[222] / p[122]
q_blower = a[223] / p[122]
q_pad = a[224] / p[122]
q_terms = ca.vertcat(q_canopy,q_main_top,q_main_out,q_injection,q_blower,q_pad,dx[0])

F_BY_H = {}
FQ_BY_H = {}
for h in HORIZONS:
    F_BY_H[h] = ca.integrator(
        f'EXP1_5_M1_NATIVE_H{h}','cvodes',
        {'x':x,'u':u,'p':ca.vertcat(d,p),'ode':dx},0.0,float(h),
        {'abstol':1e-4,'reltol':1e-4,'max_num_steps':int(1.5e5)}
    )
    FQ_BY_H[h] = ca.integrator(
        f'EXP1_5_M1_QUAD_H{h}','cvodes',
        {'x':x,'u':u,'p':ca.vertcat(d,p),'ode':dx,'quad':q_terms},0.0,float(h),
        {'abstol':1e-4,'reltol':1e-4,'max_num_steps':int(1.5e5)}
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
    d0[7] = float((forcing.loc[:int(row_idx),'global_radiation_w_m2'] * 900.0).sum() / 1e6)
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


def native_arm(horizon_s, state_id, forcing_row, c0, vent):
    key = (int(horizon_s),state_id,int(forcing_row),round(float(c0),10),float(vent))
    if key in arm_cache:
        return arm_cache[key]
    try:
        z = x0_for(state_id,c0)
        cp = np.concatenate([disturbance_for_row(forcing_row),p_native])
        r = F_BY_H[int(horizon_s)](x0=z,u=full_action(vent),p=cp)
        xf = np.asarray(r['xf'].full()).reshape(-1)
        finite = bool(np.all(np.isfinite(xf)))
        out = {'xf':xf,'final_co2_ppm':float(mg_m3_to_ppm(xf[2],xf[0])) if finite else float('nan'),
               'finite':finite,'error':None}
    except Exception as exc:
        out = {'xf':None,'final_co2_ppm':float('nan'),'finite':False,'error':repr(exc)}
        runtime_errors.append({'key':list(key),'error':repr(exc)})
    arm_cache[key] = out
    return out


def response(h,s,fr,c0):
    lo = native_arm(h,s,fr,c0,LOW)
    hi = native_arm(h,s,fr,c0,HIGH)
    if not (lo['finite'] and hi['finite']):
        return float('nan')
    return float(hi['final_co2_ppm'] - lo['final_co2_ppm'])


def refine_bracket(h,s,fr,lo,hi,flo,fhi):
    initial_lo, initial_hi = float(lo), float(hi)
    if abs(flo) <= RESPONSE_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,
                'final_bracket_width_ppm':0.0,'initial_bracket_low_ppm':initial_lo,
                'initial_bracket_high_ppm':initial_hi,'converged':True}
    if abs(fhi) <= RESPONSE_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,
                'final_bracket_width_ppm':0.0,'initial_bracket_low_ppm':initial_lo,
                'initial_bracket_high_ppm':initial_hi,'converged':True}
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo*fhi > 0:
        raise AssertionError('Invalid sign-changing bracket')
    for it in range(1,51):
        mid = 0.5*(lo+hi)
        fm = response(h,s,fr,mid)
        if not np.isfinite(fm):
            return {'root_ppm':float('nan'),'response_ppm':float('nan'),'iterations':it,
                    'final_bracket_width_ppm':float(hi-lo),'initial_bracket_low_ppm':initial_lo,
                    'initial_bracket_high_ppm':initial_hi,'converged':False}
        if abs(fm) <= RESPONSE_TOL or (hi-lo) <= ROOT_WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,
                    'final_bracket_width_ppm':float(hi-lo),'initial_bracket_low_ppm':initial_lo,
                    'initial_bracket_high_ppm':initial_hi,'converged':True}
        if flo*fm <= 0:
            hi,fhi = mid,fm
        else:
            lo,flo = mid,fm
    mid = 0.5*(lo+hi); fm = response(h,s,fr,mid)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':50,
            'final_bracket_width_ppm':float(hi-lo),'initial_bracket_low_ppm':initial_lo,
            'initial_bracket_high_ppm':initial_hi,
            'converged':bool((hi-lo)<=ROOT_WIDTH_TOL or abs(fm)<=RESPONSE_TOL)}


def dedup_roots(roots):
    good = [r for r in roots if np.isfinite(r['root_ppm'])]
    good.sort(key=lambda r:r['root_ppm'])
    out=[]
    for r in good:
        if not out or abs(r['root_ppm']-out[-1]['root_ppm']) > ROOT_DEDUP_TOL:
            out.append(r)
        elif abs(r['response_ppm']) < abs(out[-1]['response_ppm']):
            out[-1]=r
    return out


def scan_cell(h,s,fr):
    ys=[]
    for c0 in SCAN_GRID:
        ys.append(response(h,s,fr,float(c0)))
    candidates=[]
    for i,yv in enumerate(ys):
        if np.isfinite(yv) and abs(yv) <= RESPONSE_TOL:
            candidates.append(refine_bracket(h,s,fr,SCAN_GRID[i],SCAN_GRID[i],yv,yv))
    for i in range(len(SCAN_GRID)-1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i+1]) and ys[i]*ys[i+1] < 0:
            candidates.append(refine_bracket(h,s,fr,SCAN_GRID[i],SCAN_GRID[i+1],ys[i],ys[i+1]))
    return ys,dedup_roots(candidates)


def decomp_arm(h,s,fr,c0,vent):
    z = x0_for(s,c0)
    cp = np.concatenate([disturbance_for_row(fr),p_native])
    r = FQ_BY_H[int(h)](x0=z,u=full_action(vent),p=cp)
    xf = np.asarray(r['xf'].full()).reshape(-1)
    q = np.asarray(r['qf'].full()).reshape(-1)
    nat = native_arm(h,s,fr,c0,vent)
    state_err = float(np.max(np.abs(xf-nat['xf'])))
    term_integrals = {TERM_NAMES[i]:float(q[i]) for i in range(len(TERM_NAMES))}
    qsum = float(sum(term_integrals.values()))
    qtotal = float(q[-1])
    density_delta = float(xf[0]-z[0])
    return {
        'ventilation_command_fraction':float(vent),'initial_co2_ppm':float(c0),
        'final_density_mg_m3':float(xf[0]),'final_air_temperature_c':float(xf[2]),
        'final_co2_ppm':float(mg_m3_to_ppm(xf[2],xf[0])),
        'integrated_density_contributions_mg_m3':term_integrals,
        'integrated_total_ode_density_change_mg_m3':qtotal,
        'sum_named_density_contributions_mg_m3':qsum,
        'actual_density_change_mg_m3':density_delta,
        'named_vs_total_ode_closure_mg_m3':float(qsum-qtotal),
        'total_ode_vs_actual_closure_mg_m3':float(qtotal-density_delta),
        'augmented_vs_native_final_state_max_abs_diff':state_err,
        'finite':bool(np.all(np.isfinite(xf)) and np.all(np.isfinite(q))),
    }


def paired_decomp(h,s,fr,c0):
    lo = decomp_arm(h,s,fr,c0,LOW)
    hi = decomp_arm(h,s,fr,c0,HIGH)
    CH,CL = hi['final_density_mg_m3'],lo['final_density_mg_m3']
    TH,TL = hi['final_air_temperature_c'],lo['final_air_temperature_c']
    kH = float(R*(TH+K)/(P*MCO2)); kL = float(R*(TL+K)/(P*MCO2))
    kbar = 0.5*(kH+kL); Cbar = 0.5*(CH+CL)
    flux_ppm = {}
    density_parts = {}
    for name in TERM_NAMES:
        dq = hi['integrated_density_contributions_mg_m3'][name]-lo['integrated_density_contributions_mg_m3'][name]
        density_parts[name] = float(dq)
        flux_ppm[name] = float(kbar*dq)
    density_contrast = float(CH-CL)
    temp_conversion = float(Cbar*(kH-kL))
    flux_ppm['temperature_conversion'] = temp_conversion
    actual_ppm = float(hi['final_co2_ppm']-lo['final_co2_ppm'])
    reconstructed = float(sum(flux_ppm.values()))
    return {
        'horizon_s':int(h),'state_id':s,'forcing_row':int(fr),'initial_co2_ppm':float(c0),
        'LOW':lo,'HIGH':hi,
        'high_minus_low':{
            'final_density_mg_m3':density_contrast,'final_co2_ppm':actual_ppm,
            'flux_contributions_ppm':flux_ppm,
            'density_flux_contributions_mg_m3':density_parts,
            'reconstructed_final_co2_ppm':reconstructed,
            'ppm_closure_error':float(reconstructed-actual_ppm),
            'density_flux_closure_error_mg_m3':float(sum(density_parts.values())-density_contrast),
        }
    }


def group_vector(v):
    return {
        'direct_exchange':float(v['main_to_outside']),
        'canopy_biology':float(v['canopy_net']),
        'intercompartment':float(v['main_to_top']),
        'external_source':float(v['co2_injection']+v['blower_source']+v['pad_source']),
        'representation':float(v['temperature_conversion']),
    }


def vector_metrics(v, anchor=None):
    arr = np.asarray([v[k] for k in VECTOR_NAMES],dtype=float)
    l1 = float(np.sum(np.abs(arr)))
    norm = arr/l1 if l1>0 else np.zeros_like(arr)
    idx = int(np.argmax(np.abs(arr))) if l1>0 else 0
    out = {
        'l1_total_abs_ppm':l1,
        'dominant_term':VECTOR_NAMES[idx] if l1>0 else 'NONE',
        'dominance_share':float(np.max(np.abs(norm))) if l1>0 else 0.0,
        'active_terms':'|'.join(VECTOR_NAMES[i] for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
        'active_sign_pattern':'|'.join(f"{VECTOR_NAMES[i]}:{'+' if x>0 else '-'}" for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
    }
    for i,k in enumerate(VECTOR_NAMES):
        out[f'norm_{k}'] = float(norm[i])
    g = group_vector(v)
    garr=np.asarray([g[k] for k in GROUP_NAMES],dtype=float)
    gl1=float(np.sum(np.abs(garr)))
    if gl1>0:
        gi=int(np.argmax(np.abs(garr)))
        out['dominant_group']=GROUP_NAMES[gi]
        out['dominant_group_share']=float(abs(garr[gi])/gl1)
    else:
        out['dominant_group']='NONE'; out['dominant_group_share']=0.0
    if anchor is not None:
        aa=np.asarray([anchor[k] for k in VECTOR_NAMES],dtype=float)
        al1=float(np.sum(np.abs(aa))); an=aa/al1 if al1>0 else np.zeros_like(aa)
        den=float(np.linalg.norm(norm)*np.linalg.norm(an))
        out['anchor_cosine_similarity']=float(np.dot(norm,an)/den) if den>0 else float('nan')
        out['anchor_l1_distance']=float(np.sum(np.abs(norm-an)))
        aidx=int(np.argmax(np.abs(aa))) if al1>0 else 0
        out['dominant_switch_from_anchor']=bool(out['dominant_term'] != (VECTOR_NAMES[aidx] if al1>0 else 'NONE'))
        shared=[i for i in range(len(VECTOR_NAMES)) if abs(norm[i])>=ACTIVE_SHARE and abs(an[i])>=ACTIVE_SHARE]
        flips=[VECTOR_NAMES[i] for i in shared if np.sign(norm[i])!=np.sign(an[i])]
        out['active_term_sign_flips_vs_anchor']='|'.join(flips)
        out['n_active_term_sign_flips_vs_anchor']=int(len(flips))
    return out


cell_rows=[]; scan_rows=[]; root_objects=[]; edge_objects=[]; all_pair_objects=[]
for h in HORIZONS:
    for st in STATE_CASES:
        sid=st['state_id']
        for fr in FORCING_ROWS:
            cid=f'H{h}_{sid}_FROW{fr}'
            ys,roots=scan_cell(h,sid,fr)
            for c0,yv in zip(SCAN_GRID,ys):
                scan_rows.append({'model_id':'M1','condition_id':cid,'horizon_s':h,'state_id':sid,
                                  'forcing_row':fr,'initial_co2_ppm':float(c0),'high_minus_low_final_co2_ppm':float(yv)})
            topology='UNIQUE_CROSSING_IN_WINDOW' if len(roots)==1 else ('NO_CROSSING_IN_WINDOW' if len(roots)==0 else 'MULTIPLE_CROSSINGS_IN_WINDOW')
            primary=float(roots[0]['root_ppm']) if len(roots)==1 else float('nan')
            cell_rows.append({'model_id':'M1','condition_id':cid,'horizon_s':h,'state_id':sid,'forcing_row':fr,
                              'root_count':len(roots),'topology':topology,'primary_boundary_ppm':primary,
                              'response_at_250_ppm':float(ys[0]),'response_at_600_ppm':float(ys[-1])})
            if len(roots)==1:
                pair=paired_decomp(h,sid,fr,primary)
                pair['condition_id']=cid; pair['point_type']='ROOT'; pair['root_meta']=roots[0]
                root_objects.append(pair); all_pair_objects.append(pair)
            elif len(roots)==0:
                for edge in [250.0,600.0]:
                    pair=paired_decomp(h,sid,fr,edge)
                    pair['condition_id']=cid; pair['point_type']='EDGE_AUDIT'; pair['edge_ppm']=edge
                    edge_objects.append(pair); all_pair_objects.append(pair)
            else:
                for r in roots:
                    pair=paired_decomp(h,sid,fr,float(r['root_ppm']))
                    pair['condition_id']=cid; pair['point_type']='MULTIPLE_ROOT_AUDIT'; pair['root_meta']=r
                    root_objects.append(pair); all_pair_objects.append(pair)

anchor_matches=[x for x in root_objects if x['condition_id']==ANCHOR_CONDITION and x['point_type']=='ROOT']
if len(anchor_matches)!=1:
    raise AssertionError(f'Anchor root count != 1: {len(anchor_matches)}')
anchor_obj=anchor_matches[0]
anchor_vec=anchor_obj['high_minus_low']['flux_contributions_ppm']

root_rows=[]; edge_rows=[]; term_rows=[]
for obj in root_objects+edge_objects:
    v=obj['high_minus_low']['flux_contributions_ppm']
    m=vector_metrics(v,anchor_vec)
    base={'model_id':'M1','condition_id':obj['condition_id'],'point_type':obj['point_type'],
          'horizon_s':obj['horizon_s'],'state_id':obj['state_id'],'forcing_row':obj['forcing_row'],
          'initial_co2_ppm':obj['initial_co2_ppm'],'final_high_minus_low_ppm':obj['high_minus_low']['final_co2_ppm'],
          'ppm_closure_error':obj['high_minus_low']['ppm_closure_error'],**m}
    if obj['point_type'] in ['ROOT','MULTIPLE_ROOT_AUDIT']:
        root_rows.append(base)
    else:
        edge_rows.append(base)
    g=group_vector(v)
    for k in VECTOR_NAMES:
        term_rows.append({**{kk:base[kk] for kk in ['model_id','condition_id','point_type','horizon_s','state_id','forcing_row','initial_co2_ppm']},
                          'term':k,'contribution_ppm':float(v[k]),'abs_share':float(abs(v[k])/m['l1_total_abs_ppm']) if m['l1_total_abs_ppm']>0 else 0.0})
    for k in GROUP_NAMES:
        term_rows.append({**{kk:base[kk] for kk in ['model_id','condition_id','point_type','horizon_s','state_id','forcing_row','initial_co2_ppm']},
                          'term':f'GROUP::{k}','contribution_ppm':float(g[k]),'abs_share':float(abs(g[k])/sum(abs(x) for x in g.values())) if sum(abs(x) for x in g.values())>0 else 0.0})

cells_df=pd.DataFrame(cell_rows); roots_df=pd.DataFrame(root_rows); edges_df=pd.DataFrame(edge_rows); terms_df=pd.DataFrame(term_rows); scan_df=pd.DataFrame(scan_rows)
cells_df.to_csv(OUT_CELLS,index=False)
roots_df.to_csv(OUT_ROOTS,index=False)
edges_df.to_csv(OUT_EDGES,index=False)
terms_df.to_csv(OUT_TERMS,index=False)
scan_df.to_csv(OUT_SCAN,index=False)

all_arms=[obj[k] for obj in all_pair_objects for k in ['LOW','HIGH']]
max_state_err=max(a['augmented_vs_native_final_state_max_abs_diff'] for a in all_arms)
max_named_total=max(abs(a['named_vs_total_ode_closure_mg_m3']) for a in all_arms)
max_density_pair=max(abs(obj['high_minus_low']['density_flux_closure_error_mg_m3']) for obj in all_pair_objects)
max_ppm=max(abs(obj['high_minus_low']['ppm_closure_error']) for obj in all_pair_objects)
all_finite=all(a['finite'] for a in all_arms)
unique_count=int((cells_df.root_count==1).sum()); no_count=int((cells_df.root_count==0).sum()); multiple_count=int((cells_df.root_count>1).sum())
anchor_root=float(anchor_obj['initial_co2_ppm']); anchor_err=abs(anchor_root-EXP1_4_ANCHOR)
all_roots_inside=all(
    obj.get('root_meta',{}).get('initial_bracket_low_ppm',-np.inf)-1e-12 <= obj['initial_co2_ppm'] <= obj.get('root_meta',{}).get('initial_bracket_high_ppm',np.inf)+1e-12
    for obj in root_objects
)

dominant_counts=roots_df['dominant_term'].value_counts().to_dict() if len(roots_df) else {}
switch_count=int(roots_df['dominant_switch_from_anchor'].sum()) if len(roots_df) else 0
result={
    'experiment':'PhysBench-GH EXP1.5','model_id':'M1','model':'GreenLight-Gym2',
    'frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
    'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
    'surface_regression':{'unique_root_cells':unique_count,'no_crossing_cells':no_count,'multiple_root_cells':multiple_count,
                          'expected':[EXPECTED_UNIQUE,EXPECTED_NO_ROOT,EXPECTED_MULTIPLE],
                          'anchor_condition':ANCHOR_CONDITION,'anchor_root_ppm':anchor_root,'anchor_abs_error_ppm':anchor_err,
                          'all_roots_inside_detected_brackets':bool(all_roots_inside)},
    'mechanism_summary':{'native_vector_names':VECTOR_NAMES,'anchor_vector_ppm':anchor_vec,
                         'anchor_metrics':vector_metrics(anchor_vec,anchor_vec),
                         'dominant_term_counts_across_root_cells':dominant_counts,
                         'dominant_switch_count_from_anchor':switch_count,
                         'n_root_mechanisms':int(len(roots_df)),'n_edge_audits':int(len(edges_df))},
    'runtime_audit':{'all_finite':bool(all_finite),'runtime_errors':runtime_errors,
                     'max_augmented_vs_native_final_state_abs_diff':float(max_state_err),
                     'max_named_vs_total_ode_closure_mg_m3':float(max_named_total),
                     'max_pair_density_flux_closure_error_mg_m3':float(max_density_pair),
                     'max_final_ppm_decomposition_closure_error':float(max_ppm)},
    'interpretation_limit':'Read-only flux accounting across the frozen EXP1.4 surface. No flux was disabled, repaired, replayed, or retuned.'
}
result['run_pass']=bool(
    all_finite and len(runtime_errors)==0 and max_state_err<=1e-7 and max_named_total<=NAMED_TOTAL_CLOSURE_TOL
    and max_density_pair<=PAIR_DENSITY_CLOSURE_TOL and max_ppm<=PAIR_PPM_CLOSURE_TOL
    and unique_count==EXPECTED_UNIQUE and no_count==EXPECTED_NO_ROOT and multiple_count==EXPECTED_MULTIPLE
    and anchor_err<=ANCHOR_TOL and all_roots_inside
)
OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
env.close()
if not result['run_pass']:
    raise SystemExit('M1 EXP1.5 runtime/surface/closure gate failed')
