import json
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca

from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

from common import *
from generate_prehistory import generate_prehistory

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
REG=HERE/'exp0_6a_72h_state_hashes.json'
OUT=ROOT/'evidence/EXP0_6B_M1'; OUT.mkdir(parents=True,exist_ok=True)

LOW=float(CFG['low_vent_command']); NEUTRAL=float(CFG['neutral_prehistory_vent_command']); HIGH=float(CFG['high_vent_command'])
HORIZONS=[300,900,1800]
HISTORY_IDS=list(CFG['decision_primary_history_ids'])
SCAN=np.arange(CFG['root_scan']['min_ppm'],CFG['root_scan']['max_ppm']+0.1,CFG['root_scan']['step_ppm'])
WIDTH_TOL=float(CFG['root_scan']['bracket_width_tol_ppm'])
RESP_TOL=float(CFG['root_scan']['response_abs_tol_ppm'])
MAX_IT=int(CFG['root_scan']['max_iterations'])
DEDUP=float(CFG['root_scan']['dedup_tol_ppm'])
ACTIVE_SHARE=0.10

TERM_NAMES=['canopy_net','main_to_top','main_to_outside','co2_injection','blower_source','pad_source']
VECTOR_NAMES=TERM_NAMES+['temperature_conversion']

forcing=pd.read_csv(FORCING)
target=json.loads(INIT.read_text())
reg=json.loads(REG.read_text())

def full_action(v):
    return np.array([0,0,0,float(v),0,0],dtype=np.float32)

def common_state(x):
    x=np.asarray(x,dtype=float)
    return {
      'air_temperature_c':float(x[2]),
      'air_vapor_pressure_pa':float(x[15]),
      'air_rh_pct':float(rh_from_t_vp(x[2],x[15])),
      'air_co2_ppm':float(mg_m3_to_ppm(x[2],x[0])),
      'canopy_temperature_c':float(x[4])
    }

def weather_matrix(df):
    x=df.iloc[:-1].reset_index(drop=True)
    d=np.zeros((len(x),10),dtype=float)
    d[:,0]=x.global_radiation_w_m2
    d[:,1]=x.outdoor_temperature_c
    d[:,2]=x.outdoor_vapor_pressure_pa
    d[:,3]=ppm_to_mg_m3(x.outdoor_temperature_c,x.outdoor_co2_ppm)
    d[:,4]=x.wind_speed_m_s
    d[:,5]=x.sky_temperature_c
    d[:,6]=x.soil_boundary_temperature_c
    dli=[]; acc=0.0; prev=None
    for _,r in x.iterrows():
        h=float(r.local_hour)
        if prev is not None and h < prev-1e-9:
            acc=0.0
        acc += float(r.global_radiation_w_m2)*900.0/1e6
        dli.append(acc); prev=h
    d[:,7]=np.asarray(dli,float)
    d[:,8]=(x.global_radiation_w_m2.to_numpy(float)>0).astype(float)
    d[:,9]=d[:,8]
    return d

def native_reset():
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    x=e.x.copy(); p=np.asarray(e.p,dtype=float).copy()
    env.close()
    return x,p

native_x,p_native=native_reset()

def prehistory_state(hid,duration_h):
    hist=generate_prehistory(hid,duration_h,CFG['anchor_end_local_hour'])
    d=weather_matrix(hist)
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=d.copy()
    e.day_of_year=244.0-float(duration_h)/24.0
    e.hour_of_day=8.0
    errs=[]; finite=True
    for _ in range(len(d)):
        a=full_action(NEUTRAL)
        _,_,_,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        if truncated:
            raise RuntimeError(f'M1 prehistory truncated for {hid} {duration_h}h')
    x=e.x.copy()
    meta={
      'history_id':hid,'duration_h':int(duration_h),'native_state_sha256':state_sha(x),
      'all_states_finite':bool(finite),'max_requested_applied_error':float(max(errs) if errs else 0.0),
      'pre_projection_common':common_state(x),'end_hour_of_day':float(e.hour_of_day),
      'end_day_of_year':float(e.day_of_year),'forcing_rows_used':int(len(d))
    }
    env.close()
    return x,meta

# Frozen native ODE + EXP1.5 read-only CO2 quadratures.
env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
x_sym=ca.SX.sym('x',e.nx); u_sym=ca.SX.sym('u',e.nu); d_sym=ca.SX.sym('d',e.nd); p_sym=ca.SX.sym('p',len(p_native))
a_sym=aux_states.update(x_sym,u_sym,d_sym,p_sym)
dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
q_expr=[
    -a_sym[216]/p_sym[122],
    -a_sym[217]/p_sym[122],
    -a_sym[219]/p_sym[122],
    a_sym[222]/p_sym[122],
    a_sym[223]/p_sym[122],
    a_sym[224]/p_sym[122],
]
q_terms=ca.vertcat(*(q_expr+[dx_sym[0]]))
F_BY_H={}; FQ900=None
for h in HORIZONS:
    F_BY_H[h]=ca.integrator(
        f'EXP0_6B_M1_H{h}','cvodes',
        {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym},
        0.0,float(h),{'abstol':1e-4,'reltol':1e-4,'max_num_steps':150000})
FQ900=ca.integrator(
    'EXP0_6B_M1_Q900','cvodes',
    {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym,'quad':q_terms},
    0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':150000})
env.close()

def intervention_d():
    r=forcing.iloc[0]
    d=np.zeros(10,dtype=float)
    d[0]=r.global_radiation_w_m2; d[1]=r.outdoor_temperature_c; d[2]=r.outdoor_vapor_pressure_pa
    d[3]=ppm_to_mg_m3(r.outdoor_temperature_c,r.outdoor_co2_ppm)
    d[4]=r.wind_speed_m_s; d[5]=r.sky_temperature_c; d[6]=r.soil_boundary_temperature_c
    d[7]=float(r.global_radiation_w_m2*900.0/1e6); d[8]=1.0; d[9]=1.0
    return d
D0=intervention_d(); COMBINED=np.concatenate([D0,p_native])

def projected_x(latent,c0):
    z=np.asarray(latent,dtype=float).copy()
    z[2]=float(target['air_temperature_c'])
    z[15]=float(target['air_vapor_pressure_pa'])
    z[4]=float(target['canopy_temperature_c'])
    z[0]=float(ppm_to_mg_m3(z[2],float(c0)))
    return z

arm_cache={}
def native_arm(h,latent,c0,vent,cache_key):
    key=(int(h),cache_key,round(float(c0),10),float(vent))
    if key in arm_cache: return arm_cache[key]
    z=projected_x(latent,c0)
    r=F_BY_H[int(h)](x0=z,u=full_action(vent).astype(float),p=COMBINED)
    xf=np.asarray(r['xf'].full()).reshape(-1)
    if not np.all(np.isfinite(xf)): raise RuntimeError(f'non-finite M1 state {key}')
    out={'xf':xf,'final_co2_ppm':float(mg_m3_to_ppm(xf[2],xf[0]))}
    arm_cache[key]=out
    return out

def response(h,latent,c0,cache_key):
    lo=native_arm(h,latent,c0,LOW,cache_key); hi=native_arm(h,latent,c0,HIGH,cache_key)
    return float(hi['final_co2_ppm']-lo['final_co2_ppm'])

def refine(h,latent,cache_key,lo,hi,flo,fhi):
    ilo,ihi=float(lo),float(hi)
    if abs(flo)<=RESP_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if abs(fhi)<=RESP_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if flo*fhi>0: raise AssertionError('invalid bracket')
    for it in range(1,MAX_IT+1):
        mid=.5*(lo+hi); fm=response(h,latent,mid,cache_key)
        if abs(fm)<=RESP_TOL or (hi-lo)<=WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
        if flo*fm<=0: hi,fhi=mid,fm
        else: lo,flo=mid,fm
    mid=.5*(lo+hi); fm=response(h,latent,mid,cache_key)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':MAX_IT,'final_bracket_width_ppm':float(hi-lo),
            'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,
            'converged':bool((hi-lo)<=WIDTH_TOL or abs(fm)<=RESP_TOL)}

def scan_root(h,latent,cache_key):
    ys=[response(h,latent,float(c0),cache_key) for c0 in SCAN]
    candidates=[]
    for i,y in enumerate(ys):
        if abs(y)<=RESP_TOL: candidates.append(refine(h,latent,cache_key,SCAN[i],SCAN[i],y,y))
    for i in range(len(SCAN)-1):
        if ys[i]*ys[i+1]<0:
            candidates.append(refine(h,latent,cache_key,SCAN[i],SCAN[i+1],ys[i],ys[i+1]))
    roots=dedup_roots(candidates,DEDUP)
    return ys,roots

def decomp_arm(latent,c0,vent,cache_key):
    z=projected_x(latent,c0)
    r=FQ900(x0=z,u=full_action(vent).astype(float),p=COMBINED)
    xf=np.asarray(r['xf'].full()).reshape(-1); q=np.asarray(r['qf'].full()).reshape(-1)
    nat=native_arm(900,latent,c0,vent,cache_key)
    state_err=float(np.max(np.abs(xf-nat['xf'])))
    ints={TERM_NAMES[i]:float(q[i]) for i in range(len(TERM_NAMES))}
    qsum=float(sum(ints.values())); qtotal=float(q[-1]); density_delta=float(xf[0]-z[0])
    return {
      'final_density_mg_m3':float(xf[0]),'final_air_temperature_c':float(xf[2]),
      'final_co2_ppm':float(mg_m3_to_ppm(xf[2],xf[0])),
      'integrated_density_contributions_mg_m3':ints,
      'integrated_total_ode_density_change_mg_m3':qtotal,
      'sum_named_density_contributions_mg_m3':qsum,
      'actual_density_change_mg_m3':density_delta,
      'named_vs_total_ode_closure_mg_m3':float(qsum-qtotal),
      'total_ode_vs_actual_closure_mg_m3':float(qtotal-density_delta),
      'augmented_vs_native_final_state_max_abs_diff':state_err,
      'finite':bool(np.all(np.isfinite(xf)) and np.all(np.isfinite(q)))
    }

def paired_mechanism(latent,c0,cache_key):
    lo=decomp_arm(latent,c0,LOW,cache_key); hi=decomp_arm(latent,c0,HIGH,cache_key)
    CH,CL=hi['final_density_mg_m3'],lo['final_density_mg_m3']
    TH,TL=hi['final_air_temperature_c'],lo['final_air_temperature_c']
    kH=float(R*(TH+K)/(P*MCO2)); kL=float(R*(TL+K)/(P*MCO2))
    kbar=.5*(kH+kL); Cbar=.5*(CH+CL)
    v={}; density_parts={}
    for name in TERM_NAMES:
        dq=float(hi['integrated_density_contributions_mg_m3'][name]-lo['integrated_density_contributions_mg_m3'][name])
        density_parts[name]=dq; v[name]=float(kbar*dq)
    v['temperature_conversion']=float(Cbar*(kH-kL))
    actual=float(hi['final_co2_ppm']-lo['final_co2_ppm'])
    reconstructed=float(sum(v.values()))
    eps=float(actual-reconstructed)
    vals=np.asarray([v[k] for k in VECTOR_NAMES],dtype=float)
    order=np.argsort(-np.abs(vals)); top=float(abs(vals[order[0]])); second=float(abs(vals[order[1]]))
    resolved=bool((top-second)>abs(eps))
    return {'LOW':lo,'HIGH':hi,'vector':v,'actual_high_minus_low_ppm':actual,
            'reconstructed_ppm':reconstructed,'numerical_closure_correction_ppm':eps,
            'dominant_resolved_under_closure_bound':resolved}

def vector_metrics(v,anchor_v,eps,resolved):
    arr=np.asarray([v[k] for k in VECTOR_NAMES],dtype=float); l1=float(np.sum(np.abs(arr)))
    norm=arr/l1 if l1>0 else np.zeros_like(arr); idx=int(np.argmax(np.abs(arr)))
    aa=np.asarray([anchor_v[k] for k in VECTOR_NAMES],dtype=float); al1=float(np.sum(np.abs(aa)))
    an=aa/al1 if al1>0 else np.zeros_like(aa)
    den=float(np.linalg.norm(norm)*np.linalg.norm(an))
    shared=[i for i in range(len(VECTOR_NAMES)) if abs(norm[i])>=ACTIVE_SHARE and abs(an[i])>=ACTIVE_SHARE]
    flips=[VECTOR_NAMES[i] for i in shared if np.sign(norm[i])!=np.sign(an[i])]
    out={
      'dominant_term':VECTOR_NAMES[idx] if l1>0 else 'NONE',
      'dominance_share':float(np.max(np.abs(norm))) if l1>0 else 0.0,
      'dominant_resolved_under_closure_bound':bool(resolved),
      'numerical_closure_correction_ppm':float(eps),
      'active_terms':'|'.join(VECTOR_NAMES[i] for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
      'active_sign_pattern':'|'.join(f"{VECTOR_NAMES[i]}:{'+' if x>0 else '-'}" for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
      'anchor_cosine_similarity':float(np.dot(norm,an)/den) if den>0 else float('nan'),
      'anchor_l1_distance':float(np.sum(np.abs(norm-an))),
      'dominant_switch_from_hnr':bool(VECTOR_NAMES[idx] != VECTOR_NAMES[int(np.argmax(np.abs(aa)))]) if l1>0 and al1>0 else False,
      'active_term_sign_flips_vs_hnr':'|'.join(flips),
      'n_active_term_sign_flips_vs_hnr':int(len(flips)),
      'l1_total_abs_ppm':l1
    }
    for i,k in enumerate(VECTOR_NAMES): out[f'norm_{k}']=float(norm[i])
    return out

# Generate latent states.
states={('HNR',0):native_x.copy()}
meta=[{'history_id':'HNR','duration_h':0,'native_state_sha256':state_sha(native_x),'all_states_finite':True,
       'max_requested_applied_error':0.0,'pre_projection_common':common_state(native_x)}]
for dur in [72,168]:
    for hid in HISTORY_IDS:
        x,m=prehistory_state(hid,dur)
        if dur==72:
            exp=reg['M1'][hid]
            if state_sha(x)!=exp: raise AssertionError(f'M1 72h state hash drift {hid}: {state_sha(x)} != {exp}')
        states[(hid,dur)]=x; meta.append(m)
np.savez_compressed(OUT/'latent_states_72_168.npz',**{f'{hid}_{dur}h':v for (hid,dur),v in states.items()})

root_rows=[]; scan_rows=[]
# HNR at all horizons.
for h in HORIZONS:
    ys,roots=scan_root(h,native_x,f'HNR_{h}')
    for c0,y in zip(SCAN,ys):
        scan_rows.append({'model_id':'M1','duration_h':0,'history_id':'HNR','horizon_s':h,'initial_co2_ppm':float(c0),'response_ppm':float(y)})
    unique=roots[0]['root_ppm'] if len(roots)==1 else np.nan
    root_rows.append({'model_id':'M1','duration_h':0,'history_id':'HNR','horizon_s':h,'root_count':len(roots),
                      'primary_boundary_ppm':unique,'all_roots_converged':bool(all(r['converged'] for r in roots)),
                      'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))})
# 72h at 3 horizons; 168h at H900.
for dur,horizons in [(72,HORIZONS),(168,[900])]:
    for hid in HISTORY_IDS:
        latent=states[(hid,dur)]
        for h in horizons:
            ys,roots=scan_root(h,latent,f'{hid}_{dur}_{h}')
            for c0,y in zip(SCAN,ys):
                scan_rows.append({'model_id':'M1','duration_h':dur,'history_id':hid,'horizon_s':h,'initial_co2_ppm':float(c0),'response_ppm':float(y)})
            unique=roots[0]['root_ppm'] if len(roots)==1 else np.nan
            root_rows.append({'model_id':'M1','duration_h':dur,'history_id':hid,'horizon_s':h,'root_count':len(roots),
                              'primary_boundary_ppm':unique,'all_roots_converged':bool(all(r['converged'] for r in roots)),
                              'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))})
            print('M1',dur,hid,h,'root',unique,'n',len(roots),flush=True)

rdf=pd.DataFrame(root_rows); sdf=pd.DataFrame(scan_rows)

# H900 mechanism anchor from HNR.
hnr900=rdf[(rdf.duration_h==0)&(rdf.history_id=='HNR')&(rdf.horizon_s==900)].iloc[0]
if int(hnr900.root_count)!=1: raise AssertionError('M1 HNR H900 root not unique')
anchor_pair=paired_mechanism(native_x,float(hnr900.primary_boundary_ppm),'HNR_900')
anchor_v=anchor_pair['vector']

mechanism_rows=[]; term_rows=[]
def add_mechanism(duration_h,hid,latent,root):
    pair=paired_mechanism(latent,float(root),f'{hid}_{duration_h}_900')
    v=pair['vector']; m=vector_metrics(v,anchor_v,pair['numerical_closure_correction_ppm'],pair['dominant_resolved_under_closure_bound'])
    max_state=max(pair['LOW']['augmented_vs_native_final_state_max_abs_diff'],pair['HIGH']['augmented_vs_native_final_state_max_abs_diff'])
    max_named=max(abs(pair['LOW']['named_vs_total_ode_closure_mg_m3']),abs(pair['HIGH']['named_vs_total_ode_closure_mg_m3']))
    row={'model_id':'M1','duration_h':duration_h,'history_id':hid,'root_ppm':float(root),
         'actual_high_minus_low_ppm':pair['actual_high_minus_low_ppm'],
         'max_augmented_vs_native_state_error':float(max_state),
         'max_named_vs_total_ode_closure_mg_m3':float(max_named),**m}
    mechanism_rows.append(row)
    for k in VECTOR_NAMES:
        term_rows.append({'model_id':'M1','duration_h':duration_h,'history_id':hid,'root_ppm':float(root),'term':k,'contribution_ppm':float(v[k])})

# Include HNR mechanism row for explicit anchor.
add_mechanism(0,'HNR',native_x,float(hnr900.primary_boundary_ppm))
for dur in [72,168]:
    for hid in HISTORY_IDS:
        rr=rdf[(rdf.duration_h==dur)&(rdf.history_id==hid)&(rdf.horizon_s==900)].iloc[0]
        if int(rr.root_count)==1:
            add_mechanism(dur,hid,states[(hid,dur)],float(rr.primary_boundary_ppm))

mdf=pd.DataFrame(mechanism_rows); tdf=pd.DataFrame(term_rows)

# Regression gates.
hnr_err=abs(float(hnr900.primary_boundary_ppm)-float(CFG['hnr_reference']['M1_root_ppm']))
reg_errors=[]
for hid in HISTORY_IDS:
    rr=rdf[(rdf.duration_h==72)&(rdf.history_id==hid)&(rdf.horizon_s==900)].iloc[0]
    if int(rr.root_count)!=1:
        reg_errors.append(float('inf'))
    else:
        reg_errors.append(abs(float(rr.primary_boundary_ppm)-float(reg['M1_H900_root_ppm'][hid])))
max_reg_err=float(max(reg_errors))
all_mech_finite=bool(np.isfinite(mdf.select_dtypes(include=[np.number]).to_numpy()).all())
max_state_err=float(mdf.max_augmented_vs_native_state_error.max())
max_named_err=float(mdf.max_named_vs_total_ode_closure_mg_m3.max())

rdf.to_csv(OUT/'root_results.csv',index=False,float_format='%.12g')
sdf.to_csv(OUT/'root_scan.csv',index=False,float_format='%.12g')
mdf.to_csv(OUT/'mechanism_results.csv',index=False,float_format='%.12g')
tdf.to_csv(OUT/'mechanism_term_contributions.csv',index=False,float_format='%.12g')
(OUT/'latent_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

summary={
 'experiment':'PhysBench-GH EXP0.6B','model_id':'M1','model':'GreenLight-Gym2',
 'frozen_commit':CFG['frozen_models']['M1'],'history_ids':HISTORY_IDS,
 'hnr_h900_root_ppm':float(hnr900.primary_boundary_ppm),'hnr_h900_abs_error_ppm':float(hnr_err),
 'max_72h_h900_root_regression_error_ppm':max_reg_err,
 'all_root_scan_values_finite':bool(np.isfinite(sdf.response_ppm.to_numpy(float)).all()),
 'all_detected_roots_converged':bool(rdf.all_roots_converged.all()),
 'all_detected_roots_inside_bracket':bool(rdf.all_roots_inside_bracket.all()),
 'all_mechanism_outputs_finite':all_mech_finite,
 'max_mechanism_augmented_vs_native_state_error':max_state_err,
 'max_mechanism_named_vs_total_ode_closure_mg_m3':max_named_err,
 'mechanism_rows':int(len(mdf)),
 'mechanism_resolved_rows':int(mdf.dominant_resolved_under_closure_bound.sum()),
 'prehistory_all_finite':bool(all(x['all_states_finite'] for x in meta)),
 'prehistory_max_action_error':float(max(x['max_requested_applied_error'] for x in meta))
}
summary['gate_pass']=bool(
    summary['hnr_h900_abs_error_ppm']<=float(CFG['hnr_reference']['root_abs_tol_ppm'])
    and summary['max_72h_h900_root_regression_error_ppm']<=float(reg['root_regression_tolerance_ppm'])
    and summary['all_root_scan_values_finite'] and summary['all_detected_roots_converged']
    and summary['all_detected_roots_inside_bracket'] and summary['all_mechanism_outputs_finite']
    and summary['max_mechanism_augmented_vs_native_state_error']<=1e-7
    and summary['max_mechanism_named_vs_total_ode_closure_mg_m3']<=1e-10
    and summary['prehistory_all_finite'] and summary['prehistory_max_action_error']==0.0
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M1 EXP0.6B runtime/reconstruction gate failed')
