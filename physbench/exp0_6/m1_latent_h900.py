import json
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca
from gl_gym.models.GreenLight.ode import ODE

from common import *
from generate_prehistory import generate_prehistory

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
OUT=ROOT/'evidence/EXP0_6A_M1'; OUT.mkdir(parents=True,exist_ok=True)

LOW=float(CFG['low_vent_command']); NEUTRAL=float(CFG['neutral_prehistory_vent_command']); HIGH=float(CFG['high_vent_command'])
SCAN=np.arange(CFG['root_scan']['min_ppm'],CFG['root_scan']['max_ppm']+0.1,CFG['root_scan']['step_ppm'])
WIDTH_TOL=float(CFG['root_scan']['bracket_width_tol_ppm'])
RESP_TOL=float(CFG['root_scan']['response_abs_tol_ppm'])
MAX_IT=int(CFG['root_scan']['max_iterations'])
DEDUP=float(CFG['root_scan']['dedup_tol_ppm'])

forcing=pd.read_csv(FORCING)
target=json.loads(INIT.read_text())
assert sha256(ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv')=='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'

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
    # One disturbance vector per 900-s interval. The final history row is the intervention origin.
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
        dli.append(acc)
        prev=h
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

def prehistory_state(hid):
    hist=generate_prehistory(hid,CFG['primary_prehistory_h'],CFG['anchor_end_local_hour'])
    d=weather_matrix(hist)
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=d.copy()
    e.day_of_year=241.0
    e.hour_of_day=8.0
    errs=[]; finite=True
    for _ in range(len(d)):
        a=full_action(NEUTRAL)
        obs,reward,terminated,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        if truncated:
            raise RuntimeError(f'M1 prehistory truncated for {hid}')
    x=e.x.copy()
    end_hour=float(e.hour_of_day); end_day=float(e.day_of_year)
    env.close()
    return x,{
      'history_id':hid,'native_state_sha256':state_sha(x),'all_states_finite':bool(finite),
      'max_requested_applied_error':float(max(errs) if errs else 0.0),
      'pre_projection_common':common_state(x),'end_hour_of_day':end_hour,'end_day_of_year':end_day,
      'forcing_rows_used':int(len(d))
    }

# Native H900 intervention integrator, identical ODE/tolerances to EXP1.4.
env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
x_sym=ca.SX.sym('x',e.nx); u_sym=ca.SX.sym('u',e.nu); d_sym=ca.SX.sym('d',e.nd); p_sym=ca.SX.sym('p',len(p_native))
dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
F900=ca.integrator('EXP0_6A_M1_H900','cvodes',
    {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym},
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
D0=intervention_d()
COMBINED=np.concatenate([D0,p_native])

def projected_x(latent,c0):
    z=np.asarray(latent,dtype=float).copy()
    z[2]=float(target['air_temperature_c'])
    z[15]=float(target['air_vapor_pressure_pa'])
    z[4]=float(target['canopy_temperature_c'])
    z[0]=float(ppm_to_mg_m3(z[2],float(c0)))
    return z

def run_arm(latent,c0,vent):
    z=projected_x(latent,c0)
    res=F900(x0=z,u=full_action(vent).astype(float),p=COMBINED)
    xf=np.asarray(res['xf'].full()).reshape(-1)
    if not np.all(np.isfinite(xf)):
        raise RuntimeError('non-finite M1 intervention state')
    T=float(xf[2]); VP=float(xf[15])
    return {
      'T':T,'VP':VP,'AH':float(ah_g_m3_from_vp_pa(T,VP)),
      'CO2':float(mg_m3_to_ppm(T,xf[0]))
    }

def response(latent,c0):
    lo=run_arm(latent,c0,LOW); hi=run_arm(latent,c0,HIGH)
    return float(hi['CO2']-lo['CO2'])

def refine(latent,lo,hi,flo,fhi):
    ilo,ihi=float(lo),float(hi)
    if abs(flo)<=RESP_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if abs(fhi)<=RESP_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if flo*fhi>0: raise AssertionError('invalid bracket')
    for it in range(1,MAX_IT+1):
        mid=0.5*(lo+hi); fm=response(latent,mid)
        if abs(fm)<=RESP_TOL or (hi-lo)<=WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
        if flo*fm<=0: hi,fhi=mid,fm
        else: lo,flo=mid,fm
    mid=0.5*(lo+hi); fm=response(latent,mid)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':MAX_IT,'final_bracket_width_ppm':float(hi-lo),
            'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,
            'converged':bool((hi-lo)<=WIDTH_TOL or abs(fm)<=RESP_TOL)}

latent_states={'HNR':native_x.copy()}
meta=[{
  'history_id':'HNR','native_state_sha256':state_sha(native_x),'all_states_finite':True,
  'max_requested_applied_error':0.0,'pre_projection_common':common_state(native_x),
  'end_hour_of_day':None,'end_day_of_year':None,'forcing_rows_used':0
}]
for hid in CFG['anchor_primary_history_ids']:
    x,m=prehistory_state(hid); latent_states[hid]=x; meta.append(m)

# Save exact native latent vectors for audit/reuse.
np.savez_compressed(OUT/'latent_states.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in latent_states.items()})

scan_rows=[]; root_rows=[]; summary_rows=[]
for hid in ['HNR']+CFG['anchor_primary_history_ids']:
    latent=latent_states[hid]
    pre=common_state(latent)
    p500=projected_x(latent,500.0)
    post=common_state(p500)
    correction={k:float(post[k]-pre[k]) for k in ['air_temperature_c','air_vapor_pressure_pa','air_co2_ppm','canopy_temperature_c']}
    ys=[]
    for c0 in SCAN:
        y=response(latent,float(c0)); ys.append(y)
        scan_rows.append({'model_id':'M1','history_id':hid,'initial_co2_ppm':float(c0),'high_minus_low_final_co2_ppm':y})
    candidates=[]
    for i,y in enumerate(ys):
        if abs(y)<=RESP_TOL: candidates.append(refine(latent,SCAN[i],SCAN[i],y,y))
    for i in range(len(SCAN)-1):
        if ys[i]*ys[i+1]<0: candidates.append(refine(latent,SCAN[i],SCAN[i+1],ys[i],ys[i+1]))
    roots=dedup_roots(candidates,DEDUP)
    unique=roots[0]['root_ppm'] if len(roots)==1 else None
    for j,r in enumerate(roots):
        root_rows.append({'model_id':'M1','history_id':hid,'root_index':j,**r})
    lo500=run_arm(latent,500.0,LOW); hi500=run_arm(latent,500.0,HIGH)
    row={
      'model_id':'M1','history_id':hid,'history_derived':hid!='HNR',
      'native_state_sha256':state_sha(latent),'root_count':len(roots),'primary_boundary_ppm':unique,
      'delta_T_C_at_500ppm':float(hi500['T']-lo500['T']),
      'delta_AH_g_m3_at_500ppm':float(hi500['AH']-lo500['AH']),
      'delta_CO2_ppm_at_500ppm':float(hi500['CO2']-lo500['CO2']),
      'projection_T_C':correction['air_temperature_c'],'projection_VP_Pa':correction['air_vapor_pressure_pa'],
      'projection_CO2_ppm_at_500':correction['air_co2_ppm'],'projection_canopy_T_C':correction['canopy_temperature_c'],
      'all_roots_converged':bool(all(r['converged'] for r in roots)),
      'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))
    }
    summary_rows.append(row)
    print(hid,'root=',unique,'n=',len(roots),'dCO2@500=',row['delta_CO2_ppm_at_500ppm'],flush=True)

md={m['history_id']:m for m in meta}
for row in summary_rows:
    row['prehistory_finite']=bool(md[row['history_id']]['all_states_finite'])
    row['prehistory_max_action_error']=float(md[row['history_id']]['max_requested_applied_error'])

sdf=pd.DataFrame(summary_rows)
pd.DataFrame(scan_rows).to_csv(OUT/'boundary_scan.csv',index=False,float_format='%.12g')
pd.DataFrame(root_rows).to_csv(OUT/'refined_roots.csv',index=False,float_format='%.12g')
sdf.to_csv(OUT/'history_summary.csv',index=False,float_format='%.12g')
(OUT/'latent_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

hnr=sdf[sdf.history_id=='HNR'].iloc[0]
hnr_err=abs(float(hnr.primary_boundary_ppm)-float(CFG['hnr_reference']['M1_root_ppm'])) if int(hnr.root_count)==1 else float('inf')
hist=sdf[sdf.history_derived]
audit={
 'experiment':'PhysBench-GH EXP0.6A','model_id':'M1','model':'GreenLight-Gym2',
 'frozen_commit':CFG['frozen_models']['M1'],'history_count':int(len(hist)),
 'hnr_root_ppm':None if int(hnr.root_count)!=1 else float(hnr.primary_boundary_ppm),
 'hnr_root_abs_error_ppm':float(hnr_err),
 'all_prehistory_finite':bool(hist.prehistory_finite.all()),
 'max_prehistory_action_error':float(hist.prehistory_max_action_error.max()),
 'all_root_scans_finite':bool(np.isfinite(pd.DataFrame(scan_rows).high_minus_low_final_co2_ppm.to_numpy(float)).all()),
 'all_roots_converged':bool(sdf.all_roots_converged.all()),
 'all_roots_inside_bracket':bool(sdf.all_roots_inside_bracket.all())
}
audit['gate_pass']=bool(
    audit['history_count']==17 and audit['all_prehistory_finite'] and audit['max_prehistory_action_error']==0.0
    and audit['all_root_scans_finite'] and audit['all_roots_converged'] and audit['all_roots_inside_bracket']
    and audit['hnr_root_abs_error_ppm']<=float(CFG['hnr_reference']['root_abs_tol_ppm'])
)
(OUT/'summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
if not audit['gate_pass']: raise SystemExit('M1 EXP0.6A runtime/reconstruction gate failed')
