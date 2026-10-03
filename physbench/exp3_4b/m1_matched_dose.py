import json
from pathlib import Path

import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np
import pandas as pd

from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states
from common import (
    INPUT_SHA256,M1_COMMIT,COORD_COL,sha256,validate_target_file,solve_target
)

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
TARGETS_PATH=ROOT/'evidence/EXP3_4A_FINAL/common_physical_dose_targets_for_EXP3_4B.csv'
GRID_PATH=ROOT/'evidence/EXP3_4A_M1/physical_dose_grid.csv'
LOCK_SUMMARY=ROOT/'evidence/EXP3_4A_M1/summary.json'
OUT=ROOT/'evidence/EXP3_4B_M1'; OUT.mkdir(parents=True,exist_ok=True)

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

def ppm_to_mg_m3(t,ppm): return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))
def mg_m3_to_ppm(t,mg): return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)
def locked_exp31_ah(t,vp): return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

def canonical_u(v): return float(np.float32(v))
def full_action(v): return np.array([0,0,0,canonical_u(v),0,0],dtype=np.float32)

assert sha256(IN)==INPUT_SHA256
EVENTS=pd.read_csv(IN)
assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61
TARGETS=validate_target_file(TARGETS_PATH)
GRID=pd.read_csv(GRID_PATH)
LOCK=json.loads(LOCK_SUMMARY.read_text())
assert LOCK['gate_pass'] and LOCK['frozen_commit']==M1_COMMIT and LOCK['input_sha256']==INPUT_SHA256
assert len(GRID)==61*2*9 and not GRID.duplicated(['event_id','horizon_min','action']).any()

# Frozen native reset / parameters, exactly inherited from EXP3.4A.
env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
NATIVE_X=e.x.copy(); P_NATIVE=np.asarray(e.p,dtype=float).copy(); NX=e.nx; NU=e.nu; ND=e.nd
env.close()

x_sym=ca.SX.sym('x',NX); u_sym=ca.SX.sym('u',NU); d_sym=ca.SX.sym('d',ND); p_sym=ca.SX.sym('p',len(P_NATIVE))
a_sym=aux_states.update(x_sym,u_sym,d_sym,p_sym); dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
q_ext=a_sym[136]+a_sym[137]+a_sym[145]
FQ900=ca.integrator('EXP3_4B_M1_Q900','cvodes',
    {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym,'quad':q_ext},
    0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':70000})
H_EFF=float(P_NATIVE[49])

def build_weather(r):
    d=np.zeros((3,10),dtype=float)
    d[:,0]=float(r.event_Iglob); d[:,1]=float(r.event_Tout); d[:,2]=float(r.out_vp_pa)
    d[:,3]=ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed))
    d[:,4]=float(r.event_Windsp); d[:,5]=float(r.sky_temperature_c_proxy); d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,float(r.event_Iglob))*900.0)/1e6
    d[:,8]=1.0 if float(r.event_Iglob)>0 else 0.0; d[:,9]=d[:,8]
    return d

def initial_state(r):
    x=NATIVE_X.copy(); x[2]=float(r.event_Tair); x[15]=float(r.in_vp_pa)
    x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm))); x[4]=float(r.event_Tair)
    return x

CACHE={}
def simulate(r,u_raw):
    u=canonical_u(u_raw); key=(int(r.event_id),u)
    if key in CACHE: return CACHE[key].copy()
    d=build_weather(r); x0=initial_state(r); a=full_action(u)
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    ee=env.unwrapped; ee.weather_data=d.copy(); ee.day_of_year=int(r.day_of_year); ee.hour_of_day=float(r.hour_decimal)
    ee.x=x0.copy(); ee.x_prev=x0.copy(); ee.obs=ee._get_obs()
    _,_,_,truncated,info=env.step(a)
    if truncated: raise RuntimeError((int(r.event_id),u,'M1 trajectory truncated'))
    applied=np.asarray(info['controls'],dtype=float); xf=ee.x.copy(); env.close()
    qres=FQ900(x0=x0,u=a.astype(float),p=np.concatenate([d[0],P_NATIVE]))
    xq=np.asarray(qres['xf'].full()).reshape(-1); dv=float(np.asarray(qres['qf'].full()).reshape(-1)[0]); dn=float(dv/H_EFF)
    init_err=max(abs(float(x0[2])-float(r.event_Tair)),abs(float(x0[15])-float(r.in_vp_pa)),
                 abs(float(mg_m3_to_ppm(x0[2],x0[0]))-float(r.CO2_pre_ppm)),abs(float(x0[4])-float(r.event_Tair)))
    rec={
      'model_id':'M1','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,
      'horizon_min':15,'native_command':u,
      'specific_volume_dose_m3_m2':dv,'air_volume_equiv':dn,
      'T':float(xf[2]),'AH':float(locked_exp31_ah(float(xf[2]),float(xf[15]))),
      'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
      'init_max_abs_error':float(init_err),'action_max_abs_error':float(np.max(np.abs(applied-a))),
      'quadrature_vs_native_state_max_abs_error':float(np.max(np.abs(xq-xf)))
    }
    vals=np.array([rec[k] for k in ['native_command','specific_volume_dose_m3_m2','air_volume_equiv','T','AH','init_max_abs_error','action_max_abs_error','quadrature_vs_native_state_max_abs_error']],float)
    if not np.all(np.isfinite(vals)): raise RuntimeError((int(r.event_id),u,'nonfinite M1 matched run'))
    if rec['init_max_abs_error']>1e-8 or rec['action_max_abs_error']!=0.0 or rec['quadrature_vs_native_state_max_abs_error']>1e-7:
        raise RuntimeError((int(r.event_id),u,'M1 runtime gate',rec))
    CACHE[key]=rec.copy(); return rec

rows=[]
use=TARGETS[(TARGETS.horizon_min==15)&TARGETS.coordinate.isin(['DV','DN'])&TARGETS.eligible_for_exp3_4b].copy()
for eid,tt in use.groupby('event_id',sort=True):
    r=EVENTS[EVENTS.event_id==eid].iloc[0]
    gg=GRID[(GRID.event_id==eid)&(GRID.horizon_min==15)].copy()
    if len(gg)!=9: raise AssertionError(('missing M1 U9 grid',eid))
    for _,t in tt.sort_values(['coordinate','q']).iterrows():
        rec,nit,blo,bhi,tol=solve_target(gg,str(t.coordinate),float(t.target_dose),lambda u,rr=r:simulate(rr,u))
        actual=float(rec[COORD_COL[str(t.coordinate)]])
        err=abs(actual-float(t.target_dose))
        if err>tol: raise AssertionError(('M1 dose gate',eid,t.coordinate,t.q,err,tol))
        rows.append({**rec,'coordinate':str(t.coordinate),'q':float(t.q),'target_dose':float(t.target_dose),
          'actual_dose':actual,'dose_abs_error':err,'dose_tolerance':tol,'dose_match_pass':True,
          'common_low':float(t.common_low),'common_high':float(t.common_high),'common_width':float(t.common_width),
          'locked_bracket_u_lo':blo,'locked_bracket_u_hi':bhi,'inversion_evaluations':int(nit)})
    print('M1 event',int(eid),'done',flush=True)

df=pd.DataFrame(rows).sort_values(['coordinate','event_id','q']).reset_index(drop=True)
assert len(df)==515 and not df.duplicated(['coordinate','event_id','q']).any()
assert df[df.coordinate=='DV'].event_id.nunique()==42 and df[df.coordinate=='DN'].event_id.nunique()==61
assert np.all(df.native_command.to_numpy(float)>=0.1-1e-12) and np.all(df.native_command.to_numpy(float)<=0.9+1e-12)
assert bool(df.dose_match_pass.all()) and np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
df.to_csv(OUT/'matched_physical_dose_outcomes.csv',index=False,float_format='%.12g')
summary={
 'experiment':'PhysBench-GH EXP3.4B','model_id':'M1','frozen_commit':M1_COMMIT,'input_sha256':sha256(IN),
 'target_sha256':sha256(TARGETS_PATH),'rows':int(len(df)),'dv15_events':int(df[df.coordinate=='DV'].event_id.nunique()),
 'dn15_events':int(df[df.coordinate=='DN'].event_id.nunique()),'max_DV_abs_error':float(df[df.coordinate=='DV'].dose_abs_error.max()),
 'max_DN_abs_error':float(df[df.coordinate=='DN'].dose_abs_error.max()),'max_action_error':float(df.action_max_abs_error.max()),
 'max_init_error':float(df.init_max_abs_error.max()),'max_quadrature_vs_native_state_error':float(df.quadrature_vs_native_state_max_abs_error.max()),
 'max_inversion_evaluations':int(df.inversion_evaluations.max()),'all_finite':True,'gate_pass':True
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
