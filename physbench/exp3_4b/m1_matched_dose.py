import json, hashlib
from pathlib import Path

import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np
import pandas as pd

from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
A4=ROOT/'evidence/EXP3_4A_FINAL'
GRID_PATH=ROOT/'evidence/EXP3_4A_M1/physical_dose_grid.csv'
TARGET_PATH=A4/'common_physical_dose_targets_for_EXP3_4B.csv'
OUT=ROOT/'evidence/EXP3_4B_M1'; OUT.mkdir(parents=True,exist_ok=True)

EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
EXPECTED_TARGET_SHA='e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292'
MODEL_COMMIT='2d3febb1ea002b24b452e32293e990beb78d3ce1'
U9=np.round(np.arange(0.1,1.0,0.1),1)
Q=np.array([0.0,0.25,0.5,0.75,1.0],dtype=float)
R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def ah_from_t_vp(t,vp):
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

EVENTS=pd.read_csv(IN)
TARGETS=pd.read_csv(TARGET_PATH)
GRID=pd.read_csv(GRID_PATH)
assert sha256(IN)==EXPECTED_INPUT_SHA
assert sha256(TARGET_PATH)==EXPECTED_TARGET_SHA
assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61
assert len(GRID)==61*2*9
assert not GRID.duplicated(['event_id','horizon_min','action']).any()
assert set(np.round(GRID.action.unique(),10))==set(U9)
ELIG=TARGETS[TARGETS.eligible_for_exp3_4b.astype(bool)].copy()
assert len(ELIG)==1060
assert ELIG.groupby(['event_id','horizon_min','coordinate']).size().eq(5).all()

def full_action(v):
    return np.array([0,0,0,float(v),0,0],dtype=np.float32)

def build_weather(r):
    d=np.zeros((3,10),dtype=float)
    d[:,0]=float(r.event_Iglob)
    d[:,1]=float(r.event_Tout)
    d[:,2]=float(r.out_vp_pa)
    d[:,3]=ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed))
    d[:,4]=float(r.event_Windsp)
    d[:,5]=float(r.sky_temperature_c_proxy)
    d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,float(r.event_Iglob))*900.0)/1e6
    d[:,8]=1.0 if float(r.event_Iglob)>0 else 0.0
    d[:,9]=d[:,8]
    return d

env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
NATIVE_X=e.x.copy()
P_NATIVE=np.asarray(e.p,dtype=float).copy()
NX=e.nx; NU=e.nu; ND=e.nd
env.close()

x_sym=ca.SX.sym('x',NX); u_sym=ca.SX.sym('u',NU); d_sym=ca.SX.sym('d',ND); p_sym=ca.SX.sym('p',len(P_NATIVE))
a_sym=aux_states.update(x_sym,u_sym,d_sym,p_sym)
dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
q_ext=a_sym[136]+a_sym[137]+a_sym[145]
FQ900=ca.integrator(
    'EXP3_4B_M1_Q900','cvodes',
    {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym,'quad':q_ext},
    0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':70000}
)
H_EFF=float(P_NATIVE[49])

def initial_state(r):
    x=NATIVE_X.copy()
    x[2]=float(r.event_Tair)
    x[15]=float(r.in_vp_pa)
    x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    x[4]=float(r.event_Tair)
    return x

def evaluate(r,u,horizon):
    h=int(horizon); assert h in (15,30)
    d=build_weather(r); x0=initial_state(r); a=full_action(float(u))
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    ee=env.unwrapped
    ee.weather_data=d.copy()
    ee.day_of_year=int(r.day_of_year); ee.hour_of_day=float(r.hour_decimal)
    ee.x=x0.copy(); ee.x_prev=x0.copy(); ee.obs=ee._get_obs()

    dose=0.0; max_action_err=0.0; max_qstate_err=0.0
    nseg=1 if h==15 else 2
    for k in range(nseg):
        xprev=ee.x.copy()
        _,_,_,truncated,info=env.step(a)
        if truncated: raise RuntimeError((int(r.event_id),u,h,'M1 truncated'))
        applied=np.asarray(info['controls'],dtype=float)
        max_action_err=max(max_action_err,float(np.max(np.abs(applied-a))))
        if not np.all(np.isfinite(ee.x)): raise RuntimeError((int(r.event_id),u,h,'M1 nonfinite native state'))
        qr=FQ900(x0=xprev,u=a.astype(float),p=np.concatenate([d[k],P_NATIVE]))
        xq=np.asarray(qr['xf'].full()).reshape(-1)
        dq=float(np.asarray(qr['qf'].full()).reshape(-1)[0])
        if not (np.all(np.isfinite(xq)) and np.isfinite(dq)):
            raise RuntimeError((int(r.event_id),u,h,'M1 nonfinite dose integration'))
        max_qstate_err=max(max_qstate_err,float(np.max(np.abs(xq-ee.x))))
        dose+=dq
    xf=ee.x.copy(); env.close()
    T=float(xf[2]); VP=float(xf[15])
    return {
      'native_command':float(a[3]),'DV':float(dose),'DN':float(dose/H_EFF),
      'T':T,'AH':float(ah_from_t_vp(T,VP)),
      'action_error':float(max_action_err),'quadrature_state_error':float(max_qstate_err),
      'init_error':float(max(
          abs(float(x0[2])-float(r.event_Tair)),
          abs(float(x0[15])-float(r.in_vp_pa)),
          abs(float(mg_m3_to_ppm(x0[2],x0[0]))-float(r.CO2_pre_ppm)),
          abs(float(x0[4])-float(r.event_Tair))
      ))
    }

EVENT_MAP={int(r.event_id):r for _,r in EVENTS.iterrows()}

def bracket(event_id,horizon,coordinate,target):
    col='specific_volume_dose_m3_m2' if coordinate=='DV' else 'air_volume_equiv'
    g=GRID[(GRID.event_id==event_id)&(GRID.horizon_min==horizon)].sort_values('action')
    assert len(g)==9 and np.allclose(g.action.to_numpy(float),U9,rtol=0,atol=1e-12)
    d=g[col].to_numpy(float); u=g.action.to_numpy(float)
    tolmono=1e-10*max(1.0,float(np.max(np.abs(d))))
    assert np.all(np.diff(d)>=-tolmono)
    target=float(target)
    if target < d[0]-1e-8 or target > d[-1]+1e-8:
        raise AssertionError((event_id,horizon,coordinate,target,d[0],d[-1]))
    idx=int(np.searchsorted(d,target,side='left'))
    if idx<=0: return float(u[0]),float(u[0]),float(d[0]),float(d[0])
    if idx>=len(d): return float(u[-1]),float(u[-1]),float(d[-1]),float(d[-1])
    if abs(target-d[idx])<=1e-12*max(1.0,abs(target)):
        return float(u[idx]),float(u[idx]),float(d[idx]),float(d[idx])
    return float(u[idx-1]),float(u[idx]),float(d[idx-1]),float(d[idx])

def solve(row):
    eid=int(row.event_id); h=int(row.horizon_min); coord=str(row.coordinate); target=float(row.target_dose)
    width=float(row.common_width)
    tol=max(1e-8,1e-6*max(1.0,abs(target)),1e-4*abs(width))
    ulo,uhi,dlo,dhi=bracket(eid,h,coord,target)
    r=EVENT_MAP[eid]
    initial_ulo,initial_uhi=ulo,uhi
    if ulo==uhi:
        res=evaluate(r,ulo,h); n_eval=1
    else:
        if dhi<=dlo: raise AssertionError((eid,h,coord,'nonpositive bracket dose span',dlo,dhi))
        useed=ulo+(target-dlo)*(uhi-ulo)/(dhi-dlo)
        useed=float(np.clip(useed,ulo,uhi))
        res=evaluate(r,useed,h); n_eval=1
        got=float(res[coord])
        best_res=res; best_abs=abs(got-target)
        if best_abs>tol:
            if got<target:
                ulo,dlo=float(res['native_command']),got
            else:
                uhi,dhi=float(res['native_command']),got
            for _ in range(30):
                umid=0.5*(ulo+uhi)
                trial=evaluate(r,umid,h); n_eval+=1
                got=float(trial[coord])
                ae=abs(got-target)
                if ae<best_abs:
                    best_res=trial; best_abs=ae
                if ae<=tol:
                    break
                if got<target:
                    ulo,dlo=float(trial['native_command']),got
                else:
                    uhi,dhi=float(trial['native_command']),got
            res=best_res
            if best_abs>tol:
                raise RuntimeError((eid,h,coord,float(row.q),'dose root did not converge',target,float(res[coord]),tol,ulo,uhi,best_abs))
    achieved=float(res[coord]); err=achieved-target
    if abs(err)>tol:
        raise RuntimeError((eid,h,coord,float(row.q),'dose tolerance fail',target,achieved,err,tol))
    if not (0.1-1e-8 <= res['native_command'] <= 0.9+1e-8):
        raise RuntimeError((eid,h,coord,float(row.q),'command out of domain',res['native_command']))
    return {
      'model_id':'M1','event_id':eid,'team':row.team,'event_date':row.event_date,'daynight':row.daynight,
      'horizon_min':h,'coordinate':coord,'q':float(row.q),
      'target_dose':target,'achieved_dose':achieved,'dose_error':float(err),'dose_tolerance':float(tol),
      'common_width':float(width),'dose_error_fraction_of_common_span':float(abs(err)/width),
      'native_command':float(res['native_command']),
      'initial_bracket_u_lo':float(initial_ulo),'initial_bracket_u_hi':float(initial_uhi),
      'root_model_evaluations':int(n_eval),
      'T':float(res['T']),'AH':float(res['AH']),
      'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
      'action_error':float(res['action_error']),'init_error':float(res['init_error']),
      'quadrature_state_error':float(res['quadrature_state_error'])
    }

rows=[]
for cell_key,g in ELIG.groupby(['event_id','horizon_min','coordinate'],sort=True):
    for _,row in g.sort_values('q').iterrows():
        rows.append(solve(row))
    print('M1 matched cell',cell_key,'done',flush=True)

df=pd.DataFrame(rows).sort_values(['coordinate','horizon_min','event_id','q']).reset_index(drop=True)
assert len(df)==1060
assert not df.duplicated(['event_id','horizon_min','coordinate','q']).any()
assert df.groupby(['event_id','horizon_min','coordinate']).size().eq(5).all()
assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
assert float(df.action_error.max())==0.0
assert float(df.init_error.max())<=1e-8
assert float(df.quadrature_state_error.max())<=1e-7
assert np.all(np.abs(df.dose_error.to_numpy(float))<=df.dose_tolerance.to_numpy(float)+1e-15)
df.to_csv(OUT/'matched_dose_responses.csv',index=False,float_format='%.12g')

counts=df.groupby(['coordinate','horizon_min']).size().to_dict()
summary={
 'experiment':'PhysBench-GH EXP3.4B','model_id':'M1','model':'GreenLight-Gym2',
 'frozen_commit':MODEL_COMMIT,'input_sha256':sha256(IN),'target_sha256':sha256(TARGET_PATH),
 'rows':int(len(df)),
 'counts':{f'{k[0]}_{int(k[1])}':int(v) for k,v in counts.items()},
 'max_abs_dose_error':float(np.abs(df.dose_error).max()),
 'max_dose_error_fraction_of_tolerance':float(np.max(np.abs(df.dose_error)/df.dose_tolerance)),
 'max_dose_error_fraction_of_common_span':float(df.dose_error_fraction_of_common_span.max()),
 'max_action_error':float(df.action_error.max()),
 'max_init_error':float(df.init_error.max()),
 'max_quadrature_state_error':float(df.quadrature_state_error.max()),
 'max_root_model_evaluations':int(df.root_model_evaluations.max()),
 'all_finite':bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all())
}
summary['gate_pass']=bool(
 summary['input_sha256']==EXPECTED_INPUT_SHA and summary['target_sha256']==EXPECTED_TARGET_SHA
 and summary['rows']==1060
 and summary['counts']=={'DN_15':305,'DN_30':305,'DV_15':210,'DV_30':240}
 and summary['all_finite'] and summary['max_action_error']==0.0 and summary['max_init_error']<=1e-8
 and summary['max_quadrature_state_error']<=1e-7 and summary['max_dose_error_fraction_of_tolerance']<=1.0+1e-12
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M1 EXP3.4B gate failed')
