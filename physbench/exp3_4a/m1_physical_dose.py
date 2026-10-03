import json
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
REF=ROOT/'evidence/M1/action_grid_responses.csv'
OUT=ROOT/'evidence/EXP3_4A_M1'
OUT.mkdir(parents=True,exist_ok=True)

EVENTS=pd.read_csv(IN)
U9=np.round(np.arange(0.1,1.0,0.1),1)
U5=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
MODEL_COMMIT='2d3febb1ea002b24b452e32293e990beb78d3ce1'

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def locked_exp31_ah(t,vp):
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

assert sha256(IN)==EXPECTED_INPUT_SHA
assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61

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

# Frozen native reset/parameters, identical to EXP3.1.
env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
NATIVE_X=e.x.copy()
P_NATIVE=np.asarray(e.p,dtype=float).copy()
NX=e.nx; NU=e.nu; ND=e.nd
env.close()

# Physical-source lock:
# q_ext = fVentRoof + fVentSide + fVentForced = a[136]+a[137]+a[145]
x_sym=ca.SX.sym('x',NX)
u_sym=ca.SX.sym('u',NU)
d_sym=ca.SX.sym('d',ND)
p_sym=ca.SX.sym('p',len(P_NATIVE))
a_sym=aux_states.update(x_sym,u_sym,d_sym,p_sym)
dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
q_ext=a_sym[136]+a_sym[137]+a_sym[145]
FQ900=ca.integrator(
    'EXP3_4A_M1_Q900','cvodes',
    {'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym,'quad':q_ext},
    0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':70000}
)

H_EFF=float(P_NATIVE[49])
A_FLOOR=float(P_NATIVE[46])
A_ROOF=float(P_NATIVE[55])
LEAK_TOP=float(P_NATIVE[204])

def initial_state(r):
    x=NATIVE_X.copy()
    x[2]=float(r.event_Tair)
    x[15]=float(r.in_vp_pa)
    x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    x[4]=float(r.event_Tair)
    return x

def one_event_action(r,u):
    # Native EXP3.1 trajectory is retained exactly for scientific outcomes.
    d=build_weather(r)
    x0=initial_state(r)
    a=full_action(float(u))

    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    ee=env.unwrapped
    ee.weather_data=d.copy()
    ee.day_of_year=int(r.day_of_year)
    ee.hour_of_day=float(r.hour_decimal)
    ee.x=x0.copy(); ee.x_prev=x0.copy(); ee.obs=ee._get_obs()

    native=[]; action_errs=[]; finite=True
    for _ in range(2):
        _,_,_,truncated,info=env.step(a)
        if truncated:
            raise RuntimeError((int(r.event_id),u,'M1 native trajectory truncated'))
        applied=np.asarray(info['controls'],dtype=float)
        action_errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(ee.x)))
        native.append(ee.x.copy())
    env.close()
    x1_native,x2_native=native

    # Read-only augmented quadrature for the physical ventilation dose.
    # Each 900-s segment is re-anchored to the corresponding native state so
    # quadrature integration cannot perturb the trajectory used for T/AH.
    q1r=FQ900(x0=x0,u=a.astype(float),p=np.concatenate([d[0],P_NATIVE]))
    x1q=np.asarray(q1r['xf'].full()).reshape(-1)
    q1=float(np.asarray(q1r['qf'].full()).reshape(-1)[0])

    q2r=FQ900(x0=x1_native,u=a.astype(float),p=np.concatenate([d[1],P_NATIVE]))
    x2q=np.asarray(q2r['xf'].full()).reshape(-1)
    q2=float(np.asarray(q2r['qf'].full()).reshape(-1)[0])

    qerr1=float(np.max(np.abs(x1q-x1_native)))
    qerr2=float(np.max(np.abs(x2q-x2_native)))
    if not (finite and np.all(np.isfinite(x1q)) and np.all(np.isfinite(x2q)) and np.isfinite(q1) and np.isfinite(q2)):
        raise RuntimeError((int(r.event_id),u,'non-finite M1 dose trajectory'))

    init_err=max(
        abs(float(x0[2])-float(r.event_Tair)),
        abs(float(x0[15])-float(r.in_vp_pa)),
        abs(float(mg_m3_to_ppm(x0[2],x0[0]))-float(r.CO2_pre_ppm)),
        abs(float(x0[4])-float(r.event_Tair))
    )

    out=[]
    for h,xf,dose,qerr in [(15,x1_native,q1,qerr1),(30,x2_native,q1+q2,qerr2)]:
        T=float(xf[2]); VP=float(xf[15])
        DN=float(dose/H_EFF)
        out.append({
          'model_id':'M1','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,
          'horizon_min':h,'action':float(u),
          'specific_volume_dose_m3_m2':float(dose),
          'air_volume_equiv':DN,
          'average_specific_flux_m3_m2_s':float(dose/(h*60.0)),
          'average_ach_h_1':float(DN/(h/60.0)),
          'T':T,'AH':float(locked_exp31_ah(T,VP)),
          'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
          'init_max_abs_error':float(init_err),
          'action_max_abs_error':float(max(action_errs)),
          'quadrature_vs_native_state_max_abs_error':qerr
        })
    return out
rows=[]
for _,r in EVENTS.iterrows():
    for u in U9:
        rows.extend(one_event_action(r,float(u)))
    print('M1 event',int(r.event_id),'done',flush=True)

df=pd.DataFrame(rows).sort_values(['event_id','horizon_min','action']).reset_index(drop=True)
assert len(df)==61*2*9
assert not df.duplicated(['event_id','horizon_min','action']).any()
assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
df.to_csv(OUT/'physical_dose_grid.csv',index=False,float_format='%.12g')

# Same-run reconstruction against the untouched EXP3.1 runner.
ref=pd.read_csv(REF)
u5=df[np.isclose((df.action*10)%2,1,atol=1e-12)].copy()
u5=u5[u5.action.isin(U5)]
cmp=ref.merge(
    u5[['event_id','horizon_min','action','T','AH']],
    on=['event_id','horizon_min','action'],suffixes=('_ref','_dose'),validate='one_to_one'
)
assert len(cmp)==61*2*5
cmp['abs_T_error']=np.abs(cmp.T_ref-cmp.T_dose)
cmp['abs_AH_error']=np.abs(cmp.AH_ref-cmp.AH_dose)
cmp.to_csv(OUT/'same_run_exp3_1_reconstruction.csv',index=False,float_format='%.12g')
max_recon=float(max(cmp.abs_T_error.max(),cmp.abs_AH_error.max()))

summary={
 'experiment':'PhysBench-GH EXP3.4A','model_id':'M1','model':'GreenLight-Gym2',
 'frozen_commit':MODEL_COMMIT,'input_sha256':sha256(IN),
 'events':61,'actions':U9.tolist(),'rows':int(len(df)),
 'physical_flux_definition':'a[136]+a[137]+a[145]',
 'physical_flux_units':'m3 m-2 s-1',
 'air_volume_per_floor_area_m':H_EFF,
 'floor_area_m2':A_FLOOR,'roof_area_m2':A_ROOF,
 'max_roof_aperture_per_floor_area_m2_m2':float(A_ROOF/A_FLOOR),
 'leak_top_fraction':LEAK_TOP,
 'all_finite':bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()),
 'max_init_error':float(df.init_max_abs_error.max()),
 'max_action_error':float(df.action_max_abs_error.max()),
 'same_run_exp3_1_max_T_AH_error':max_recon,
 'max_quadrature_vs_native_state_error':float(df.quadrature_vs_native_state_max_abs_error.max())
}
summary['gate_pass']=bool(
 summary['input_sha256']==EXPECTED_INPUT_SHA and summary['events']==61 and summary['rows']==61*2*9
 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0
 and summary['same_run_exp3_1_max_T_AH_error']<=1e-9
 and summary['max_quadrature_vs_native_state_error']<=1e-7
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('M1 EXP3.4A runtime/reconstruction gate failed')
