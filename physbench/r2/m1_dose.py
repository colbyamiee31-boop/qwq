import json,sys
from pathlib import Path
import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np,pandas as pd
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physbench/r1_2'))
from common import ppm_to_mg_m3,mg_m3_to_ppm
OUT=ROOT/'evidence/R2/M1'; OUT.mkdir(parents=True,exist_ok=True)
INPUTS=[('primary',ROOT/'physbench/r2/generated/PRIMARY_97_R2_INPUT.csv'),('strict',ROOT/'physbench/r2/generated/STRICT_36_R2_INPUT.csv')]
def build_weather(r):
    d=np.zeros((3,10),float)
    d[:,0]=float(r.event_Iglob); d[:,1]=float(r.event_Tout); d[:,2]=float(r.out_vp_pa)
    d[:,3]=ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed)); d[:,4]=float(r.event_Windsp)
    d[:,5]=float(r.sky_temperature_c_proxy); d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,float(r.event_Iglob))*900.0)/1e6; d[:,8]=1.0 if float(r.event_Iglob)>0 else 0.0; d[:,9]=d[:,8]
    return d
def full_action(v): return np.array([0,0,0,float(v),0,0],dtype=np.float32)
env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped; NATIVE_X=e.x.copy(); P_NATIVE=np.asarray(e.p,float).copy(); NX=e.nx; NU=e.nu; ND=e.nd; env.close()
x_sym=ca.SX.sym('x',NX); u_sym=ca.SX.sym('u',NU); d_sym=ca.SX.sym('d',ND); p_sym=ca.SX.sym('p',len(P_NATIVE))
a_sym=aux_states.update(x_sym,u_sym,d_sym,p_sym); dx_sym=ODE(x_sym,u_sym,d_sym,p_sym)
q_ext=a_sym[136]+a_sym[137]+a_sym[145]
FQ=ca.integrator('R2_M1_Q900','cvodes',{'x':x_sym,'u':u_sym,'p':ca.vertcat(d_sym,p_sym),'ode':dx_sym,'quad':q_ext},0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':70000})
H_EFF=float(P_NATIVE[49])
def run_arm(r,u):
    if not 0<=float(u)<=1: raise AssertionError(('action_bounds',r.event_id,u))
    d=build_weather(r); x=NATIVE_X.copy()
    x[2]=float(r.event_Tair); x[15]=float(r.in_vp_pa); x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm))); x[4]=float(r.canopy_temperature_c_closure)
    a=full_action(u)
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    ee=env.unwrapped; ee.weather_data=d.copy(); ee.day_of_year=int(r.day_of_year); ee.hour_of_day=float(r.hour_decimal); ee.x=x.copy(); ee.x_prev=x.copy(); ee.obs=ee._get_obs()
    _,_,_,truncated,info=env.step(a)
    if truncated: raise RuntimeError((r.event_id,u,'truncated'))
    x_native=ee.x.copy(); applied=np.asarray(info['controls'],float); finite=bool(np.all(np.isfinite(x_native))); env.close()
    qr=FQ(x0=x,u=a.astype(float),p=np.concatenate([d[0],P_NATIVE])); xq=np.asarray(qr['xf'].full()).reshape(-1); dose=float(np.asarray(qr['qf'].full()).reshape(-1)[0])
    qerr=float(np.max(np.abs(xq-x_native)))
    init_err=max(abs(float(x[2])-float(r.event_Tair)),abs(float(x[15])-float(r.in_vp_pa)),abs(float(mg_m3_to_ppm(x[2],x[0]))-float(r.CO2_pre_ppm)),abs(float(x[4])-float(r.canopy_temperature_c_closure)))
    return {'DV':dose,'DN':dose/H_EFF,'init_err':init_err,'action_err':float(np.max(np.abs(applied-a))),'qstate_err':qerr,'finite':bool(finite and np.isfinite(dose) and np.all(np.isfinite(xq)))}
rows=[]
for cohort,inp in INPUTS:
    E=pd.read_csv(inp)
    for _,r in E.iterrows():
        for arm,u in [('pre',r.u_pre),('post',r.u_post)]:
            z=run_arm(r,float(u)); rows.append({'cohort':cohort,'event_id':r.event_id,'event_date':r.event_date,'arm':arm,'action':float(u),'event_Windsp':float(r.event_Windsp),'DV':z['DV'],'DN':z['DN'],'H_eff':H_EFF,**z})
df=pd.DataFrame(rows); df.to_csv(OUT/'dose_arms.csv',index=False,float_format='%.12g')
summary={'model':'M1','rows':len(df),'events_primary':97,'events_strict':36,'H_eff_m':H_EFF,'max_init_error':float(df.init_err.max()),'max_action_error':float(df.action_err.max()),'max_qstate_error':float(df.qstate_err.max()),'all_finite':bool(df.finite.all())}
summary['gate_pass']=bool(len(df)==2*(97+36) and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-7 and summary['max_qstate_error']<=1e-7)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R2 M1 runtime gate failed')
