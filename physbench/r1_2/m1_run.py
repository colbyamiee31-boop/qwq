import json
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym
import gl_gym
from common import *
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/R1_2/M1'; OUT.mkdir(parents=True,exist_ok=True)
INPUTS=[('primary',ROOT/'physbench/r1_2/generated/PRIMARY_97_MODEL_INPUT.csv'),('strict',ROOT/'physbench/r1_2/generated/STRICT_36_MODEL_INPUT.csv')]

def build_weather(r):
    d=np.zeros((3,10),dtype=float)
    d[:,0]=float(r.event_Iglob); d[:,1]=float(r.event_Tout); d[:,2]=float(r.out_vp_pa)
    d[:,3]=ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed)); d[:,4]=float(r.event_Windsp)
    d[:,5]=float(r.sky_temperature_c_proxy); d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,float(r.event_Iglob))*900.0)/1e6; d[:,8]=1.0 if float(r.event_Iglob)>0 else 0.0; d[:,9]=d[:,8]
    return d

def run_arm(r,vent):
    if not (0.0<=float(vent)<=1.0): raise AssertionError(('action_bounds',r.event_id,vent))
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped; e.weather_data=build_weather(r); e.day_of_year=int(r.day_of_year); e.hour_of_day=float(r.hour_decimal)
    x=e.x.copy(); x[2]=float(r.event_Tair); x[15]=float(r.in_vp_pa); x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm))); x[4]=float(r.canopy_temperature_c_closure)
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    init={'T':float(e.x[2]),'VP':float(e.x[15]),'CO2ppm':float(mg_m3_to_ppm(e.x[2],e.x[0])),'Tcan':float(e.x[4])}
    a=np.array([0,0,0,float(vent),0,0],dtype=np.float32); trace=[]; errs=[]; finite=True
    for k in range(2):
        obs,reward,terminated,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        T=float(e.x[2]); VP=float(e.x[15])
        trace.append({'horizon_min':15*(k+1),'T':T,'VP':VP,'AH':float(ah_from_t_vp(T,VP))})
    env.close(); return init,trace,max(errs),finite

rows=[]; audits=[]
for cohort,inp in INPUTS:
    E=pd.read_csv(inp)
    for _,r in E.iterrows():
        pi,po,pe,pf=run_arm(r,float(r.u_pre)); qi,qo,qe,qf=run_arm(r,float(r.u_post))
        init_err=max(abs(pi['T']-float(r.event_Tair)),abs(pi['VP']-float(r.in_vp_pa)),abs(pi['CO2ppm']-float(r.CO2_pre_ppm)),abs(pi['Tcan']-float(r.canopy_temperature_c_closure)),abs(qi['T']-float(r.event_Tair)),abs(qi['VP']-float(r.in_vp_pa)),abs(qi['CO2ppm']-float(r.CO2_pre_ppm)),abs(qi['Tcan']-float(r.canopy_temperature_c_closure)))
        audits.append({'cohort':cohort,'event_id':r.event_id,'init_max_abs_error':init_err,'action_max_abs_error':max(pe,qe),'finite':bool(pf and qf),'u_pre':r.u_pre,'u_post':r.u_post})
        for p,q in zip(po,qo):
            dt=float(q['T']-p['T']); dah=float(q['AH']-p['AH'])
            rows.append({'cohort':cohort,'event_id':r.event_id,'event_date':r.event_date,'daynight':r.daynight,'direction':r.direction,'horizon_min':int(q['horizon_min']),'delta_u':float(r.delta_u),'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'model_T_delta':dt,'model_AH_delta':dah,'model_T_sign':sign_model(dt,q['T'],p['T']),'model_AH_sign':sign_model(dah,q['AH'],p['AH']),'pre_T':p['T'],'post_T':q['T'],'pre_AH':p['AH'],'post_AH':q['AH']})
pd.DataFrame(rows).to_csv(OUT/'model_event_responses.csv',index=False,float_format='%.12g')
ad=pd.DataFrame(audits); ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
summary={'model':'M1','commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1','rows':len(rows),'events_primary':pd.read_csv(INPUTS[0][1]).event_id.nunique(),'events_strict':pd.read_csv(INPUTS[1][1]).event_id.nunique(),'all_finite':bool(ad.finite.all()),'max_init_error':float(ad.init_max_abs_error.max()),'max_action_error':float(ad.action_max_abs_error.max())}
summary['gate_pass']=bool(summary['rows']==2*(97+36) and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-7)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M1 R1.2 runtime gate failed')
