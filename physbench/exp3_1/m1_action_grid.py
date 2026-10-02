import json
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym
import gl_gym
from common import *

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/M1'; OUT.mkdir(parents=True,exist_ok=True)
EVENTS=pd.read_csv(IN)
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)

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

def run_action(r,vent):
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=build_weather(r)
    e.day_of_year=int(r.day_of_year); e.hour_of_day=float(r.hour_decimal)
    x=e.x.copy()
    x[2]=float(r.event_Tair)
    x[15]=float(r.in_vp_pa)
    x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    x[4]=float(r.event_Tair)
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    init_err=max(
        abs(float(e.x[2])-float(r.event_Tair)),
        abs(float(e.x[15])-float(r.in_vp_pa)),
        abs(float(mg_m3_to_ppm(e.x[2],e.x[0]))-float(r.CO2_pre_ppm)),
        abs(float(e.x[4])-float(r.event_Tair))
    )
    a=np.array([0,0,0,vent,0,0],dtype=np.float32)
    trace=[]; errs=[]; finite=True
    for k in range(2):
        obs,reward,terminated,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        T=float(e.x[2]); VP=float(e.x[15])
        trace.append({'horizon_min':15*(k+1),'T':T,'VP':VP,'AH':float(ah_from_t_vp(T,VP))})
    env.close()
    return trace,float(max(errs)),bool(finite),float(init_err)

rows=[]; audits=[]
for _,r in EVENTS.iterrows():
    for u in ACTIONS:
        trace,ae,finite,ie=run_action(r,float(u))
        audits.append({'event_id':int(r.event_id),'action':u,'init_max_abs_error':ie,'action_max_abs_error':ae,'finite':finite})
        for q in trace:
            rows.append({
                'event_id':int(r.event_id),'team':r.team,'timestamp':r.timestamp,'event_date':r.event_date,'daynight':r.daynight,
                'horizon_min':q['horizon_min'],'action':u,
                'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
                'T':q['T'],'AH':q['AH']
            })

df=pd.DataFrame(rows).sort_values(['event_id','horizon_min','action'])
ad=pd.DataFrame(audits).sort_values(['event_id','action'])
df.to_csv(OUT/'action_grid_responses.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
summary={
 'model':'M1','frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),
 'actions':ACTIONS.tolist(),'rows':len(df),'all_finite':bool(ad.finite.all()),
 'max_init_error':float(ad.init_max_abs_error.max()),'max_action_error':float(ad.action_max_abs_error.max())
}
summary['gate_pass']=bool(summary['events']==61 and summary['rows']==61*5*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M1 EXP3.1 runtime gate failed')
