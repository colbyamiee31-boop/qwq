import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym

from common import *
from generate_prehistory import generate_prehistory

IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
EVENTS=pd.read_csv(IN)
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
HISTORY_IDS=['HNR']+list(CFG['decision_primary_history_ids'])
NEUTRAL=float(CFG['neutral_prehistory_vent_command'])
DURATION_H=int(CFG['primary_prehistory_h'])

parser=argparse.ArgumentParser()
parser.add_argument('--history-id',required=True,choices=HISTORY_IDS)
args=parser.parse_args()
HID=args.history_id
OUT=ROOT/f'evidence/EXP0_6C_M1_{HID}'
OUT.mkdir(parents=True,exist_ok=True)

EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
assert sha256(IN)==EXPECTED_INPUT_SHA
assert EVENTS.event_id.nunique()==61 and len(EVENTS)==61
assert list(CFG['decision_primary_history_ids'])==['H00','H01','H04','H06','H07','H10','H11','H13','H16']

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
    z=df.iloc[:-1].reset_index(drop=True)
    d=np.zeros((len(z),10),dtype=float)
    d[:,0]=z.global_radiation_w_m2
    d[:,1]=z.outdoor_temperature_c
    d[:,2]=z.outdoor_vapor_pressure_pa
    d[:,3]=ppm_to_mg_m3(z.outdoor_temperature_c,z.outdoor_co2_ppm)
    d[:,4]=z.wind_speed_m_s
    d[:,5]=z.sky_temperature_c
    d[:,6]=z.soil_boundary_temperature_c
    dli=[]; acc=0.0; prev=None
    for _,r in z.iterrows():
        h=float(r.local_hour)
        if prev is not None and h < prev-1e-9:
            acc=0.0
        acc += float(r.global_radiation_w_m2)*900.0/1e6
        dli.append(acc); prev=h
    d[:,7]=np.asarray(dli,float)
    d[:,8]=(z.global_radiation_w_m2.to_numpy(float)>0).astype(float)
    d[:,9]=d[:,8]
    return d

def event_weather(r):
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

def native_reset():
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    x=env.unwrapped.x.copy()
    env.close()
    return x

NATIVE=native_reset()

def forcing_sha(df):
    raw=df.to_csv(index=False,float_format='%.12g').encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def prehistory_state(r):
    if HID=='HNR':
        return NATIVE.copy(),{
          'forcing_sha256':None,'all_states_finite':True,'max_requested_applied_error':0.0,
          'start_day_of_year':None,'start_hour_decimal':None,
          'end_day_of_year':None,'end_hour_decimal':None,'clock_error_h':0.0
        }
    hist=generate_prehistory(HID,DURATION_H,float(r.hour_decimal))
    d=weather_matrix(hist)
    event_ts=pd.Timestamp(r.timestamp)
    start_ts=event_ts-pd.Timedelta(hours=DURATION_H)
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=d.copy()
    e.day_of_year=float(start_ts.dayofyear)
    e.hour_of_day=float(start_ts.hour+start_ts.minute/60.0+start_ts.second/3600.0)
    errs=[]; finite=True
    for _ in range(len(d)):
        a=full_action(NEUTRAL)
        _,_,_,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        if truncated:
            raise RuntimeError(f'M1 prehistory truncated event={int(r.event_id)} history={HID}')
    x=e.x.copy()
    end_hour=float(e.hour_of_day)%24.0
    target_hour=float(r.hour_decimal)%24.0
    dh=abs(end_hour-target_hour); clock_err=min(dh,24.0-dh)
    meta={
      'forcing_sha256':forcing_sha(hist),'all_states_finite':bool(finite),
      'max_requested_applied_error':float(max(errs) if errs else 0.0),
      'start_day_of_year':float(start_ts.dayofyear),
      'start_hour_decimal':float(start_ts.hour+start_ts.minute/60.0+start_ts.second/3600.0),
      'end_day_of_year':float(e.day_of_year),'end_hour_decimal':end_hour,
      'clock_error_h':float(clock_err)
    }
    env.close()
    return x,meta

def projected_state(latent,r):
    x=np.asarray(latent,dtype=float).copy()
    x[2]=float(r.event_Tair)
    x[15]=float(r.in_vp_pa)
    x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    x[4]=float(r.event_Tair)
    return x

def run_action(latent,r,vent):
    env=gym.make('gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=event_weather(r)
    e.day_of_year=float(r.day_of_year)
    e.hour_of_day=float(r.hour_decimal)
    x=projected_state(latent,r)
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    init_err=max(
        abs(float(e.x[2])-float(r.event_Tair)),
        abs(float(e.x[15])-float(r.in_vp_pa)),
        abs(float(mg_m3_to_ppm(e.x[2],e.x[0]))-float(r.CO2_pre_ppm)),
        abs(float(e.x[4])-float(r.event_Tair))
    )
    a=full_action(vent)
    trace=[]; errs=[]; finite=True
    for k in range(2):
        _,_,_,truncated,info=env.step(a)
        if truncated:
            raise RuntimeError(f'M1 intervention truncated event={int(r.event_id)} history={HID} action={vent}')
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        T=float(e.x[2]); VP=float(e.x[15])
        trace.append({'horizon_min':15*(k+1),'T':T,'VP':VP,'AH':float(ah_g_m3_from_vp_pa(T,VP))})
    env.close()
    return trace,float(max(errs)),bool(finite),float(init_err)

rows=[]; audits=[]; latent_meta=[]; latent_states={}
for _,r in EVENTS.iterrows():
    eid=int(r.event_id)
    latent,pm=prehistory_state(r)
    latent_states[f'E{eid}']=np.asarray(latent,dtype=np.float64)
    pre=common_state(latent)
    post=common_state(projected_state(latent,r))
    latent_meta.append({
      'model_id':'M1','history_id':HID,'event_id':eid,'timestamp':r.timestamp,
      'native_state_sha256':state_sha(latent),'forcing_sha256':pm['forcing_sha256'],
      'prehistory_finite':pm['all_states_finite'],'prehistory_max_action_error':pm['max_requested_applied_error'],
      'prehistory_clock_error_h':pm['clock_error_h'],
      'pre_air_temperature_c':pre['air_temperature_c'],'pre_air_vapor_pressure_pa':pre['air_vapor_pressure_pa'],
      'pre_air_co2_ppm':pre['air_co2_ppm'],'pre_canopy_temperature_c':pre['canopy_temperature_c'],
      'post_air_temperature_c':post['air_temperature_c'],'post_air_vapor_pressure_pa':post['air_vapor_pressure_pa'],
      'post_air_co2_ppm':post['air_co2_ppm'],'post_canopy_temperature_c':post['canopy_temperature_c'],
      'projection_T_C':post['air_temperature_c']-pre['air_temperature_c'],
      'projection_VP_Pa':post['air_vapor_pressure_pa']-pre['air_vapor_pressure_pa'],
      'projection_CO2_ppm':post['air_co2_ppm']-pre['air_co2_ppm'],
      'projection_canopy_T_C':post['canopy_temperature_c']-pre['canopy_temperature_c']
    })
    for u in ACTIONS:
        trace,ae,finite,ie=run_action(latent,r,float(u))
        audits.append({'model_id':'M1','history_id':HID,'event_id':eid,'action':float(u),
                       'init_max_abs_error':ie,'action_max_abs_error':ae,'finite':finite})
        for q in trace:
            rows.append({
              'model_id':'M1','history_id':HID,'event_id':eid,'team':r.team,'timestamp':r.timestamp,
              'event_date':r.event_date,'daynight':r.daynight,'horizon_min':q['horizon_min'],'action':float(u),
              'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
              'T':q['T'],'AH':q['AH']
            })
    print('M1',HID,'event',eid,'done',flush=True)

state_file=OUT/'latent_states_by_event.npz'
np.savez_compressed(state_file,**latent_states)
re=np.load(state_file)
restore_exact=all(np.array_equal(latent_states[k],re[k]) for k in latent_states)

df=pd.DataFrame(rows).sort_values(['event_id','horizon_min','action'])
ad=pd.DataFrame(audits).sort_values(['event_id','action'])
lm=pd.DataFrame(latent_meta).sort_values('event_id')
df.to_csv(OUT/'action_grid_responses.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
lm.to_csv(OUT/'latent_metadata.csv',index=False,float_format='%.12g')

summary={
 'experiment':'PhysBench-GH EXP0.6C','model':'M1','history_id':HID,
 'frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),'actions':ACTIONS.tolist(),
 'rows':int(len(df)),'all_finite':bool(ad.finite.all()),
 'max_init_error':float(ad.init_max_abs_error.max()),
 'max_action_error':float(ad.action_max_abs_error.max()),
 'prehistory_all_finite':bool(lm.prehistory_finite.all()),
 'prehistory_max_action_error':float(lm.prehistory_max_action_error.max()),
 'prehistory_max_clock_error_h':float(lm.prehistory_clock_error_h.max()),
 'same_run_state_restore_exact':bool(restore_exact)
}
summary['gate_pass']=bool(
 summary['input_sha256']==EXPECTED_INPUT_SHA and summary['events']==61 and summary['rows']==61*5*2
 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0
 and summary['prehistory_all_finite'] and summary['prehistory_max_action_error']==0.0
 and summary['prehistory_max_clock_error_h']<=1e-9 and summary['same_run_state_restore_exact']
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('M1 EXP0.6C runtime gate failed')
