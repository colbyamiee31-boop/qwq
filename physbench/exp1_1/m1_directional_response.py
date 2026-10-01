import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym

ROOT = Path(__file__).resolve().parents[2]
FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT = ROOT / 'EXP1_1_M1_DIRECTIONAL_RESPONSE.json'

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0
ARMS={'LOW':0.1,'BASE':0.3,'HIGH':0.9}
OUTPUTS=['air_temperature_c','air_vapor_pressure_pa','air_rh_pct','air_co2_ppm','canopy_temperature_c']
PRIMARY=['air_temperature_c','air_vapor_pressure_pa','air_co2_ppm']
EXPECTED_PRIMARY_SIGN={k:-1 for k in PRIMARY}

def sat_vp(t):
    return 610.78*np.exp(17.2694*np.asarray(t)/(np.asarray(t)+238.3))

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp(t),0,100)

def ppm_to_mg_m3(t, ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t, mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sign_class(delta, high, low):
    scale=max(1.0,abs(float(high)),abs(float(low)))
    eps=1e-8*scale
    if delta>eps: return 1
    if delta<-eps: return -1
    return 0

f=pd.read_csv(FORCING)
target_init=json.loads(INIT.read_text())
assert np.array_equal(f['time_s'].to_numpy(),np.arange(0,10801,900))

# Frozen GreenLight disturbance mapping from EXP0.5.
d=np.zeros((len(f),10),dtype=float)
d[:,0]=f['global_radiation_w_m2']
d[:,1]=f['outdoor_temperature_c']
d[:,2]=f['outdoor_vapor_pressure_pa']
d[:,3]=ppm_to_mg_m3(f['outdoor_temperature_c'],f['outdoor_co2_ppm'])
d[:,4]=f['wind_speed_m_s']
d[:,5]=f['sky_temperature_c']
d[:,6]=f['soil_boundary_temperature_c']
d[:,7]=np.cumsum(f['global_radiation_w_m2'].to_numpy()*900.0)/1e6
d[:,8]=(f['global_radiation_w_m2'].to_numpy()>0).astype(float)
d[:,9]=d[:,8]

def export_state(e,t):
    return {
      'time_s':float(t),
      'air_temperature_c':float(e.x[2]),
      'air_vapor_pressure_pa':float(e.x[15]),
      'air_rh_pct':float(rh_from_t_vp(e.x[2],e.x[15])),
      'air_co2_ppm':float(mg_m3_to_ppm(e.x[2],e.x[0])),
      'canopy_temperature_c':float(e.x[4]),
    }

def run_arm(v):
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=d.copy(); e.day_of_year=244; e.hour_of_day=8.0
    x=e.x.copy()
    x[2]=target_init['air_temperature_c']
    x[15]=target_init['air_vapor_pressure_pa']
    x[0]=ppm_to_mg_m3(target_init['air_temperature_c'],target_init['air_co2_ppm'])
    x[4]=target_init['canopy_temperature_c']
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    trace=[export_state(e,0.0)]
    errs=[]; finite=True
    a=np.array([0,0,0,v,0,0],dtype=np.float32)
    for k in range(12):
        obs,reward,terminated,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float)
        errs.append(float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        trace.append(export_state(e,(k+1)*900.0))
    env.close()
    return {'ventilation_command_fraction':v,'trace':trace,'max_requested_applied_error':max(errs),'finite':finite}

arms={name:run_arm(v) for name,v in ARMS.items()}
low=arms['LOW']['trace']; high=arms['HIGH']['trace']; base=arms['BASE']['trace']
contrasts=[]
for i in range(len(low)):
    row={'time_s':low[i]['time_s']}
    for key in OUTPUTS:
        delta=float(high[i][key]-low[i][key])
        row[key+'_high_minus_low']=delta
        row[key+'_sign']=sign_class(delta,high[i][key],low[i][key])
    contrasts.append(row)

primary_one_step={}
for key in PRIMARY:
    s=contrasts[1][key+'_sign']
    primary_one_step[key]={
      'delta_high_minus_low':contrasts[1][key+'_high_minus_low'],
      'observed_sign':s,
      'expected_sign':EXPECTED_PRIMARY_SIGN[key],
      'pass':bool(s==EXPECTED_PRIMARY_SIGN[key])
    }

result={
 'experiment':'PhysBench-GH EXP1.1',
 'model_id':'M1','model':'GreenLight-Gym2','frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
 'arms':arms,'high_minus_low':contrasts,
 'primary_horizon_s':900,
 'primary_one_step':primary_one_step,
 'primary_direction_pass_count':int(sum(v['pass'] for v in primary_one_step.values())),
 'primary_direction_total':len(PRIMARY),
 'all_actions_exact':bool(all(a['max_requested_applied_error']==0.0 for a in arms.values())),
 'all_runtime_finite':bool(all(a['finite'] for a in arms.values())),
 'interpretation_limit':'Directional experiment only; normalized command fractions are not calibrated equal physical airflow across models.'
}
result['run_pass']=bool(result['all_actions_exact'] and result['all_runtime_finite'])
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']:
    raise SystemExit('M1 EXP1.1 runtime audit failed')
