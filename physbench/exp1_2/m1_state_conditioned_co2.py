import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym

ROOT = Path(__file__).resolve().parents[2]
FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT = ROOT / 'EXP1_2_M1_STATE_CONDITIONED_CO2.json'

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0
BASE_VENT=0.3; LOW_VENT=0.1; HIGH_VENT=0.9
OUTPUTS=['air_temperature_c','air_vapor_pressure_pa','air_rh_pct','air_co2_ppm','canopy_temperature_c']
EXCHANGE_MAP={
 'air_temperature_c':'outdoor_temperature_c',
 'air_vapor_pressure_pa':'outdoor_vapor_pressure_pa',
 'air_co2_ppm':'outdoor_co2_ppm',
}
CO2_SWEEP=[355.0,375.0,395.0,405.0,410.0,412.5,415.0,417.5,420.0,425.0,435.0,455.0,475.0]

def sat_vp(t):
    return 610.78*np.exp(17.2694*np.asarray(t)/(np.asarray(t)+238.3))

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp(t),0,100)

def ppm_to_mg_m3(t, ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t, mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_sha(x):
    return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()

def sign_class(delta,a,b):
    scale=max(1.0,abs(float(a)),abs(float(b)))
    eps=1e-8*scale
    if delta>eps: return 1
    if delta<-eps: return -1
    return 0

def zero_crossings(xs,ys):
    xs=np.asarray(xs,dtype=float); ys=np.asarray(ys,dtype=float)
    out=[]
    for i in range(len(xs)-1):
        if ys[i]==0.0:
            out.append(float(xs[i]))
        elif ys[i]*ys[i+1] < 0.0:
            out.append(float(xs[i] + (0.0-ys[i])*(xs[i+1]-xs[i])/(ys[i+1]-ys[i])))
    if ys[-1]==0.0:
        out.append(float(xs[-1]))
    return out

f=pd.read_csv(FORCING)
target_init=json.loads(INIT.read_text())
assert np.array_equal(f['time_s'].to_numpy(),np.arange(0,10801,900))

# Frozen GreenLight disturbance mapping inherited from EXP0.5/EXP1.1.
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

def make_env(weather_start_idx=0):
    env=gym.make(
        'gl_gym/GreenLightTomato-v0',
        controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
        normalize_actions=False,
        parameter_provider='fixed'
    )
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped
    e.weather_data=d[weather_start_idx:].copy()
    e.day_of_year=244
    e.hour_of_day=8.0 + 0.25*weather_start_idx
    return env,e

def set_native_state(e,x):
    e.x=np.asarray(x,dtype=float).copy()
    e.x_prev=e.x.copy()
    e.obs=e._get_obs()

def export_state_from_x(x,t):
    x=np.asarray(x,dtype=float)
    return {
      'time_s':float(t),
      'air_temperature_c':float(x[2]),
      'air_vapor_pressure_pa':float(x[15]),
      'air_rh_pct':float(rh_from_t_vp(x[2],x[15])),
      'air_co2_ppm':float(mg_m3_to_ppm(x[2],x[0])),
      'canopy_temperature_c':float(x[4]),
    }

def full_action(v):
    return np.array([0,0,0,v,0,0],dtype=np.float32)

# BASE native-state trajectory.
env,e=make_env(0)
x=e.x.copy()
x[2]=target_init['air_temperature_c']
x[15]=target_init['air_vapor_pressure_pa']
x[0]=ppm_to_mg_m3(target_init['air_temperature_c'],target_init['air_co2_ppm'])
x[4]=target_init['canopy_temperature_c']
set_native_state(e,x)
baseline_native=[e.x.copy()]
baseline_common=[export_state_from_x(e.x,0.0)]
baseline_action_errors=[]
baseline_finite=True
for k in range(12):
    a=full_action(BASE_VENT)
    obs,reward,terminated,truncated,info=env.step(a)
    applied=np.asarray(info['controls'],dtype=float)
    baseline_action_errors.append(float(np.max(np.abs(applied-a))))
    baseline_finite=baseline_finite and bool(np.all(np.isfinite(e.x)))
    baseline_native.append(e.x.copy())
    baseline_common.append(export_state_from_x(e.x,(k+1)*900.0))
env.close()

# One matched-native-state probe.
def one_step_probe(x_start,j,v):
    env,e=make_env(j)
    set_native_state(e,x_start)
    a=full_action(v)
    obs,reward,terminated,truncated,info=env.step(a)
    applied=np.asarray(info['controls'],dtype=float)
    err=float(np.max(np.abs(applied-a)))
    finite=bool(np.all(np.isfinite(e.x)))
    final_x=e.x.copy()
    final_state=export_state_from_x(final_x,(j+1)*900.0)
    env.close()
    return final_state,err,finite,state_sha(final_x)

local_probes=[]
all_action_errors=list(baseline_action_errors)
all_finite=[baseline_finite]
for j in range(12):
    x0=baseline_native[j]
    low,err_l,fin_l,sha_l=one_step_probe(x0,j,LOW_VENT)
    high,err_h,fin_h,sha_h=one_step_probe(x0,j,HIGH_VENT)
    all_action_errors.extend([err_l,err_h]); all_finite.extend([fin_l,fin_h])
    row={
      'probe_time_s':float(j*900),
      'probe_time_min':float(j*15),
      'start_native_state_sha256':state_sha(x0),
      'low_final_native_state_sha256':sha_l,
      'high_final_native_state_sha256':sha_h,
      'baseline_common_state':baseline_common[j],
      'weather':{k:float(f.loc[j,k]) for k in ['outdoor_temperature_c','outdoor_vapor_pressure_pa','outdoor_co2_ppm']},
      'low_final':low,
      'high_final':high,
    }
    for key in OUTPUTS:
        delta=float(high[key]-low[key])
        row[key+'_high_minus_low']=delta
        row[key+'_observed_sign']=sign_class(delta,high[key],low[key])
    for key,outkey in EXCHANGE_MAP.items():
        indoor=float(baseline_common[j][key]); outdoor=float(f.loc[j,outkey])
        grad=indoor-outdoor
        gsign=sign_class(grad,indoor,outdoor)
        expected=-gsign if gsign!=0 else 0
        row[key+'_gradient_in_minus_out']=float(grad)
        row[key+'_gradient_sign']=int(gsign)
        row[key+'_expected_sign']=int(expected)
        row[key+'_evaluable']=bool(expected!=0)
        row[key+'_direction_consistent']=bool(expected!=0 and row[key+'_observed_sign']==expected)
    local_probes.append(row)

# Conditioned-direction summary.
cdc={}
for key in EXCHANGE_MAP:
    evaluable=[r for r in local_probes if r[key+'_evaluable']]
    passed=[r for r in evaluable if r[key+'_direction_consistent']]
    cdc[key]={
      'pass_count':len(passed),
      'evaluable_count':len(evaluable),
      'fraction':float(len(passed)/len(evaluable)) if evaluable else None,
      'mismatch_probe_times_min':[r['probe_time_min'] for r in evaluable if not r[key+'_direction_consistent']],
    }

# Linear estimate of where the BASE CO2 trajectory crosses outdoor CO2.
base_grad=[float(baseline_common[j]['air_co2_ppm']-f.loc[j,'outdoor_co2_ppm']) for j in range(13)]
base_times=f['time_s'].to_numpy(dtype=float)
base_cross=zero_crossings(base_times,base_grad)

# Fixed-state initial-CO2 sweep at t=0.
sweep=[]
initial_native=baseline_native[0].copy()
for c0 in CO2_SWEEP:
    xs=initial_native.copy()
    xs[0]=ppm_to_mg_m3(xs[2],c0)
    low,err_l,fin_l,_=one_step_probe(xs,0,LOW_VENT)
    high,err_h,fin_h,_=one_step_probe(xs,0,HIGH_VENT)
    all_action_errors.extend([err_l,err_h]); all_finite.extend([fin_l,fin_h])
    delta=float(high['air_co2_ppm']-low['air_co2_ppm'])
    sweep.append({
      'initial_indoor_co2_ppm':float(c0),
      'outdoor_co2_ppm':float(f.loc[0,'outdoor_co2_ppm']),
      'initial_gradient_ppm':float(c0-f.loc[0,'outdoor_co2_ppm']),
      'low_final_co2_ppm':float(low['air_co2_ppm']),
      'high_final_co2_ppm':float(high['air_co2_ppm']),
      'delta_high_minus_low_ppm':delta,
      'observed_sign':int(sign_class(delta,high['air_co2_ppm'],low['air_co2_ppm'])),
    })

sx=[r['initial_indoor_co2_ppm'] for r in sweep]
sy=[r['delta_high_minus_low_ppm'] for r in sweep]
sweep_cross=zero_crossings(sx,sy)
primary_cross=min(sweep_cross,key=lambda x:abs(x-float(f.loc[0,'outdoor_co2_ppm']))) if sweep_cross else None
monotonic_decreasing=bool(np.all(np.diff(np.asarray(sy,dtype=float)) < 0.0))

result={
 'experiment':'PhysBench-GH EXP1.2',
 'model_id':'M1','model':'GreenLight-Gym2','frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'benchmark_forcing_sha256':file_sha(FORCING),
 'matched_initial_state_sha256':file_sha(INIT),
 'protocol':{
   'base_ventilation_command_fraction':BASE_VENT,
   'low_ventilation_command_fraction':LOW_VENT,
   'high_ventilation_command_fraction':HIGH_VENT,
   'local_probe_horizon_s':900,
   'probe_times_s':[float(j*900) for j in range(12)],
   'co2_sweep_ppm':CO2_SWEEP,
 },
 'baseline_common_trace':baseline_common,
 'baseline_native_state_sha256':[state_sha(x) for x in baseline_native],
 'baseline_co2_gradient_ppm':base_grad,
 'baseline_co2_gradient_crossing_time_s_linear':base_cross,
 'local_probes':local_probes,
 'conditioned_directional_consistency':cdc,
 'co2_gradient_sweep':sweep,
 'co2_response_zero_crossings_ppm_linear':sweep_cross,
 'co2_primary_response_boundary_ppm':primary_cross,
 'co2_primary_boundary_shift_from_outdoor_ppm':None if primary_cross is None else float(primary_cross-f.loc[0,'outdoor_co2_ppm']),
 'co2_sweep_delta_monotonically_decreasing':monotonic_decreasing,
 'runtime_audit':{
   'max_requested_applied_error':float(max(all_action_errors)),
   'all_states_finite':bool(all(all_finite)),
   'baseline_native_state_count':len(baseline_native),
 },
 'interpretation_limit':'Local directional structure under normalized native ventilation command. No equal-airflow or empirical-validity claim.'
}
result['run_pass']=bool(result['runtime_audit']['max_requested_applied_error']==0.0 and result['runtime_audit']['all_states_finite'])
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']:
    raise SystemExit('M1 EXP1.2 runtime audit failed')
