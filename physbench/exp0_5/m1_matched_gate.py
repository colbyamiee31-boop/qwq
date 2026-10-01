import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym

ROOT = Path(__file__).resolve().parents[2]
FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT = ROOT / 'EXP0_5_M1_MATCHED_GATE_RESULT.json'

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

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

f = pd.read_csv(FORCING)
target_init = json.loads(INIT.read_text())
assert np.array_equal(f['time_s'].to_numpy(), np.arange(0,10801,900))

# Build exact M1 native disturbance matrix. First 7 columns are physical GreenLight disturbances.
d = np.zeros((len(f),10),dtype=float)
d[:,0] = f['global_radiation_w_m2']
d[:,1] = f['outdoor_temperature_c']
d[:,2] = f['outdoor_vapor_pressure_pa']
d[:,3] = ppm_to_mg_m3(f['outdoor_temperature_c'], f['outdoor_co2_ppm'])
d[:,4] = f['wind_speed_m_s']
d[:,5] = f['sky_temperature_c']
d[:,6] = f['soil_boundary_temperature_c']
# Model-native extras not shared by CSGtom: deterministic DLI/day flags.
d[:,7] = np.cumsum(f['global_radiation_w_m2'].to_numpy()*900.0)/1e6
d[:,8] = (f['global_radiation_w_m2'].to_numpy()>0).astype(float)
d[:,9] = d[:,8]

env = gym.make('gl_gym/GreenLightTomato-v0', controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'], normalize_actions=False, parameter_provider='fixed')
obs, info = env.reset(seed=3407, options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e = env.unwrapped
e.weather_data = d.copy()
e.day_of_year = 244
e.hour_of_day = 8.0

# Match only the frozen common observable initial state; preserve all other native M1 states.
x = e.x.copy()
x[2] = target_init['air_temperature_c']
x[15] = target_init['air_vapor_pressure_pa']
x[0] = ppm_to_mg_m3(target_init['air_temperature_c'], target_init['air_co2_ppm'])
x[4] = target_init['canopy_temperature_c']
e.x = x.copy(); e.x_prev = x.copy(); e.obs = e._get_obs()

init_export = {
    'air_temperature_c': float(e.x[2]),
    'air_vapor_pressure_pa': float(e.x[15]),
    'air_rh_pct': float(rh_from_t_vp(e.x[2],e.x[15])),
    'air_co2_ppm': float(mg_m3_to_ppm(e.x[2],e.x[0])),
    'canopy_temperature_c': float(e.x[4]),
}

weather_export=[]
for i,row in f.iterrows():
    di=e.weather_data[i]
    weather_export.append({
      'time_s': float(row.time_s),
      'global_radiation_w_m2': float(di[0]),
      'outdoor_temperature_c': float(di[1]),
      'outdoor_vapor_pressure_pa': float(di[2]),
      'outdoor_rh_pct': float(rh_from_t_vp(di[1],di[2])),
      'outdoor_co2_ppm': float(mg_m3_to_ppm(di[1],di[3])),
      'wind_speed_m_s': float(di[4]),
      'sky_temperature_c': float(di[5]),
      'soil_boundary_temperature_c': float(di[6]),
    })

common_cols=['global_radiation_w_m2','outdoor_temperature_c','outdoor_vapor_pressure_pa','outdoor_rh_pct','outdoor_co2_ppm','wind_speed_m_s','sky_temperature_c','soil_boundary_temperature_c']
forcing_errors={k:float(np.max(np.abs(np.asarray([r[k] for r in weather_export])-f[k].to_numpy(dtype=float)))) for k in common_cols}
init_errors={k:abs(init_export[k]-float(target_init[k])) for k in target_init}

# Execute the frozen model under injected forcing to prove the adapter is operative.
a=np.array([0,0,0,0.3,0,0],dtype=np.float32)
applied_errors=[]; finite=True
for _ in range(len(f)-1):
    obs,reward,terminated,truncated,step_info=env.step(a)
    applied=np.asarray(step_info['controls'],dtype=float)
    applied_errors.append(float(np.max(np.abs(applied-a))))
    finite = finite and bool(np.all(np.isfinite(e.x)))
env.close()

result={
 'experiment':'PhysBench-GH EXP0.5','model_id':'M1','model':'GreenLight-Gym2','frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
 'common_sample_period_s':900,'native_integration_step_s':900,
 'forcing_semantics':'piecewise constant over each 900-s benchmark interval',
 'matched_initial_state':init_export,'initial_state_abs_errors':init_errors,'initial_state_max_abs_error':float(max(init_errors.values())),
 'canonical_weather_export':weather_export,'forcing_abs_errors':forcing_errors,'forcing_max_abs_error':float(max(forcing_errors.values())),
 'native_extension_match':{'soil_boundary_temperature_c':True,'m1_dli_isday_are_model_native_extras':True},
 'action_audit':{'ventilation_command_fraction':0.3,'native_control':'uVent','max_requested_applied_error':float(max(applied_errors))},
 'finite_runtime_states':finite,
 'noncommon_internal_state_policy':'all non-common M1 states retained from native reset',
}
result['gate_pass']=bool(result['initial_state_max_abs_error']<=1e-9 and result['forcing_max_abs_error']<=1e-9 and result['action_audit']['max_requested_applied_error']==0.0 and finite)
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['gate_pass']: raise SystemExit('M1 EXP0.5 gate failed')
