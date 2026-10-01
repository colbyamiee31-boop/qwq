import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSG = ROOT / 'CSGtom'
import sys
sys.path.insert(0, str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT = ROOT / 'EXP0_5_M2_MATCHED_GATE_RESULT.json'

def sat_vp(t):
    return 610.78*np.exp(17.2694*np.asarray(t)/(np.asarray(t)+238.3))

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp(t),0,100)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

f = pd.read_csv(FORCING)
target_init = json.loads(INIT.read_text())
assert np.array_equal(f['time_s'].to_numpy(), np.arange(0,10801,900))

# Piecewise-constant benchmark forcing on the 900-s common grid.
def pwc(column, scale=1.0):
    vals=f[column].to_numpy(dtype=float)*scale
    def fn(t):
        idx=int(np.floor(float(np.asarray(t).reshape(-1)[0])/900.0 + 1e-12))
        idx=max(0,min(idx,len(vals)-1))
        return vals[idx]
    return fn

p=example.parameters()
# The frozen constructor insists that StartTime exists in example_data.xls.
# 00:00 is known to exist from EXP0.3, so it is used only for construction.
p['StartTime']='2017-09-01T00:00'
p['EndTime']='2017-09-01T03:00'
p['dtsim']=900
p['dt']=30
p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
p['T_soilbound']=float(f['soil_boundary_temperature_c'].iloc[0])

# Match only common observable initial state; retain all other CSGtom native initial values.
sv=list(p['StateVariable']); iv=p['InitialValues'].copy()
for name,val in {
    'T_air':target_init['air_temperature_c'],
    'VP':target_init['air_vapor_pressure_pa'],
    'CO2':target_init['air_co2_ppm'],
    'T_can':target_init['canopy_temperature_c'],
}.items():
    iv[sv.index(name)]=val
p['InitialValues']=iv

tsim=np.arange(0,10800+p['dtsim'],p['dtsim'],dtype=float)
x0={name:p['InitialValues'][i] for i,name in enumerate(p['StateVariable'])}
model=CSG_Climate(tsim,p['dt'],x0,p)

# Rebuild only the model-native solar geometry for a daylight 08:00-11:00 benchmark.
# Geometry/physics equations are unchanged; the example XLS is no longer used after this point.
model.p['StartTime']='2017-09-01T08:00'
model.p['EndTime']='2017-09-01T11:00'
model.D=csg_shape.csg_shape(model.p)

model.d={
 'f_Rad':pwc('global_radiation_w_m2'),
 'f_Tem':pwc('outdoor_temperature_c'),
 'f_RH':pwc('outdoor_rh_pct',0.01),
 'f_CO2':pwc('outdoor_co2_ppm'),
 'f_Wind':pwc('wind_speed_m_s'),
 'f_Tsky':pwc('sky_temperature_c'),
}
vent=0.3
model.U={
 'u_blanket':lambda t:0.0,
 'u_vent':lambda t:vent,
 'u_venttop':lambda t:1.0,
 'u_ventside':lambda t:0.0,
 'u_venttopbot':lambda t:0.0,
}

init_export={
 'air_temperature_c':float(p['InitialValues'][sv.index('T_air')]),
 'air_vapor_pressure_pa':float(p['InitialValues'][sv.index('VP')]),
 'air_rh_pct':float(rh_from_t_vp(p['InitialValues'][sv.index('T_air')],p['InitialValues'][sv.index('VP')])),
 'air_co2_ppm':float(p['InitialValues'][sv.index('CO2')]),
 'canopy_temperature_c':float(p['InitialValues'][sv.index('T_can')]),
}

weather_export=[]
for _,row in f.iterrows():
    t=float(row.time_s); T=float(model.d['f_Tem'](t)); RH=float(model.d['f_RH'](t))*100.0
    weather_export.append({
      'time_s':t,
      'global_radiation_w_m2':float(model.d['f_Rad'](t)),
      'outdoor_temperature_c':T,
      'outdoor_vapor_pressure_pa':float(sat_vp(T)*RH/100.0),
      'outdoor_rh_pct':RH,
      'outdoor_co2_ppm':float(model.d['f_CO2'](t)),
      'wind_speed_m_s':float(model.d['f_Wind'](t)),
      'sky_temperature_c':float(model.d['f_Tsky'](t)),
      'soil_boundary_temperature_c':float(p['T_soilbound']),
    })

common_cols=['global_radiation_w_m2','outdoor_temperature_c','outdoor_vapor_pressure_pa','outdoor_rh_pct','outdoor_co2_ppm','wind_speed_m_s','sky_temperature_c','soil_boundary_temperature_c']
forcing_errors={k:float(np.max(np.abs(np.asarray([r[k] for r in weather_export])-f[k].to_numpy(dtype=float)))) for k in common_cols}
init_errors={k:abs(init_export[k]-float(target_init[k])) for k in target_init}

# Audit actual controls consumed by ctl_csg1 while running the unmodified frozen equations.
records=[]; orig=csg_fun.ctl_csg1
def logged_ctl(p_,D,d,Tair,t,U):
    res=orig(p_,D,d,Tair,t,U)
    records.append((float(t),float(np.asarray(res[1]).reshape(-1)[0])))
    return res
csg_fun.ctl_csg1=logged_ctl
try:
    y=model.run((0.0,10800.0))
finally:
    csg_fun.ctl_csg1=orig
rec=np.asarray(records,dtype=float)
action_err=float(np.max(np.abs(rec[:,1]-vent)))
finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in p['StateVariable'])

result={
 'experiment':'PhysBench-GH EXP0.5','model_id':'M2','model':'CSGtom','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
 'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
 'common_sample_period_s':900,'native_integration_step_s':30,
 'forcing_semantics':'piecewise constant over each 900-s benchmark interval, evaluated at every 30-s Euler step',
 'matched_initial_state':init_export,'initial_state_abs_errors':init_errors,'initial_state_max_abs_error':float(max(init_errors.values())),
 'canonical_weather_export':weather_export,'forcing_abs_errors':forcing_errors,'forcing_max_abs_error':float(max(forcing_errors.values())),
 'native_extension_match':{'soil_boundary_temperature_c':True,'csg_solar_geometry_remains_model_native':True,'solar_geometry_clock':'2017-09-01T08:00 to 11:00'},
 'action_audit':{'ventilation_command_fraction':vent,'native_control':'u_vent','max_requested_applied_error':action_err,'u_venttop_fixed':1.0,'u_blanket_fixed':0.0},
 'finite_runtime_states':bool(finite),
 'noncommon_internal_state_policy':'all non-common CSGtom InitialValues retained from native parameter file',
 'stateful_step_supported_by_frozen_source':False,
 'batch_rollout_required':True,
}
result['gate_pass']=bool(result['initial_state_max_abs_error']<=1e-9 and result['forcing_max_abs_error']<=1e-6 and action_err==0.0 and finite)
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['gate_pass']: raise SystemExit('M2 EXP0.5 gate failed')
