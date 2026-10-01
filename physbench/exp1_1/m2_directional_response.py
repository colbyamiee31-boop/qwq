import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
OUT=ROOT/'EXP1_1_M2_DIRECTIONAL_RESPONSE.json'

ARMS={'LOW':0.1,'BASE':0.3,'HIGH':0.9}
OUTPUTS=['air_temperature_c','air_vapor_pressure_pa','air_rh_pct','air_co2_ppm','canopy_temperature_c']
PRIMARY=['air_temperature_c','air_vapor_pressure_pa','air_co2_ppm']
EXPECTED_PRIMARY_SIGN={k:-1 for k in PRIMARY}

def sat_vp(t):
    return 610.78*np.exp(17.2694*np.asarray(t)/(np.asarray(t)+238.3))

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp(t),0,100)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sign_class(delta,high,low):
    scale=max(1.0,abs(float(high)),abs(float(low)))
    eps=1e-8*scale
    if delta>eps: return 1
    if delta<-eps: return -1
    return 0

f=pd.read_csv(FORCING)
target_init=json.loads(INIT.read_text())
assert np.array_equal(f['time_s'].to_numpy(),np.arange(0,10801,900))

def pwc(column,scale=1.0):
    vals=f[column].to_numpy(dtype=float)*scale
    def fn(t):
        idx=int(np.floor(float(np.asarray(t).reshape(-1)[0])/900.0+1e-12))
        idx=max(0,min(idx,len(vals)-1))
        return vals[idx]
    return fn

def build_and_run(v):
    p=example.parameters()
    # Constructor-compatible time window; daylight geometry rebuilt immediately afterward.
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(f['soil_boundary_temperature_c'].iloc[0])
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
    model.p['StartTime']='2017-09-01T08:00'; model.p['EndTime']='2017-09-01T11:00'
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':pwc('global_radiation_w_m2'),
      'f_Tem':pwc('outdoor_temperature_c'),
      'f_RH':pwc('outdoor_rh_pct',0.01),
      'f_CO2':pwc('outdoor_co2_ppm'),
      'f_Wind':pwc('wind_speed_m_s'),
      'f_Tsky':pwc('sky_temperature_c'),
    }
    model.U={
      'u_blanket':lambda t:0.0,
      'u_vent':lambda t:v,
      'u_venttop':lambda t:1.0,
      'u_ventside':lambda t:0.0,
      'u_venttopbot':lambda t:0.0,
    }
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
    action_err=float(np.max(np.abs(rec[:,1]-v)))
    t_native=np.asarray(y['t'],dtype=float)
    idx=[]
    for t in np.arange(0,10801,900,dtype=float):
        hits=np.where(np.isclose(t_native,t,rtol=0,atol=1e-9))[0]
        if len(hits)!=1: raise AssertionError(f'Cannot uniquely sample t={t}, hits={hits}')
        idx.append(int(hits[0]))
    trace=[]
    for j,t in zip(idx,np.arange(0,10801,900,dtype=float)):
        T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
        trace.append({
          'time_s':float(t),
          'air_temperature_c':T,
          'air_vapor_pressure_pa':VP,
          'air_rh_pct':float(rh_from_t_vp(T,VP)),
          'air_co2_ppm':float(np.asarray(y['CO2'])[j]),
          'canopy_temperature_c':float(np.asarray(y['T_can'])[j]),
        })
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in p['StateVariable'])
    return {'ventilation_command_fraction':v,'trace':trace,'max_requested_applied_error':action_err,'finite':bool(finite)}

arms={name:build_and_run(v) for name,v in ARMS.items()}
low=arms['LOW']['trace']; high=arms['HIGH']['trace']
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
 'model_id':'M2','model':'CSGtom','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
 'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
 'arms':arms,'high_minus_low':contrasts,
 'primary_horizon_s':900,
 'primary_one_step':primary_one_step,
 'primary_direction_pass_count':int(sum(v['pass'] for v in primary_one_step.values())),
 'primary_direction_total':len(PRIMARY),
 'all_actions_exact':bool(all(a['max_requested_applied_error']==0.0 for a in arms.values())),
 'all_runtime_finite':bool(all(a['finite'] for a in arms.values())),
 'native_integration_step_s':30,
 'common_output_grid_s':900,
 'interpretation_limit':'Directional experiment only; normalized command fractions are not calibrated equal physical airflow across models.'
}
result['run_pass']=bool(result['all_actions_exact'] and result['all_runtime_finite'])
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']:
    raise SystemExit('M2 EXP1.1 runtime audit failed')
