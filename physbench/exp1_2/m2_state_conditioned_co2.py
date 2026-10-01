import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timedelta
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
OUT=ROOT/'EXP1_2_M2_STATE_CONDITIONED_CO2.json'

BASE_VENT=0.3; LOW_VENT=0.1; HIGH_VENT=0.9
OUTPUTS=['air_temperature_c','air_vapor_pressure_pa','air_rh_pct','air_co2_ppm','canopy_temperature_c']
EXCHANGE_MAP={
 'air_temperature_c':'outdoor_temperature_c',
 'air_vapor_pressure_pa':'outdoor_vapor_pressure_pa',
 'air_co2_ppm':'outdoor_co2_ppm',
}
CO2_SWEEP=[355.0,375.0,395.0,405.0,410.0,412.5,415.0,417.5,420.0,425.0,435.0,455.0,475.0]
BASE_CLOCK=datetime(2017,9,1,8,0)

def sat_vp(t):
    return 610.78*np.exp(17.2694*np.asarray(t)/(np.asarray(t)+238.3))

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp(t),0,100)

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

def pwc(column,scale=1.0):
    vals=f[column].to_numpy(dtype=float)*scale
    def fn(t):
        idx=int(np.floor(float(np.asarray(t).reshape(-1)[0])/900.0+1e-12))
        idx=max(0,min(idx,len(vals)-1))
        return vals[idx]
    return fn

def const(v):
    return lambda t, v=float(v): v

def common_from_vector(x,sv,t):
    x=np.asarray(x,dtype=float)
    T=float(x[sv.index('T_air')]); VP=float(x[sv.index('VP')])
    return {
      'time_s':float(t),
      'air_temperature_c':T,
      'air_vapor_pressure_pa':VP,
      'air_rh_pct':float(rh_from_t_vp(T,VP)),
      'air_co2_ppm':float(x[sv.index('CO2')]),
      'canopy_temperature_c':float(x[sv.index('T_can')]),
    }

def set_matched_common_initial(p):
    sv=list(p['StateVariable']); iv=p['InitialValues'].copy()
    for name,val in {
      'T_air':target_init['air_temperature_c'],
      'VP':target_init['air_vapor_pressure_pa'],
      'CO2':target_init['air_co2_ppm'],
      'T_can':target_init['canopy_temperature_c'],
    }.items():
        iv[sv.index(name)]=val
    p['InitialValues']=iv
    return sv

# Continuous BASE trajectory exactly following the EXP0.5/EXP1.1 adapter semantics.
def build_base_trajectory():
    p=example.parameters()
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(f['soil_boundary_temperature_c'].iloc[0])
    sv=set_matched_common_initial(p)
    tsim=np.arange(0,10800+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:p['InitialValues'][i] for i,name in enumerate(sv)}
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
      'u_blanket':const(0.0),'u_vent':const(BASE_VENT),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0),
    }
    records=[]; orig=csg_fun.ctl_csg1
    def logged_ctl(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        records.append(float(np.asarray(res[1]).reshape(-1)[0]))
        return res
    csg_fun.ctl_csg1=logged_ctl
    try:
        y=model.run((0.0,10800.0))
    finally:
        csg_fun.ctl_csg1=orig
    t_native=np.asarray(y['t'],dtype=float)
    common_idx=[]
    for t in np.arange(0,10801,900,dtype=float):
        hits=np.where(np.isclose(t_native,t,rtol=0,atol=1e-9))[0]
        if len(hits)!=1: raise AssertionError(f'Cannot sample BASE t={t}: hits={hits}')
        common_idx.append(int(hits[0]))
    native=[]; common=[]
    for j,idx in enumerate(common_idx):
        xv=np.asarray([float(np.asarray(y[name])[idx]) for name in sv],dtype=float)
        native.append(xv)
        common.append(common_from_vector(xv,sv,j*900.0))
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in sv)
    action_err=float(np.max(np.abs(np.asarray(records,dtype=float)-BASE_VENT)))
    return sv,native,common,action_err,bool(finite)

sv,baseline_native,baseline_common,baseline_action_err,baseline_finite=build_base_trajectory()

# One 900-s probe from a complete native BASE state.
def one_step_probe(x_start,j,v):
    p=example.parameters()
    # Constructor-compatible example-data time; replaced by benchmark forcing immediately after construction.
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T00:15'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(f.loc[j,'soil_boundary_temperature_c'])
    sv_local=list(p['StateVariable'])
    if sv_local!=sv: raise AssertionError('State-variable order changed')
    p['InitialValues']=np.asarray(x_start,dtype=float).copy()
    tsim=np.asarray([0.0,900.0])
    x0={name:p['InitialValues'][i] for i,name in enumerate(sv)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    actual_start=BASE_CLOCK+timedelta(seconds=900*j)
    actual_end=actual_start+timedelta(seconds=900)
    model.p['StartTime']=actual_start.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=actual_end.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    row=f.iloc[j]
    model.d={
      'f_Rad':const(row['global_radiation_w_m2']),
      'f_Tem':const(row['outdoor_temperature_c']),
      'f_RH':const(row['outdoor_rh_pct']/100.0),
      'f_CO2':const(row['outdoor_co2_ppm']),
      'f_Wind':const(row['wind_speed_m_s']),
      'f_Tsky':const(row['sky_temperature_c']),
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(v),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0),
    }
    records=[]; orig=csg_fun.ctl_csg1
    def logged_ctl(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        records.append(float(np.asarray(res[1]).reshape(-1)[0]))
        return res
    csg_fun.ctl_csg1=logged_ctl
    try:
        y=model.run((0.0,900.0))
    finally:
        csg_fun.ctl_csg1=orig
    final=np.asarray([float(np.asarray(y[name])[-1]) for name in sv],dtype=float)
    err=float(np.max(np.abs(np.asarray(records,dtype=float)-v)))
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in sv)
    return common_from_vector(final,sv,(j+1)*900.0),err,bool(finite),state_sha(final)

local_probes=[]
all_action_errors=[baseline_action_err]
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

base_grad=[float(baseline_common[j]['air_co2_ppm']-f.loc[j,'outdoor_co2_ppm']) for j in range(13)]
base_times=f['time_s'].to_numpy(dtype=float)
base_cross=zero_crossings(base_times,base_grad)

# Fixed-state CO2-only sweep at t=0.
sweep=[]
initial_native=baseline_native[0].copy()
co2_idx=sv.index('CO2')
for c0 in CO2_SWEEP:
    xs=initial_native.copy(); xs[co2_idx]=c0
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
 'model_id':'M2','model':'CSGtom','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
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
   'native_integration_step_s':30,
   'common_probe_horizon_s':900,
 },
 'interpretation_limit':'Local directional structure under normalized native ventilation command. No equal-airflow or empirical-validity claim.'
}
result['run_pass']=bool(result['runtime_audit']['max_requested_applied_error']==0.0 and result['runtime_audit']['all_states_finite'])
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']:
    raise SystemExit('M2 EXP1.2 runtime audit failed')
