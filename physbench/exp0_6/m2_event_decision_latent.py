import argparse, hashlib, json, sys, datetime as _dt
from pathlib import Path
import numpy as np
import pandas as pd

from common import *
from generate_prehistory import generate_prehistory

CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
EVENTS=pd.read_csv(IN)
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
HISTORY_IDS=['HNR']+list(CFG['decision_primary_history_ids'])
NEUTRAL=float(CFG['neutral_prehistory_vent_command'])
DURATION_H=int(CFG['primary_prehistory_h'])
DT=30.0

parser=argparse.ArgumentParser()
parser.add_argument('--history-id',required=True,choices=HISTORY_IDS)
args=parser.parse_args()
HID=args.history_id
OUT=ROOT/f'evidence/EXP0_6C_M2_{HID}'
OUT.mkdir(parents=True,exist_ok=True)

EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
assert sha256(IN)==EXPECTED_INPUT_SHA
assert EVENTS.event_id.nunique()==61 and len(EVENTS)==61
assert list(CFG['decision_primary_history_ids'])==['H00','H01','H04','H06','H07','H10','H11','H13','H16']

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])

def const(v):
    return lambda t,vv=float(v):vv

def locked_exp31_ah(t,vp):
    # Deliberately preserve the locked EXP3.1 utility representation exactly.
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

def base_parameters():
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'
    p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900
    p['dt']=30
    p['ctl_vent_type']='timebasedControl'
    p['ctl_blank_type']='timebasedControl'
    return p

P0=base_parameters()
SV=list(P0['StateVariable'])
NATIVE=np.asarray(P0['InitialValues'],dtype=float).copy()

def common_state(x):
    x=np.asarray(x,dtype=float)
    T=float(x[SV.index('T_air')]); VP=float(x[SV.index('VP')])
    return {
      'air_temperature_c':T,'air_vapor_pressure_pa':VP,
      'air_rh_pct':float(rh_from_t_vp(T,VP)),
      'air_co2_ppm':float(x[SV.index('CO2')]),
      'canopy_temperature_c':float(x[SV.index('T_can')])
    }

def forcing_sha(df):
    raw=df.to_csv(index=False,float_format='%.12g').encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def pwc(values):
    vals=np.asarray(values,dtype=float)
    def fn(t):
        sec=float(np.asarray(t).reshape(-1)[0])
        idx=int(np.floor(sec/900.0+1e-12))
        return float(vals[max(0,min(idx,len(vals)-1))])
    return fn

def surrogate_event_datetime(r):
    hh=int(float(r.hour_decimal))
    mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60:
        hh=(hh+1)%24
        mm=0
    return _dt.datetime(2017,int(r.month),int(r.day),hh,mm)

def prehistory_state(r):
    if HID=='HNR':
        return NATIVE.copy(),{
          'forcing_sha256':None,'all_states_finite':True,'max_requested_applied_error':0.0,
          'clock_error_h':0.0
        }
    hist=generate_prehistory(HID,DURATION_H,float(r.hour_decimal))
    intervals=hist.iloc[:-1].reset_index(drop=True)
    duration=float(DURATION_H*3600)
    p=base_parameters()
    p['T_soilbound']=float(intervals.soil_boundary_temperature_c.iloc[0])
    p['InitialValues']=NATIVE.copy()
    tsim=np.arange(0,duration+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:p['InitialValues'][i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    en=surrogate_event_datetime(r)
    st=en-_dt.timedelta(hours=DURATION_H)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':pwc(intervals.global_radiation_w_m2),
      'f_Tem':pwc(intervals.outdoor_temperature_c),
      'f_RH':pwc(intervals.outdoor_rh_pct.to_numpy(float)/100.0),
      'f_CO2':pwc(intervals.outdoor_co2_ppm),
      'f_Wind':pwc(intervals.wind_speed_m_s),
      'f_Tsky':pwc(intervals.sky_temperature_c)
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(NEUTRAL),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0)
    }
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        rec.append(scalar(res[1]))
        return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,duration))
    finally:
        csg_fun.ctl_csg1=orig
    x=np.asarray([float(np.asarray(y[name])[-1]) for name in SV],dtype=float)
    finite=bool(all(np.all(np.isfinite(np.asarray(y[name],dtype=float))) for name in SV))
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-NEUTRAL))) if rec else float('inf')
    return x,{
      'forcing_sha256':forcing_sha(hist),'all_states_finite':finite,
      'max_requested_applied_error':err,'clock_error_h':0.0
    }

def projected_state(latent,r):
    z=np.asarray(latent,dtype=float).copy()
    z[SV.index('T_air')]=float(r.event_Tair)
    z[SV.index('VP')]=float(r.in_vp_pa)
    z[SV.index('CO2')]=float(r.CO2_pre_ppm)
    z[SV.index('T_can')]=float(r.event_Tair)
    return z

def build_action_model(latent,r,vent):
    p=base_parameters()
    p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    z=projected_state(latent,r)
    p['InitialValues']=z.copy()
    tsim=np.arange(0,1800+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:z[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    st=surrogate_event_datetime(r)
    en=st+_dt.timedelta(minutes=30)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':const(r.event_Iglob),'f_Tem':const(r.event_Tout),
      'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),
      'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(r.event_Windsp),
      'f_Tsky':const(r.sky_temperature_c_proxy)
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(vent),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0)
    }
    return model,z

def run_action(latent,r,vent):
    model,z=build_action_model(latent,r,vent)
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        rec.append(scalar(res[1]))
        return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,1800.0))
    finally:
        csg_fun.ctl_csg1=orig
    t=np.asarray(y['t'],dtype=float)
    trace=[]
    for sec in (900.0,1800.0):
        hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-9))[0]
        if len(hits)!=1:
            raise AssertionError((int(r.event_id),HID,vent,sec,hits))
        j=int(hits[0])
        T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
        trace.append({'horizon_min':int(sec/60),'T':T,'VP':VP,'AH':float(locked_exp31_ah(T,VP))})
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in SV))
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-vent))) if rec else float('inf')
    init_err=max(
      abs(float(z[SV.index('T_air')])-float(r.event_Tair)),
      abs(float(z[SV.index('VP')])-float(r.in_vp_pa)),
      abs(float(z[SV.index('CO2')])-float(r.CO2_pre_ppm)),
      abs(float(z[SV.index('T_can')])-float(r.event_Tair))
    )
    return trace,err,finite,float(init_err)

rows=[]; audits=[]; latent_meta=[]; latent_states={}
for _,r in EVENTS.iterrows():
    eid=int(r.event_id)
    latent,pm=prehistory_state(r)
    latent_states[f'E{eid}']=np.asarray(latent,dtype=np.float64)
    pre=common_state(latent)
    post=common_state(projected_state(latent,r))
    latent_meta.append({
      'model_id':'M2','history_id':HID,'event_id':eid,'timestamp':r.timestamp,
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
        audits.append({'model_id':'M2','history_id':HID,'event_id':eid,'action':float(u),
                       'init_max_abs_error':ie,'action_max_abs_error':ae,'finite':finite})
        for q in trace:
            rows.append({
              'model_id':'M2','history_id':HID,'event_id':eid,'team':r.team,'timestamp':r.timestamp,
              'event_date':r.event_date,'daynight':r.daynight,'horizon_min':q['horizon_min'],'action':float(u),
              'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
              'T':q['T'],'AH':q['AH']
            })
    print('M2',HID,'event',eid,'done',flush=True)

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
 'experiment':'PhysBench-GH EXP0.6C','model':'M2','history_id':HID,
 'frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
 'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),'actions':ACTIONS.tolist(),
 'rows':int(len(df)),'all_finite':bool(ad.finite.all()),
 'max_init_error':float(ad.init_max_abs_error.max()),
 'max_action_error':float(ad.action_max_abs_error.max()),
 'prehistory_all_finite':bool(lm.prehistory_finite.all()),
 'prehistory_max_action_error':float(lm.prehistory_max_action_error.max()),
 'prehistory_max_clock_error_h':float(lm.prehistory_clock_error_h.max()),
 'same_run_state_restore_exact':bool(restore_exact),
 'native_integration_step_s':30
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
    raise SystemExit('M2 EXP0.6C runtime gate failed')
