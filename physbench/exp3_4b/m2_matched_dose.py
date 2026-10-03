import json, sys, datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    INPUT_SHA256,M2_COMMIT,COORD_COL,sha256,validate_target_file,solve_target
)

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
TARGETS_PATH=ROOT/'evidence/EXP3_4A_FINAL/common_physical_dose_targets_for_EXP3_4B.csv'
GRID_PATH=ROOT/'evidence/EXP3_4A_M2/physical_dose_grid.csv'
LOCK_SUMMARY=ROOT/'evidence/EXP3_4A_M2/summary.json'
OUT=ROOT/'evidence/EXP3_4B_M2'; OUT.mkdir(parents=True,exist_ok=True)
DT=30.0

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float); return 610.78*np.exp(17.2694*t/(t+238.3))
def locked_exp31_ah(t,vp): return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)
def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])
def const(v): return lambda t,vv=float(v):vv

assert sha256(IN)==INPUT_SHA256
EVENTS=pd.read_csv(IN); assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61
TARGETS=validate_target_file(TARGETS_PATH)
GRID=pd.read_csv(GRID_PATH)
LOCK=json.loads(LOCK_SUMMARY.read_text())
assert LOCK['gate_pass'] and LOCK['frozen_commit']==M2_COMMIT and LOCK['input_sha256']==INPUT_SHA256
assert len(GRID)==61*2*9 and not GRID.duplicated(['event_id','horizon_min','action']).any()

def base_parameters():
    p=example.parameters(); p['outdoorDataFileURL']=str(CSG/'data/example_data.xls'); p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T00:15'; p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'; return p

P0=base_parameters(); SV=list(P0['StateVariable']); NATIVE=np.asarray(P0['InitialValues'],dtype=float).copy()

def event_datetime(r):
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    return _dt.datetime(2017,int(r.month),int(r.day),hh,mm)

def build_model(r,u):
    p=base_parameters(); p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    iv=NATIVE.copy()
    for name,val in {'T_air':float(r.event_Tair),'VP':float(r.in_vp_pa),'CO2':float(r.CO2_pre_ppm),'T_can':float(r.event_Tair)}.items():
        iv[SV.index(name)]=val
    p['InitialValues']=iv.copy(); tsim=np.arange(0,900+p['dtsim'],p['dtsim'],dtype=float); x0={name:iv[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p); st=event_datetime(r); en=st+_dt.timedelta(minutes=15)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M'); model.D=csg_shape.csg_shape(model.p)
    model.d={'f_Rad':const(r.event_Iglob),'f_Tem':const(r.event_Tout),'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),
             'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(r.event_Windsp),'f_Tsky':const(r.sky_temperature_c_proxy)}
    model.U={'u_blanket':const(0.0),'u_vent':const(float(u)),'u_venttop':const(1.0),'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    return model,iv

CACHE={}
def simulate(r,u):
    u=float(u); key=(int(r.event_id),round(u,15))
    if key in CACHE: return CACHE[key].copy()
    model,iv=build_model(r,u); vent_records=[]; action_records=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); action_records.append(scalar(res[1])); vent_records.append(scalar(res[4])); return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,900.0))
    finally: csg_fun.ctl_csg1=orig
    if len(vent_records)!=30: raise AssertionError((int(r.event_id),u,'expected 30 M2 ventilation evaluations',len(vent_records)))
    vent=np.asarray(vent_records,dtype=float); applied=np.asarray(action_records,dtype=float)
    t=np.asarray(y['t'],dtype=float); hits=np.where(np.isclose(t,900.0,rtol=0,atol=1e-9))[0]
    if len(hits)!=1: raise AssertionError((int(r.event_id),u,'missing H900'))
    j=int(hits[0]); dv=float(np.sum(vent)*DT); h_eff=float(model.D.Vair/model.D.area_floor); dn=float(dv/h_eff)
    T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in SV) and np.all(np.isfinite(vent)))
    init_err=max(abs(float(iv[SV.index('T_air')])-float(r.event_Tair)),abs(float(iv[SV.index('VP')])-float(r.in_vp_pa)),
                 abs(float(iv[SV.index('CO2')])-float(r.CO2_pre_ppm)),abs(float(iv[SV.index('T_can')])-float(r.event_Tair)))
    rec={'model_id':'M2','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':15,
         'native_command':u,'specific_volume_dose_m3_m2':dv,'air_volume_equiv':dn,'T':T,'AH':float(locked_exp31_ah(T,VP)),
         'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'init_max_abs_error':float(init_err),
         'action_max_abs_error':float(np.max(np.abs(applied-u))),'air_volume_per_floor_area_m':h_eff}
    vals=np.array([rec[k] for k in ['native_command','specific_volume_dose_m3_m2','air_volume_equiv','T','AH','init_max_abs_error','action_max_abs_error']],float)
    if not finite or not np.all(np.isfinite(vals)): raise RuntimeError((int(r.event_id),u,'nonfinite M2 matched run'))
    if rec['init_max_abs_error']>1e-8 or rec['action_max_abs_error']>1e-12: raise RuntimeError((int(r.event_id),u,'M2 runtime gate',rec))
    CACHE[key]=rec.copy(); return rec

rows=[]
use=TARGETS[(TARGETS.horizon_min==15)&TARGETS.coordinate.isin(['DV','DN'])&TARGETS.eligible_for_exp3_4b].copy()
for eid,tt in use.groupby('event_id',sort=True):
    r=EVENTS[EVENTS.event_id==eid].iloc[0]; gg=GRID[(GRID.event_id==eid)&(GRID.horizon_min==15)].copy()
    if len(gg)!=9: raise AssertionError(('missing M2 U9 grid',eid))
    for _,trow in tt.sort_values(['coordinate','q']).iterrows():
        rec,nit,blo,bhi,tol=solve_target(gg,str(trow.coordinate),float(trow.target_dose),lambda u,rr=r:simulate(rr,u))
        actual=float(rec[COORD_COL[str(trow.coordinate)]]); err=abs(actual-float(trow.target_dose))
        if err>tol: raise AssertionError(('M2 dose gate',eid,trow.coordinate,trow.q,err,tol))
        rows.append({**rec,'coordinate':str(trow.coordinate),'q':float(trow.q),'target_dose':float(trow.target_dose),
          'actual_dose':actual,'dose_abs_error':err,'dose_tolerance':tol,'dose_match_pass':True,
          'common_low':float(trow.common_low),'common_high':float(trow.common_high),'common_width':float(trow.common_width),
          'locked_bracket_u_lo':blo,'locked_bracket_u_hi':bhi,'inversion_evaluations':int(nit)})
    print('M2 event',int(eid),'done',flush=True)

df=pd.DataFrame(rows).sort_values(['coordinate','event_id','q']).reset_index(drop=True)
assert len(df)==515 and not df.duplicated(['coordinate','event_id','q']).any()
assert df[df.coordinate=='DV'].event_id.nunique()==42 and df[df.coordinate=='DN'].event_id.nunique()==61
assert np.all(df.native_command.to_numpy(float)>=0.1-1e-12) and np.all(df.native_command.to_numpy(float)<=0.9+1e-12)
assert bool(df.dose_match_pass.all()) and np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
df.to_csv(OUT/'matched_physical_dose_outcomes.csv',index=False,float_format='%.12g')
summary={'experiment':'PhysBench-GH EXP3.4B','model_id':'M2','frozen_commit':M2_COMMIT,'input_sha256':sha256(IN),
 'target_sha256':sha256(TARGETS_PATH),'rows':int(len(df)),'dv15_events':int(df[df.coordinate=='DV'].event_id.nunique()),
 'dn15_events':int(df[df.coordinate=='DN'].event_id.nunique()),'max_DV_abs_error':float(df[df.coordinate=='DV'].dose_abs_error.max()),
 'max_DN_abs_error':float(df[df.coordinate=='DN'].dose_abs_error.max()),'max_action_error':float(df.action_max_abs_error.max()),
 'max_init_error':float(df.init_max_abs_error.max()),'max_inversion_evaluations':int(df.inversion_evaluations.max()),
 'all_finite':True,'native_integration_step_s':30,'gate_pass':True}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
