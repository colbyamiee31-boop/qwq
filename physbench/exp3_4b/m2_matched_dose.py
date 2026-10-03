import json, hashlib, sys, datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG))

from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
A4=ROOT/'evidence/EXP3_4A_FINAL'
GRID_PATH=ROOT/'evidence/EXP3_4A_M2/physical_dose_grid.csv'
TARGET_PATH=A4/'common_physical_dose_targets_for_EXP3_4B.csv'
OUT=ROOT/'evidence/EXP3_4B_M2'; OUT.mkdir(parents=True,exist_ok=True)

EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
EXPECTED_TARGET_SHA='e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292'
MODEL_COMMIT='bea8c3b0a1324162a4b5487db578aa674c8b587c'
U9=np.round(np.arange(0.1,1.0,0.1),1)
Q=np.array([0.0,0.25,0.5,0.75,1.0],dtype=float)
DT=30.0

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def ah_from_t_vp(t,vp):
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])

def const(v):
    return lambda t,vv=float(v):vv

EVENTS=pd.read_csv(IN)
TARGETS=pd.read_csv(TARGET_PATH)
GRID=pd.read_csv(GRID_PATH)
assert sha256(IN)==EXPECTED_INPUT_SHA
assert sha256(TARGET_PATH)==EXPECTED_TARGET_SHA
assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61
assert len(GRID)==61*2*9
assert not GRID.duplicated(['event_id','horizon_min','action']).any()
assert set(np.round(GRID.action.unique(),10))==set(U9)
ELIG=TARGETS[TARGETS.eligible_for_exp3_4b.astype(bool)].copy()
assert len(ELIG)==1060
assert ELIG.groupby(['event_id','horizon_min','coordinate']).size().eq(5).all()

def base_parameters():
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    return p

P0=base_parameters()
SV=list(P0['StateVariable'])
NATIVE=np.asarray(P0['InitialValues'],dtype=float).copy()

def event_datetime(r):
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    return _dt.datetime(2017,int(r.month),int(r.day),hh,mm)

def build_model(r,u,horizon):
    p=base_parameters()
    p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    iv=NATIVE.copy()
    for name,val in {'T_air':float(r.event_Tair),'VP':float(r.in_vp_pa),
                     'CO2':float(r.CO2_pre_ppm),'T_can':float(r.event_Tair)}.items():
        iv[SV.index(name)]=val
    p['InitialValues']=iv.copy()
    sec=int(horizon)*60
    tsim=np.arange(0,sec+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:iv[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    st=event_datetime(r); en=st+_dt.timedelta(seconds=sec)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':const(r.event_Iglob),
      'f_Tem':const(r.event_Tout),
      'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),
      'f_CO2':const(r.outdoor_co2_ppm_fixed),
      'f_Wind':const(r.event_Windsp),
      'f_Tsky':const(r.sky_temperature_c_proxy)
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(float(u)),
      'u_venttop':const(1.0),'u_ventside':const(0.0),'u_venttopbot':const(0.0)
    }
    return model,iv

def evaluate(r,u,horizon):
    h=int(horizon); assert h in (15,30)
    model,iv=build_model(r,u,h)
    vent_records=[]; action_records=[]
    orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        action_records.append(scalar(res[1]))
        vent_records.append(scalar(res[4]))
        return res
    csg_fun.ctl_csg1=logged
    sec=h*60
    try:
        y=model.run((0.0,float(sec)))
    finally:
        csg_fun.ctl_csg1=orig
    n=sec//30
    if len(vent_records)!=n:
        raise AssertionError((int(r.event_id),u,h,'expected M2 ventilation evaluations',n,len(vent_records)))
    vent=np.asarray(vent_records,dtype=float)
    applied=np.asarray(action_records,dtype=float)
    if not np.all(np.isfinite(vent)): raise RuntimeError((int(r.event_id),u,h,'nonfinite vent'))
    for k in SV:
        if not np.all(np.isfinite(np.asarray(y[k],dtype=float))):
            raise RuntimeError((int(r.event_id),u,h,'nonfinite state',k))
    t=np.asarray(y['t'],dtype=float)
    hits=np.where(np.isclose(t,float(sec),rtol=0,atol=1e-9))[0]
    if len(hits)!=1: raise AssertionError((int(r.event_id),u,h,'terminal time',hits))
    j=int(hits[0])
    dose=float(np.sum(vent)*DT)
    heff=float(model.D.Vair/model.D.area_floor)
    T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
    init_err=max(
      abs(float(iv[SV.index('T_air')])-float(r.event_Tair)),
      abs(float(iv[SV.index('VP')])-float(r.in_vp_pa)),
      abs(float(iv[SV.index('CO2')])-float(r.CO2_pre_ppm)),
      abs(float(iv[SV.index('T_can')])-float(r.event_Tair))
    )
    return {
      'native_command':float(u),'DV':dose,'DN':float(dose/heff),
      'T':T,'AH':float(ah_from_t_vp(T,VP)),
      'action_error':float(np.max(np.abs(applied-float(u)))),
      'init_error':float(init_err),'air_volume_per_floor_area_m':heff
    }

EVENT_MAP={int(r.event_id):r for _,r in EVENTS.iterrows()}

def bracket(event_id,horizon,coordinate,target):
    col='specific_volume_dose_m3_m2' if coordinate=='DV' else 'air_volume_equiv'
    g=GRID[(GRID.event_id==event_id)&(GRID.horizon_min==horizon)].sort_values('action')
    assert len(g)==9 and np.allclose(g.action.to_numpy(float),U9,rtol=0,atol=1e-12)
    d=g[col].to_numpy(float); u=g.action.to_numpy(float)
    tolmono=1e-10*max(1.0,float(np.max(np.abs(d))))
    assert np.all(np.diff(d)>=-tolmono)
    target=float(target)
    if target < d[0]-1e-8 or target > d[-1]+1e-8:
        raise AssertionError((event_id,horizon,coordinate,target,d[0],d[-1]))
    idx=int(np.searchsorted(d,target,side='left'))
    if idx<=0: return float(u[0]),float(u[0]),float(d[0]),float(d[0])
    if idx>=len(d): return float(u[-1]),float(u[-1]),float(d[-1]),float(d[-1])
    if abs(target-d[idx])<=1e-12*max(1.0,abs(target)):
        return float(u[idx]),float(u[idx]),float(d[idx]),float(d[idx])
    return float(u[idx-1]),float(u[idx]),float(d[idx-1]),float(d[idx])

def solve(row):
    eid=int(row.event_id); h=int(row.horizon_min); coord=str(row.coordinate); target=float(row.target_dose)
    width=float(row.common_width)
    tol=max(1e-8,1e-6*max(1.0,abs(target),abs(width)))
    ulo,uhi,dlo,dhi=bracket(eid,h,coord,target)
    r=EVENT_MAP[eid]; initial_ulo,initial_uhi=ulo,uhi
    if ulo==uhi:
        res=evaluate(r,ulo,h); n_eval=1
    else:
        if dhi<=dlo: raise AssertionError((eid,h,coord,'nonpositive bracket dose span',dlo,dhi))
        useed=ulo+(target-dlo)*(uhi-ulo)/(dhi-dlo)
        useed=float(np.clip(useed,ulo,uhi))
        res=evaluate(r,useed,h); n_eval=1
        got=float(res[coord])
        if abs(got-target)>tol:
            if got<target: ulo,dlo=float(res['native_command']),got
            else: uhi,dhi=float(res['native_command']),got
            for _ in range(30):
                umid=0.5*(ulo+uhi)
                res=evaluate(r,umid,h); n_eval+=1
                got=float(res[coord])
                if abs(got-target)<=tol: break
                if got<target: ulo,dlo=float(res['native_command']),got
                else: uhi,dhi=float(res['native_command']),got
            else:
                raise RuntimeError((eid,h,coord,float(row.q),'dose root did not converge',target,got,tol,ulo,uhi))
    achieved=float(res[coord]); err=achieved-target
    if abs(err)>tol:
        raise RuntimeError((eid,h,coord,float(row.q),'dose tolerance fail',target,achieved,err,tol))
    if not (0.1-1e-12 <= res['native_command'] <= 0.9+1e-12):
        raise RuntimeError((eid,h,coord,float(row.q),'command out of domain',res['native_command']))
    return {
      'model_id':'M2','event_id':eid,'team':row.team,'event_date':row.event_date,'daynight':row.daynight,
      'horizon_min':h,'coordinate':coord,'q':float(row.q),
      'target_dose':target,'achieved_dose':achieved,'dose_error':float(err),'dose_tolerance':float(tol),
      'native_command':float(res['native_command']),
      'initial_bracket_u_lo':float(initial_ulo),'initial_bracket_u_hi':float(initial_uhi),
      'root_model_evaluations':int(n_eval),
      'T':float(res['T']),'AH':float(res['AH']),
      'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
      'action_error':float(res['action_error']),'init_error':float(res['init_error']),
      'air_volume_per_floor_area_m':float(res['air_volume_per_floor_area_m'])
    }

rows=[]
for cell_key,g in ELIG.groupby(['event_id','horizon_min','coordinate'],sort=True):
    for _,row in g.sort_values('q').iterrows():
        rows.append(solve(row))
    print('M2 matched cell',cell_key,'done',flush=True)

df=pd.DataFrame(rows).sort_values(['coordinate','horizon_min','event_id','q']).reset_index(drop=True)
assert len(df)==1060
assert not df.duplicated(['event_id','horizon_min','coordinate','q']).any()
assert df.groupby(['event_id','horizon_min','coordinate']).size().eq(5).all()
assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
assert float(df.action_error.max())<=1e-12
assert float(df.init_error.max())<=1e-8
assert np.all(np.abs(df.dose_error.to_numpy(float))<=df.dose_tolerance.to_numpy(float)+1e-15)
df.to_csv(OUT/'matched_dose_responses.csv',index=False,float_format='%.12g')

counts=df.groupby(['coordinate','horizon_min']).size().to_dict()
heff=df.air_volume_per_floor_area_m.to_numpy(float)
summary={
 'experiment':'PhysBench-GH EXP3.4B','model_id':'M2','model':'CSGtom',
 'frozen_commit':MODEL_COMMIT,'input_sha256':sha256(IN),'target_sha256':sha256(TARGET_PATH),
 'rows':int(len(df)),
 'counts':{f'{k[0]}_{int(k[1])}':int(v) for k,v in counts.items()},
 'max_abs_dose_error':float(np.abs(df.dose_error).max()),
 'max_dose_error_fraction_of_tolerance':float(np.max(np.abs(df.dose_error)/df.dose_tolerance)),
 'max_action_error':float(df.action_error.max()),
 'max_init_error':float(df.init_error.max()),
 'max_root_model_evaluations':int(df.root_model_evaluations.max()),
 'air_volume_per_floor_area_m_min':float(heff.min()),
 'air_volume_per_floor_area_m_max':float(heff.max()),
 'all_finite':bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()),
 'native_integration_step_s':30
}
summary['gate_pass']=bool(
 summary['input_sha256']==EXPECTED_INPUT_SHA and summary['target_sha256']==EXPECTED_TARGET_SHA
 and summary['rows']==1060
 and summary['counts']=={'DN_15':305,'DN_30':305,'DV_15':210,'DV_30':240}
 and summary['all_finite'] and summary['max_action_error']<=1e-12 and summary['max_init_error']<=1e-8
 and summary['max_dose_error_fraction_of_tolerance']<=1.0+1e-12
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M2 EXP3.4B gate failed')
