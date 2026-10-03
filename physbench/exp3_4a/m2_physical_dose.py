import json, sys, datetime as _dt
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
REF=ROOT/'evidence/M2/action_grid_responses.csv'
OUT=ROOT/'evidence/EXP3_4A_M2'
OUT.mkdir(parents=True,exist_ok=True)

EVENTS=pd.read_csv(IN)
U9=np.round(np.arange(0.1,1.0,0.1),1)
U5=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
EXPECTED_INPUT_SHA='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
MODEL_COMMIT='bea8c3b0a1324162a4b5487db578aa674c8b587c'
DT=30.0

def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def locked_exp31_ah(t,vp):
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])

def const(v):
    return lambda t,vv=float(v):vv

assert sha256(IN)==EXPECTED_INPUT_SHA
assert len(EVENTS)==61 and EVENTS.event_id.nunique()==61

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

def event_datetime(r):
    hh=int(float(r.hour_decimal))
    mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60:
        hh=(hh+1)%24; mm=0
    return _dt.datetime(2017,int(r.month),int(r.day),hh,mm)

def build_model(r,u):
    p=base_parameters()
    p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    iv=NATIVE.copy()
    for name,val in {
        'T_air':float(r.event_Tair),
        'VP':float(r.in_vp_pa),
        'CO2':float(r.CO2_pre_ppm),
        'T_can':float(r.event_Tair)
    }.items():
        iv[SV.index(name)]=val
    p['InitialValues']=iv.copy()

    tsim=np.arange(0,1800+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:iv[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)

    st=event_datetime(r); en=st+_dt.timedelta(minutes=30)
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
      'u_blanket':const(0.0),
      'u_vent':const(float(u)),
      'u_venttop':const(1.0),
      'u_ventside':const(0.0),
      'u_venttopbot':const(0.0)
    }
    return model,iv

def one_event_action(r,u):
    model,iv=build_model(r,u)
    vent_records=[]; action_records=[]
    orig=csg_fun.ctl_csg1

    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        action_records.append(scalar(res[1]))
        vent_records.append(scalar(res[4]))
        return res

    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,1800.0))
    finally:
        csg_fun.ctl_csg1=orig

    if len(vent_records)!=60:
        raise AssertionError((int(r.event_id),u,'expected 60 M2 Euler ventilation evaluations',len(vent_records)))

    vent=np.asarray(vent_records,dtype=float)
    applied=np.asarray(action_records,dtype=float)
    if not np.all(np.isfinite(vent)):
        raise RuntimeError((int(r.event_id),u,'non-finite M2 ventilation flux'))

    t=np.asarray(y['t'],dtype=float)
    h_eff=float(model.D.Vair/model.D.area_floor)
    action_err=float(np.max(np.abs(applied-float(u))))
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in SV))
    init_err=max(
      abs(float(iv[SV.index('T_air')])-float(r.event_Tair)),
      abs(float(iv[SV.index('VP')])-float(r.in_vp_pa)),
      abs(float(iv[SV.index('CO2')])-float(r.CO2_pre_ppm)),
      abs(float(iv[SV.index('T_can')])-float(r.event_Tair))
    )

    out=[]
    for sec,n in [(900.0,30),(1800.0,60)]:
        hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-9))[0]
        if len(hits)!=1:
            raise AssertionError((int(r.event_id),u,sec,hits))
        j=int(hits[0])
        dose=float(np.sum(vent[:n])*DT)
        T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
        DN=float(dose/h_eff)
        out.append({
          'model_id':'M2','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,
          'horizon_min':int(sec/60),'action':float(u),
          'specific_volume_dose_m3_m2':dose,
          'air_volume_equiv':DN,
          'average_specific_flux_m3_m2_s':float(dose/sec),
          'average_ach_h_1':float(DN/(sec/3600.0)),
          'T':T,'AH':float(locked_exp31_ah(T,VP)),
          'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
          'init_max_abs_error':float(init_err),
          'action_max_abs_error':action_err,
          'air_volume_per_floor_area_m':h_eff,
          'floor_area_m2_per_m_length':float(model.D.area_floor),
          'air_volume_m3_per_m_length':float(model.D.Vair)
        })
    return out,finite

rows=[]; all_finite=True
for _,r in EVENTS.iterrows():
    for u in U9:
        rr,finite=one_event_action(r,float(u))
        rows.extend(rr); all_finite=all_finite and finite
    print('M2 event',int(r.event_id),'done',flush=True)

df=pd.DataFrame(rows).sort_values(['event_id','horizon_min','action']).reset_index(drop=True)
assert len(df)==61*2*9
assert not df.duplicated(['event_id','horizon_min','action']).any()
assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
df.to_csv(OUT/'physical_dose_grid.csv',index=False,float_format='%.12g')

# Same-run reconstruction against untouched EXP3.1.
ref=pd.read_csv(REF)
u5=df[df.action.isin(U5)].copy()
cmp=ref.merge(
    u5[['event_id','horizon_min','action','T','AH']],
    on=['event_id','horizon_min','action'],suffixes=('_ref','_dose'),validate='one_to_one'
)
assert len(cmp)==61*2*5
cmp['abs_T_error']=np.abs(cmp.T_ref-cmp.T_dose)
cmp['abs_AH_error']=np.abs(cmp.AH_ref-cmp.AH_dose)
cmp['scaled_T_error']=cmp.abs_T_error/np.maximum(1.0,np.maximum(np.abs(cmp.T_ref),np.abs(cmp.T_dose)))
cmp['scaled_AH_error']=cmp.abs_AH_error/np.maximum(1.0,np.maximum(np.abs(cmp.AH_ref),np.abs(cmp.AH_dose)))
cmp.to_csv(OUT/'same_run_exp3_1_reconstruction.csv',index=False,float_format='%.12g')
max_abs_T=float(cmp.abs_T_error.max())
max_abs_AH=float(cmp.abs_AH_error.max())
max_scaled_T=float(cmp.scaled_T_error.max())
max_scaled_AH=float(cmp.scaled_AH_error.max())

hvals=df.air_volume_per_floor_area_m.to_numpy(float)
assert np.max(hvals)-np.min(hvals)<=1e-12
H_EFF=float(hvals[0])

p=base_parameters()
net_top=float(p['Atop_vent']*p['r_net'])
max_top_per_floor=float(net_top/float(df.floor_area_m2_per_m_length.iloc[0]))

summary={
 'experiment':'PhysBench-GH EXP3.4A','model_id':'M2','model':'CSGtom',
 'frozen_commit':MODEL_COMMIT,'input_sha256':sha256(IN),
 'events':61,'actions':U9.tolist(),'rows':int(len(df)),
 'physical_flux_definition':'ctl_csg1 return Vent',
 'physical_flux_units':'m3 m-2 s-1',
 'air_volume_per_floor_area_m':H_EFF,
 'floor_area_m2_per_m_length':float(df.floor_area_m2_per_m_length.iloc[0]),
 'air_volume_m3_per_m_length':float(df.air_volume_m3_per_m_length.iloc[0]),
 'max_net_top_vent_area_m2_per_m_length':net_top,
 'max_top_aperture_per_floor_area_m2_m2':max_top_per_floor,
 'all_finite':bool(all_finite and np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()),
 'max_init_error':float(df.init_max_abs_error.max()),
 'max_action_error':float(df.action_max_abs_error.max()),
 'same_run_exp3_1_max_abs_T_error':max_abs_T,
 'same_run_exp3_1_max_abs_AH_error':max_abs_AH,
 'same_run_exp3_1_max_scaled_T_error':max_scaled_T,
 'same_run_exp3_1_max_scaled_AH_error':max_scaled_AH,
 'native_integration_step_s':30
}
summary['gate_pass']=bool(
 summary['input_sha256']==EXPECTED_INPUT_SHA and summary['events']==61 and summary['rows']==61*2*9
 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0
 and summary['same_run_exp3_1_max_abs_T_error']<=1e-9
 and summary['same_run_exp3_1_max_scaled_T_error']<=1e-11
 and summary['same_run_exp3_1_max_scaled_AH_error']<=1e-11
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('M2 EXP3.4A runtime/reconstruction gate failed')
