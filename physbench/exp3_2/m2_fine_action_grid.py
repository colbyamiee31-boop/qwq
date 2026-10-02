import json, sys, datetime as _dt
from pathlib import Path
import numpy as np, pandas as pd
from common import *

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/EXP3_2_M2'; OUT.mkdir(parents=True,exist_ok=True)
EVENTS=pd.read_csv(IN)
ACTIONS=np.round(np.arange(0.1,1.0,0.1),1)

def const(v):
    return lambda t, vv=float(v): vv

def run_action(r,vent):
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    sv=list(p['StateVariable']); iv=p['InitialValues'].copy()
    for name,val in {
        'T_air':float(r.event_Tair),'VP':float(r.in_vp_pa),
        'CO2':float(r.CO2_pre_ppm),'T_can':float(r.event_Tair)
    }.items():
        iv[sv.index(name)]=val
    p['InitialValues']=iv
    tsim=np.arange(0,1800+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:p['InitialValues'][i] for i,name in enumerate(p['StateVariable'])}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    st=_dt.datetime(2017,int(r.month),int(r.day),hh,mm); en=st+_dt.timedelta(minutes=30)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':const(r.event_Iglob),'f_Tem':const(r.event_Tout),
      'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),
      'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(r.event_Windsp),
      'f_Tsky':const(r.sky_temperature_c_proxy)
    }
    model.U={
      'u_blanket':lambda t:0.0,'u_vent':lambda t,v=vent:v,
      'u_venttop':lambda t:1.0,'u_ventside':lambda t:0.0,'u_venttopbot':lambda t:0.0
    }
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        rec.append(float(np.asarray(res[1]).reshape(-1)[0]))
        return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,1800.0))
    finally: csg_fun.ctl_csg1=orig
    t=np.asarray(y['t'],dtype=float); trace=[]
    for sec in (900.0,1800.0):
        hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-9))[0]
        if len(hits)!=1: raise AssertionError((r.event_id,sec,hits))
        j=int(hits[0]); T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
        trace.append({'horizon_min':int(sec/60),'T':T,'VP':VP,'AH':float(ah_from_t_vp(T,VP))})
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in p['StateVariable'])
    err=max(abs(v-vent) for v in rec) if rec else float('inf')
    init_err=max(
        abs(float(iv[sv.index('T_air')])-float(r.event_Tair)),
        abs(float(iv[sv.index('VP')])-float(r.in_vp_pa)),
        abs(float(iv[sv.index('CO2')])-float(r.CO2_pre_ppm)),
        abs(float(iv[sv.index('T_can')])-float(r.event_Tair))
    )
    return trace,float(err),bool(finite),float(init_err)

rows=[]; audits=[]
for _,r in EVENTS.iterrows():
    for u in ACTIONS:
        trace,ae,finite,ie=run_action(r,float(u))
        audits.append({'event_id':int(r.event_id),'action':u,'init_max_abs_error':ie,'action_max_abs_error':ae,'finite':finite})
        for q in trace:
            rows.append({
                'event_id':int(r.event_id),'team':r.team,'timestamp':r.timestamp,'event_date':r.event_date,'daynight':r.daynight,
                'horizon_min':q['horizon_min'],'action':u,
                'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
                'T':q['T'],'AH':q['AH']
            })

df=pd.DataFrame(rows).sort_values(['event_id','horizon_min','action'])
ad=pd.DataFrame(audits).sort_values(['event_id','action'])
df.to_csv(OUT/'action_grid_responses.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
summary={
 'model':'M2','experiment':'EXP3.2 fine action grid','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
 'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),
 'actions':ACTIONS.tolist(),'rows':len(df),'all_finite':bool(ad.finite.all()),
 'max_init_error':float(ad.init_max_abs_error.max()),'max_action_error':float(ad.action_max_abs_error.max()),
 'native_integration_step_s':30
}
summary['gate_pass']=bool(summary['events']==61 and summary['rows']==61*9*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M2 EXP3.2 runtime gate failed')
