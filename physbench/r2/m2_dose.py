import json,sys,datetime as _dt
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG)); sys.path.insert(0,str(ROOT/'physbench/r1_2'))
from common import sat_vp_pa
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun,csg_shape
OUT=ROOT/'evidence/R2/M2'; OUT.mkdir(parents=True,exist_ok=True)
INPUTS=[('primary',ROOT/'physbench/r2/generated/PRIMARY_97_R2_INPUT.csv'),('strict',ROOT/'physbench/r2/generated/STRICT_36_R2_INPUT.csv')]
def const(v): return lambda t,vv=float(v):vv
def base_parameters():
    p=example.parameters(); p['outdoorDataFileURL']=str(CSG/'data/example_data.xls'); p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'; p['dtsim']=900; p['dt']=30; p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'; return p
def run_arm(r,u):
    if not 0<=float(u)<=1: raise AssertionError(('action_bounds',r.event_id,u))
    p=base_parameters(); p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed); sv=list(p['StateVariable']); iv=np.asarray(p['InitialValues'],float).copy()
    for name,val in {'T_air':r.event_Tair,'VP':r.in_vp_pa,'CO2':r.CO2_pre_ppm,'T_can':r.canopy_temperature_c_closure}.items(): iv[sv.index(name)]=float(val)
    p['InitialValues']=iv.copy(); tsim=np.arange(0,900+p['dtsim'],p['dtsim'],dtype=float); x0={name:iv[i] for i,name in enumerate(sv)}; model=CSG_Climate(tsim,p['dt'],x0,p)
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    st=_dt.datetime(2017,int(r.month),int(r.day),hh,mm); en=st+_dt.timedelta(minutes=15)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M'); model.D=csg_shape.csg_shape(model.p)
    model.d={'f_Rad':const(r.event_Iglob),'f_Tem':const(r.event_Tout),'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(r.event_Windsp),'f_Tsky':const(r.sky_temperature_c_proxy)}
    model.U={'u_blanket':const(0.0),'u_vent':const(float(u)),'u_venttop':const(1.0),'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    vents=[]; acts=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); acts.append(float(np.asarray(res[1]).reshape(-1)[0])); vents.append(float(np.asarray(res[4]).reshape(-1)[0])); return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,900.0))
    finally: csg_fun.ctl_csg1=orig
    if len(vents)!=30: raise AssertionError((r.event_id,u,len(vents)))
    vent=np.asarray(vents,float); applied=np.asarray(acts,float); dose=float(np.sum(vent)*30.0); H=float(model.D.Vair/model.D.area_floor)
    init_err=max(abs(float(iv[sv.index('T_air')])-float(r.event_Tair)),abs(float(iv[sv.index('VP')])-float(r.in_vp_pa)),abs(float(iv[sv.index('CO2')])-float(r.CO2_pre_ppm)),abs(float(iv[sv.index('T_can')])-float(r.canopy_temperature_c_closure)))
    finite=bool(np.all(np.isfinite(vent)) and all(np.all(np.isfinite(np.asarray(y[k],float))) for k in sv))
    return {'DV':dose,'DN':dose/H,'H_eff':H,'init_err':init_err,'action_err':float(np.max(np.abs(applied-float(u)))),'finite':finite}
rows=[]
for cohort,inp in INPUTS:
    E=pd.read_csv(inp)
    for _,r in E.iterrows():
        for arm,u in [('pre',r.u_pre),('post',r.u_post)]:
            z=run_arm(r,float(u)); rows.append({'cohort':cohort,'event_id':r.event_id,'event_date':r.event_date,'arm':arm,'action':float(u),'event_Windsp':float(r.event_Windsp),**z})
df=pd.DataFrame(rows); df.to_csv(OUT/'dose_arms.csv',index=False,float_format='%.12g')
hs=df.H_eff.to_numpy(float)
summary={'model':'M2','rows':len(df),'events_primary':97,'events_strict':36,'H_eff_m':float(np.median(hs)),'H_eff_range':float(hs.max()-hs.min()),'max_init_error':float(df.init_err.max()),'max_action_error':float(df.action_err.max()),'all_finite':bool(df.finite.all()),'native_integration_step_s':30}
summary['gate_pass']=bool(len(df)==2*(97+36) and summary['all_finite'] and summary['H_eff_range']<=1e-12 and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-10)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R2 M2 runtime gate failed')
