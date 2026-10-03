import json,sys,datetime as _dt
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG)); sys.path.insert(0,str(ROOT/'physbench/r3'))
from common import *
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun,csg_shape
CT=ROOT/'physbench/r3/generated/PRIMARY_97_R2_INPUT.csv'
D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/R3/M2'; OUT.mkdir(parents=True,exist_ok=True)
CTE=pd.read_csv(CT); D2E=pd.read_csv(D2)
def const(v): return lambda t,vv=float(v):vv

def base_parameters():
    p=example.parameters(); p['outdoorDataFileURL']=str(CSG/'data/example_data.xls'); p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'; p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    return p

def run_arm(r,u,stage):
    p=base_parameters(); p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    sv=list(p['StateVariable']); iv=np.asarray(p['InitialValues'],float).copy()
    for name,val in {'T_air':r.event_Tair,'VP':r.in_vp_pa,'CO2':r.CO2_pre_ppm,'T_can':getattr(r,'canopy_temperature_c_closure',r.event_Tair)}.items():
        iv[sv.index(name)]=float(val)
    p['InitialValues']=iv.copy()
    tsim=np.arange(0,1800+p['dtsim'],p['dtsim'],dtype=float); x0={name:iv[i] for i,name in enumerate(sv)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    st=_dt.datetime(2017,int(r.month),int(r.day),hh,mm); en=st+_dt.timedelta(minutes=30)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    # shared semantically comparable envelope only
    model.D.Vair=float(model.D.area_floor)*H_TARGET
    if stage=='HV':
        model.p['Atop_vent']=VENT_PROJECTED_RATIO*float(model.D.area_floor)/float(model.p['r_net'])
    elif stage!='H':
        raise ValueError(stage)
    model.d={'f_Rad':const(r.event_Iglob),'f_Tem':const(r.event_Tout),'f_RH':const(float(r.out_vp_pa)/float(sat_vp_pa(float(r.event_Tout)))),'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(r.event_Windsp),'f_Tsky':const(r.sky_temperature_c_proxy)}
    model.U={'u_blanket':const(0.0),'u_vent':const(float(u)),'u_venttop':const(1.0),'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    vents=[]; acts=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); acts.append(float(np.asarray(res[1]).reshape(-1)[0])); vents.append(float(np.asarray(res[4]).reshape(-1)[0])); return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,1800.0))
    finally: csg_fun.ctl_csg1=orig
    if len(vents)!=60: raise AssertionError((stage,r.event_id,u,len(vents)))
    vent=np.asarray(vents,float); applied=np.asarray(acts,float); t=np.asarray(y['t'],float)
    init_err=max(abs(float(iv[sv.index('T_air')])-float(r.event_Tair)),abs(float(iv[sv.index('VP')])-float(r.in_vp_pa)),abs(float(iv[sv.index('CO2')])-float(r.CO2_pre_ppm)),abs(float(iv[sv.index('T_can')])-float(getattr(r,'canopy_temperature_c_closure',r.event_Tair))))
    finite=bool(np.all(np.isfinite(vent)) and all(np.all(np.isfinite(np.asarray(y[k],float))) for k in sv))
    out=[]
    for sec,n in [(900.0,30),(1800.0,60)]:
        hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-9))[0]
        if len(hits)!=1: raise AssertionError((stage,r.event_id,u,sec,hits))
        j=int(hits[0]); T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j]); dose=float(np.sum(vent[:n])*30.0)
        out.append({'horizon_min':int(sec/60),'T':T,'AH':float(ah_from_t_vp(T,VP)),'DV':dose,'DN':dose/H_TARGET})
    geom={'effective_height_m':float(model.D.Vair/model.D.area_floor),'roof_aperture_ratio_m2_m2':float(model.p['Atop_vent']*model.p['r_net']/model.D.area_floor),'area_floor_m2_per_m':float(model.D.area_floor),'Vair_m3_per_m':float(model.D.Vair),'Atop_vent_parameter':float(model.p['Atop_vent']),'r_net':float(model.p['r_net'])}
    return out,{'init_err':init_err,'action_err':float(np.max(np.abs(applied-float(u)))),'finite':finite,'geometry':geom}

ct_rows=[]; grid_rows=[]; audits=[]
for stage in STAGES:
    for _,r in CTE.iterrows():
        pre,ap=run_arm(r,float(r.u_pre),stage); post,aq=run_arm(r,float(r.u_post),stage)
        audits.append({'track':'CTIFL','stage':stage,'event_id':r.event_id,'init_err':max(ap['init_err'],aq['init_err']),'action_err':max(ap['action_err'],aq['action_err']),'finite':ap['finite'] and aq['finite'],**{f'g_{k}':v for k,v in ap['geometry'].items()}})
        for p0,p1 in zip(pre,post):
            h=p0['horizon_min']
            ct_rows.append({'model':'M2','stage':stage,'event_id':r.event_id,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':h,'delta_u':float(r.delta_u),'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'event_Windsp':float(r.event_Windsp),'model_T_delta':p1['T']-p0['T'],'model_AH_delta':p1['AH']-p0['AH'],'pre_T':p0['T'],'post_T':p1['T'],'pre_AH':p0['AH'],'post_AH':p1['AH'],'delta_DV':p1['DV']-p0['DV'],'delta_DN':p1['DN']-p0['DN'],'u_pre':float(r.u_pre),'u_post':float(r.u_post),'obs_T_delta':float(r[f'obs_T_matched_{h}']),'obs_AH_delta':float(r[f'obs_AH_matched_{h}'])})
    for _,r in D2E.iterrows():
        for u in ACTIONS:
            tr,aud=run_arm(r,float(u),stage)
            audits.append({'track':'D2','stage':stage,'event_id':int(r.event_id),'action':float(u),'init_err':aud['init_err'],'action_err':aud['action_err'],'finite':aud['finite'],**{f'g_{k}':v for k,v in aud['geometry'].items()}})
            for q in tr:
                grid_rows.append({'model':'M2','stage':stage,'event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':q['horizon_min'],'action':float(u),'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'T':q['T'],'AH':q['AH'],'DV':q['DV'],'DN':q['DN']})
    print('M2 stage',stage,'done',flush=True)

ct=pd.DataFrame(ct_rows); grid=pd.DataFrame(grid_rows); ad=pd.DataFrame(audits)
ct.to_csv(OUT/'ctifl_event_responses.csv',index=False,float_format='%.12g')
grid.to_csv(OUT/'d2_action_grid.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
summary={'model':'M2','ctifl_rows':len(ct),'d2_rows':len(grid),'all_finite':bool(ad.finite.all()),'max_init_error':float(ad.init_err.max()),'max_action_error':float(ad.action_err.max()),'stage_geometry':{}}
for stage in STAGES:
    a=ad[ad.stage==stage].iloc[0]; summary['stage_geometry'][stage]={k[2:]:float(a[k]) for k in ad.columns if k.startswith('g_')}
summary['gate_pass']=bool(len(ct)==97*2*2 and len(grid)==61*5*2*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-10 and all(abs(summary['stage_geometry'][s]['effective_height_m']-H_TARGET)<=1e-12 for s in STAGES) and abs(summary['stage_geometry']['HV']['roof_aperture_ratio_m2_m2']-VENT_PROJECTED_RATIO)<=1e-12)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R3 M2 gate failed')
