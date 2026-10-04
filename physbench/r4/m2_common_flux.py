import json,sys,datetime as _dt
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG))
sys.path.insert(0,str(ROOT/'physbench/r4'))
from common import *
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun,csg_shape

P97=ROOT/'physbench/r4/generated/PRIMARY_97_R2_INPUT.csv'
S36=ROOT/'physbench/r4/generated/STRICT_36_R2_INPUT.csv'
D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/R4/M2'; OUT.mkdir(parents=True,exist_ok=True)

def const(v):
    return lambda t,vv=float(v):vv

def base_parameters():
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'
    p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900
    p['dt']=10
    p['ctl_vent_type']='timebasedControl'
    p['ctl_blank_type']='timebasedControl'
    return p

def event_datetime(r):
    hh=int(float(r.hour_decimal))
    mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60:
        hh=(hh+1)%24
        mm=0
    return _dt.datetime(2017,int(r.month),int(r.day),hh,mm)

def run_arm(r,action,q_common,stage):
    p=base_parameters()
    p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    sv=list(p['StateVariable'])
    iv=np.asarray(p['InitialValues'],float).copy()
    for name,val in {
        'T_air':r.event_Tair,
        'VP':r.in_vp_pa,
        'CO2':r.CO2_pre_ppm,
        'T_can':getattr(r,'canopy_temperature_c_closure',r.event_Tair)
    }.items():
        iv[sv.index(name)]=float(val)
    p['InitialValues']=iv.copy()

    tsim=np.arange(0,900+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:iv[i] for i,name in enumerate(sv)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    st=event_datetime(r)
    en=st+_dt.timedelta(minutes=15)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)

    # R3 HV common physical envelope retained.
    model.D.Vair=float(model.D.area_floor)*H_TARGET
    model.p['Atop_vent']=VENT_PROJECTED_RATIO*float(model.D.area_floor)/float(model.p['r_net'])

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
        'u_vent':const(float(action)),
        'u_venttop':const(1.0),
        'u_ventside':const(0.0),
        'u_venttopbot':const(0.0)
    }

    if stage=='CF0':
        leak=0.0
    elif stage=='CFN':
        leak=m2_native_leakage(model.p,float(r.event_Windsp))
    else:
        raise ValueError(stage)
    qtotal=float(q_common)+float(leak)
    if not np.isfinite(qtotal) or qtotal<0:
        raise AssertionError(('bad_flux',stage,r.event_id,qtotal))

    vents=[]
    acts=[]
    orig=csg_fun.ctl_csg1
    def common_flux_ctl(p_,D,d,Tair,t,U):
        native=orig(p_,D,d,Tair,t,U)
        u_blanket=float(np.asarray(native[0]).reshape(-1)[0])
        u_vent=float(np.asarray(native[1]).reshape(-1)[0])
        u_topbot=float(np.asarray(native[2]).reshape(-1)[0])
        u_side=float(np.asarray(native[3]).reshape(-1)[0])
        Tout=float(d['f_Tem'](t))
        h_airvent=D.area_floor*qtotal*(Tair-Tout)*p_['capacity_air']*p_['density_air']*1000
        vents.append(qtotal)
        acts.append(u_vent)
        return u_blanket,u_vent,u_topbot,u_side,qtotal,h_airvent

    csg_fun.ctl_csg1=common_flux_ctl
    try:
        y=model.run((0.0,900.0))
    finally:
        csg_fun.ctl_csg1=orig

    if len(vents)!=90:
        raise AssertionError((stage,r.event_id,action,'vent evaluations',len(vents)))
    if np.max(np.abs(np.asarray(vents,float)-qtotal))>1e-15:
        raise AssertionError(('flux application mismatch',stage,r.event_id))
    if np.max(np.abs(np.asarray(acts,float)-float(action)))>1e-10:
        raise AssertionError(('action mismatch',stage,r.event_id))

    t=np.asarray(y['t'],float)
    hits=np.where(np.isclose(t,900.0,rtol=0,atol=1e-9))[0]
    if len(hits)!=1:
        raise AssertionError((stage,r.event_id,'900s missing'))
    j=int(hits[0])
    T=float(np.asarray(y['T_air'])[j])
    VP=float(np.asarray(y['VP'])[j])
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],float))) for k in sv))
    if not finite:
        raise RuntimeError(('nonfinite',stage,r.event_id,action))
    init_err=max(
        abs(float(iv[sv.index('T_air')])-float(r.event_Tair)),
        abs(float(iv[sv.index('VP')])-float(r.in_vp_pa)),
        abs(float(iv[sv.index('CO2')])-float(r.CO2_pre_ppm)),
        abs(float(iv[sv.index('T_can')])-float(getattr(r,'canopy_temperature_c_closure',r.event_Tair)))
    )
    dv=qtotal*900.0
    return {
        'T':T,'AH':float(ah_from_t_vp(T,VP)),
        'q_common':float(q_common),'q_total':qtotal,'native_leakage':float(leak),
        'DV_total':dv,'DN_total':dv/H_TARGET,
        'init_err':init_err,'action_err':float(np.max(np.abs(np.asarray(acts,float)-float(action)))),
        'finite':finite,
        'height':float(model.D.Vair/model.D.area_floor),
        'vent_ratio':float(model.p['Atop_vent']*model.p['r_net']/model.D.area_floor)
    }

ct_rows=[]
audit=[]
for cohort,path in [('primary',P97),('strict',S36)]:
    E=pd.read_csv(path)
    E=E[E.anchor_primary_3_6.astype(bool)].copy()
    expected=40 if cohort=='primary' else 14
    assert len(E)==expected
    for stage in ['CF0','CFN']:
        for _,r in E.iterrows():
            qpre=float(r.emp_flux_pre_23_23)
            qpost=float(r.emp_flux_post_23_23)
            pre=run_arm(r,float(r.u_pre),qpre,stage)
            post=run_arm(r,float(r.u_post),qpost,stage)
            ct_rows.append({
                'model':'M2','cohort':cohort,'stage':stage,'event_id':r.event_id,'event_date':r.event_date,
                'daynight':r.daynight,'horizon_min':15,'delta_u':float(r.delta_u),
                'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
                'event_Windsp':float(r.event_Windsp),
                'model_T_delta':post['T']-pre['T'],'model_AH_delta':post['AH']-pre['AH'],
                'pre_T':pre['T'],'post_T':post['T'],'pre_AH':pre['AH'],'post_AH':post['AH'],
                'q_common_pre':qpre,'q_common_post':qpost,
                'delta_DV':post['DV_total']-pre['DV_total'],'delta_DN':post['DN_total']-pre['DN_total'],
                'u_pre':float(r.u_pre),'u_post':float(r.u_post),
                'obs_T_delta':float(r.obs_T_matched_15),'obs_AH_delta':float(r.obs_AH_matched_15)
            })
            for arm,z in [('pre',pre),('post',post)]:
                audit.append({
                    'track':'CTIFL','cohort':cohort,'stage':stage,'event_id':r.event_id,'arm':arm,
                    'q_common':z['q_common'],'q_total':z['q_total'],'native_leakage':z['native_leakage'],
                    'DV_total':z['DV_total'],
                    'flux_identity_error':abs(z['DV_total']-z['q_total']*900.0),
                    'init_err':z['init_err'],'action_err':z['action_err'],'finite':z['finite'],
                    'height':z['height'],'vent_ratio':z['vent_ratio']
                })
        print('M2',cohort,stage,'done',flush=True)

D=pd.read_csv(D2)
D=D[(D.event_Windsp>=3.0)&(D.event_Windsp<=6.0)].copy()
assert len(D)==28
grid=[]
for _,r in D.iterrows():
    for action in ACTIONS:
        qc=float(q_common_symmetric(float(action),float(r.event_Windsp)))
        z=run_arm(r,float(action),qc,'CF0')
        grid.append({
            'model':'M2','stage':'CF0','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,
            'daynight':r.daynight,'horizon_min':15,'action':float(action),
            'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
            'event_Windsp':float(r.event_Windsp),'q_common':qc,'T':z['T'],'AH':z['AH'],
            'DV':z['DV_total'],'DN':z['DN_total']
        })
print('M2 D2 CF0 done',flush=True)

ct=pd.DataFrame(ct_rows)
gd=pd.DataFrame(grid)
ad=pd.DataFrame(audit)
ct.to_csv(OUT/'ctifl_common_flux_responses.csv',index=False,float_format='%.12g')
gd.to_csv(OUT/'d2_common_flux_grid.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')

summary={
    'model':'M2','primary_events':40,'strict_events':14,'d2_events':28,
    'ctifl_rows':len(ct),'d2_rows':len(gd),
    'H_target':H_TARGET,'vent_ratio_target':VENT_PROJECTED_RATIO,
    'all_finite':bool(ad.finite.all()),
    'max_init_error':float(ad.init_err.max()),
    'max_action_error':float(ad.action_err.max()),
    'max_flux_identity_error':float(ad.flux_identity_error.max()),
    'max_height_error':float(np.max(np.abs(ad.height.to_numpy(float)-H_TARGET))),
    'max_vent_ratio_error':float(np.max(np.abs(ad.vent_ratio.to_numpy(float)-VENT_PROJECTED_RATIO))),
    'CF0_max_common_DV_identity_error':float(np.max(np.abs(
        ad.loc[ad.stage=='CF0','DV_total'].to_numpy(float)-
        ad.loc[ad.stage=='CF0','q_common'].to_numpy(float)*900.0
    ))),
    'integration_step_s':10
}
summary['gate_pass']=bool(
    len(ct)==(40+14)*2 and len(gd)==28*5 and summary['all_finite']
    and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-10
    and summary['max_flux_identity_error']<=1e-12
    and summary['max_height_error']<=1e-12 and summary['max_vent_ratio_error']<=1e-12
    and summary['CF0_max_common_DV_identity_error']<=1e-12
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('R4 M2 runtime gate failed')
