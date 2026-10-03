import json,sys
from pathlib import Path
import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np,pandas as pd
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physbench/r3'))
from common import *
CT=ROOT/'physbench/r3/generated/PRIMARY_97_R2_INPUT.csv'
D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/R3/M1'; OUT.mkdir(parents=True,exist_ok=True)
CTE=pd.read_csv(CT); D2E=pd.read_csv(D2)

def build_weather(r):
    d=np.zeros((3,10),dtype=float)
    d[:,0]=float(r.event_Iglob); d[:,1]=float(r.event_Tout); d[:,2]=float(r.out_vp_pa)
    d[:,3]=ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed)); d[:,4]=float(r.event_Windsp)
    d[:,5]=float(r.sky_temperature_c_proxy); d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,float(r.event_Iglob))*900.0)/1e6
    d[:,8]=1.0 if float(r.event_Iglob)>0 else 0.0; d[:,9]=d[:,8]
    return d

def full_action(v): return np.array([0,0,0,float(v),0,0],dtype=np.float32)

# native symbolic model for read-only physical-dose quadrature
_env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
_env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
_e=_env.unwrapped; NATIVE_X=_e.x.copy(); P0=np.asarray(_e.p,float).copy(); NX=_e.nx; NU=_e.nu; ND=_e.nd; _env.close()
xs=ca.SX.sym('x',NX); us=ca.SX.sym('u',NU); ds=ca.SX.sym('d',ND); ps=ca.SX.sym('p',len(P0))
aa=aux_states.update(xs,us,ds,ps); dx=ODE(xs,us,ds,ps); qext=aa[136]+aa[137]+aa[145]
FQ=ca.integrator('R3_M1_Q900','cvodes',{'x':xs,'u':us,'p':ca.vertcat(ds,ps),'ode':dx,'quad':qext},0.0,900.0,{'abstol':1e-4,'reltol':1e-4,'max_num_steps':70000})

def run_arm(r,u,stage):
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped; p=harmonise_m1_p(e.p,stage); e.p=p.copy()
    d=build_weather(r); e.weather_data=d.copy(); e.day_of_year=int(r.day_of_year); e.hour_of_day=float(r.hour_decimal)
    x=e.x.copy(); x[2]=float(r.event_Tair); x[15]=float(r.in_vp_pa); x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    x[4]=float(getattr(r,'canopy_temperature_c_closure',r.event_Tair))
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    init_err=max(abs(float(x[2])-float(r.event_Tair)),abs(float(x[15])-float(r.in_vp_pa)),abs(float(mg_m3_to_ppm(x[2],x[0]))-float(r.CO2_pre_ppm)),abs(float(x[4])-float(getattr(r,'canopy_temperature_c_closure',r.event_Tair))))
    a=full_action(u); states=[]; errs=[]; finite=True
    for _ in range(2):
        _,_,_,trunc,info=env.step(a)
        if trunc: raise RuntimeError((stage,r.event_id,u,'truncated'))
        states.append(e.x.copy()); errs.append(float(np.max(np.abs(np.asarray(info['controls'],float)-a)))); finite=finite and bool(np.all(np.isfinite(e.x)))
    env.close()
    x1,x2=states
    q1r=FQ(x0=x,u=a.astype(float),p=np.concatenate([d[0],p])); q1=float(np.asarray(q1r['qf'].full()).reshape(-1)[0]); x1q=np.asarray(q1r['xf'].full()).reshape(-1)
    q2r=FQ(x0=x1,u=a.astype(float),p=np.concatenate([d[1],p])); q2=float(np.asarray(q2r['qf'].full()).reshape(-1)[0]); x2q=np.asarray(q2r['xf'].full()).reshape(-1)
    qerr=max(float(np.max(np.abs(x1q-x1))),float(np.max(np.abs(x2q-x2))))
    out=[]
    for h,xf,dose in [(15,x1,q1),(30,x2,q1+q2)]:
        T=float(xf[2]); VP=float(xf[15])
        out.append({'horizon_min':h,'T':T,'AH':float(ah_from_t_vp(T,VP)),'DV':float(dose),'DN':float(dose/H_TARGET)})
    return out,{'init_err':init_err,'action_err':max(errs),'qstate_err':qerr,'finite':bool(finite and np.isfinite(q1) and np.isfinite(q2)),'geometry':m1_geometry_audit(p)}

ct_rows=[]; grid_rows=[]; audits=[]
for stage in STAGES:
    for _,r in CTE.iterrows():
        pre,ap=run_arm(r,float(r.u_pre),stage); post,aq=run_arm(r,float(r.u_post),stage)
        audits.append({'track':'CTIFL','stage':stage,'event_id':r.event_id,'init_err':max(ap['init_err'],aq['init_err']),'action_err':max(ap['action_err'],aq['action_err']),'qstate_err':max(ap['qstate_err'],aq['qstate_err']),'finite':ap['finite'] and aq['finite'],**{f'g_{k}':v for k,v in ap['geometry'].items()}})
        for p0,p1 in zip(pre,post):
            h=p0['horizon_min']
            ct_rows.append({'model':'M1','stage':stage,'event_id':r.event_id,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':h,'delta_u':float(r.delta_u),'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'event_Windsp':float(r.event_Windsp),'model_T_delta':p1['T']-p0['T'],'model_AH_delta':p1['AH']-p0['AH'],'pre_T':p0['T'],'post_T':p1['T'],'pre_AH':p0['AH'],'post_AH':p1['AH'],'delta_DV':p1['DV']-p0['DV'],'delta_DN':p1['DN']-p0['DN'],'u_pre':float(r.u_pre),'u_post':float(r.u_post),'obs_T_delta':float(r[f'obs_T_matched_{h}']),'obs_AH_delta':float(r[f'obs_AH_matched_{h}'])})
    for _,r in D2E.iterrows():
        for u in ACTIONS:
            tr,aud=run_arm(r,float(u),stage)
            audits.append({'track':'D2','stage':stage,'event_id':int(r.event_id),'action':float(u),'init_err':aud['init_err'],'action_err':aud['action_err'],'qstate_err':aud['qstate_err'],'finite':aud['finite'],**{f'g_{k}':v for k,v in aud['geometry'].items()}})
            for q in tr:
                grid_rows.append({'model':'M1','stage':stage,'event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':q['horizon_min'],'action':float(u),'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'T':q['T'],'AH':q['AH'],'DV':q['DV'],'DN':q['DN']})
    print('M1 stage',stage,'done',flush=True)

ct=pd.DataFrame(ct_rows); grid=pd.DataFrame(grid_rows); ad=pd.DataFrame(audits)
ct.to_csv(OUT/'ctifl_event_responses.csv',index=False,float_format='%.12g')
grid.to_csv(OUT/'d2_action_grid.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
summary={'model':'M1','ctifl_rows':len(ct),'d2_rows':len(grid),'all_finite':bool(ad.finite.all()),'max_init_error':float(ad.init_err.max()),'max_action_error':float(ad.action_err.max()),'max_qstate_error':float(ad.qstate_err.max()),'stage_geometry':{}}
for stage in STAGES:
    a=ad[ad.stage==stage].iloc[0]
    summary['stage_geometry'][stage]={k[2:]:float(a[k]) for k in ad.columns if k.startswith('g_')}
summary['gate_pass']=bool(len(ct)==97*2*2 and len(grid)==61*5*2*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']<=1e-7 and summary['max_qstate_error']<=1e-7 and all(abs(summary['stage_geometry'][s]['effective_height_m']-H_TARGET)<=1e-12 for s in STAGES) and abs(summary['stage_geometry']['HV']['roof_aperture_ratio_m2_m2']-VENT_PROJECTED_RATIO)<=1e-12)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R3 M1 gate failed')
