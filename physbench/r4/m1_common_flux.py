import json,sys
from pathlib import Path
import casadi as ca
import gymnasium as gym
import gl_gym
import numpy as np,pandas as pd
from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physbench/r4'))
from common import *

P97=ROOT/'physbench/r4/generated/PRIMARY_97_R2_INPUT.csv'
S36=ROOT/'physbench/r4/generated/STRICT_36_R2_INPUT.csv'
D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
OUT=ROOT/'evidence/R4/M1'; OUT.mkdir(parents=True,exist_ok=True)

def build_weather(r):
    d=np.zeros(10,dtype=float)
    d[0]=float(r.event_Iglob)
    d[1]=float(r.event_Tout)
    d[2]=float(r.out_vp_pa)
    d[3]=float(ppm_to_mg_m3(float(r.event_Tout),float(r.outdoor_co2_ppm_fixed)))
    d[4]=float(r.event_Windsp)
    d[5]=float(r.sky_temperature_c_proxy)
    d[6]=float(r.soil_boundary_temperature_c_fixed)
    d[7]=float(r.event_Iglob)*900.0/1e6
    d[8]=1.0 if float(r.event_Iglob)>0 else 0.0
    d[9]=d[8]
    return d

def full_action(v):
    return np.array([0,0,0,float(v),0,0],dtype=float)

env=gym.make('gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,parameter_provider='fixed')
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
X_NATIVE=e.x.copy()
P_NATIVE=np.asarray(e.p,dtype=float).copy()
NX=e.nx; NU=e.nu; ND=e.nd
env.close()

P_HV=harmonise_m1_hv(P_NATIVE)
assert abs(P_HV[49]-H_TARGET)<=1e-12
assert abs(P_HV[55]/P_HV[46]-VENT_PROJECTED_RATIO)<=1e-12

# Common-flux GreenLight ODE.
# Native external ventilation terms are algebraically replaced by prescribed
# q_top and q_main while all internal/topology/process terms remain native.
x=ca.SX.sym('x',NX)
u=ca.SX.sym('u',NU)
d=ca.SX.sym('d',ND)
p=ca.SX.sym('p',len(P_NATIVE))
qtop=ca.SX.sym('qtop')
qmain=ca.SX.sym('qmain')
a=aux_states.update(x,u,d,p)
dx=ODE(x,u,d,p)

def sensible(hec,t1,t2):
    return ca.fabs(hec)*(t1-t2)
def air_mv(f,vp1,vp2,t1,t2):
    return 0.002165*ca.fabs(f)*(vp1/(t1+273.15)-vp2/(t2+273.15))
def air_mc(f,c1,c2):
    return ca.fabs(f)*(c1-c2)

native_top=a[136]
native_main=a[137]+a[145]

# CO2 external exchange replacement
dx[0] += (air_mc(native_main,x[0],d[3])-air_mc(qmain,x[0],d[3]))/p[122]
dx[1] += (air_mc(native_top,x[1],d[3])-air_mc(qtop,x[1],d[3]))/p[123]
# sensible-heat external exchange replacement
cp_rho=p[111]*p[23]
dx[2] += (sensible(cp_rho*native_main,x[2],d[1])-sensible(cp_rho*qmain,x[2],d[1]))/p[112]
dx[3] += (sensible(cp_rho*native_top,x[3],d[1])-sensible(cp_rho*qtop,x[3],d[1]))/p[120]
# vapour external exchange replacement
dx[15] += (air_mv(native_main,x[15],d[2],x[2],d[1])-air_mv(qmain,x[15],d[2],x[2],d[1]))/a[35]
dx[16] += (air_mv(native_top,x[16],d[2],x[3],d[1])-air_mv(qtop,x[16],d[2],x[3],d[1]))/a[36]

FCF=ca.integrator(
    'R4_M1_CF900','cvodes',
    {'x':x,'u':u,'p':ca.vertcat(d,p,qtop,qmain),'ode':dx},
    0.0,900.0,
    {'abstol':1e-5,'reltol':1e-5,'max_num_steps':90000}
)
FAUX=ca.Function('R4_M1_AUX',[x,u,d,p],[native_top,native_main])

def initial_state(r):
    xx=X_NATIVE.copy()
    xx[2]=float(r.event_Tair)
    xx[15]=float(r.in_vp_pa)
    xx[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm)))
    xx[4]=float(getattr(r,'canopy_temperature_c_closure',r.event_Tair))
    return xx

def flux_route(r,action,q_common,stage):
    if stage=='CF0':
        return float(q_common),0.0,0.0
    if stage=='CFN':
        leak=m1_native_leakage(P_HV,float(r.event_Windsp))
        return float(q_common)+float(P_HV[204])*leak,(1.0-float(P_HV[204]))*leak,leak
    raise ValueError(stage)

def run_arm(r,action,q_common,stage):
    xx=initial_state(r)
    dd=build_weather(r)
    uu=full_action(action)
    qt,qm,leak=flux_route(r,action,q_common,stage)
    if not (np.isfinite(qt) and np.isfinite(qm) and qt>=0 and qm>=0):
        raise AssertionError(('bad_flux',stage,r.event_id,qt,qm))
    rr=FCF(x0=xx,u=uu,p=np.concatenate([dd,P_HV,[qt,qm]]))
    xf=np.asarray(rr['xf'].full()).reshape(-1)
    if not np.all(np.isfinite(xf)):
        raise RuntimeError(('nonfinite',stage,r.event_id,action))
    T=float(xf[2]); VP=float(xf[15])
    init_err=max(
        abs(float(xx[2])-float(r.event_Tair)),
        abs(float(xx[15])-float(r.in_vp_pa)),
        abs(float(mg_m3_to_ppm(xx[2],xx[0]))-float(r.CO2_pre_ppm)),
        abs(float(xx[4])-float(getattr(r,'canopy_temperature_c_closure',r.event_Tair)))
    )
    dv=(qt+qm)*900.0
    return {
        'T':T,'AH':float(ah_from_t_vp(T,VP)),
        'q_common':float(q_common),'q_top':qt,'q_main':qm,'native_leakage':leak,
        'DV_total':dv,'DN_total':dv/H_TARGET,
        'init_err':init_err,'finite':True
    }

def event_fluxes(r):
    return float(r.emp_flux_pre_23_23),float(r.emp_flux_post_23_23)

ct_rows=[]
audit=[]
for cohort,path in [('primary',P97),('strict',S36)]:
    E=pd.read_csv(path)
    E=E[E.anchor_primary_3_6.astype(bool)].copy()
    expected=40 if cohort=='primary' else 14
    assert len(E)==expected
    for stage in ['CF0','CFN']:
        for _,r in E.iterrows():
            qpre,qpost=event_fluxes(r)
            pre=run_arm(r,float(r.u_pre),qpre,stage)
            post=run_arm(r,float(r.u_post),qpost,stage)
            ct_rows.append({
                'model':'M1','cohort':cohort,'stage':stage,'event_id':r.event_id,'event_date':r.event_date,
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
                audit.append({'track':'CTIFL','cohort':cohort,'stage':stage,'event_id':r.event_id,'arm':arm,
                              'q_common':z['q_common'],'q_top':z['q_top'],'q_main':z['q_main'],
                              'native_leakage':z['native_leakage'],'DV_total':z['DV_total'],
                              'flux_identity_error':abs(z['DV_total']-(z['q_top']+z['q_main'])*900.0),
                              'init_err':z['init_err'],'finite':z['finite']})
        print('M1',cohort,stage,'done',flush=True)

# D2 wind-dominated mechanistic common-flux replay, CF0 only.
D=pd.read_csv(D2)
D=D[(D.event_Windsp>=3.0)&(D.event_Windsp<=6.0)].copy()
assert len(D)==28
grid=[]
for _,r in D.iterrows():
    for action in ACTIONS:
        qc=float(q_common_symmetric(float(action),float(r.event_Windsp)))
        z=run_arm(r,float(action),qc,'CF0')
        grid.append({
            'model':'M1','stage':'CF0','event_id':int(r.event_id),'team':r.team,'event_date':r.event_date,
            'daynight':r.daynight,'horizon_min':15,'action':float(action),
            'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),
            'event_Windsp':float(r.event_Windsp),'q_common':qc,'T':z['T'],'AH':z['AH'],
            'DV':z['DV_total'],'DN':z['DN_total']
        })
print('M1 D2 CF0 done',flush=True)

ct=pd.DataFrame(ct_rows)
gd=pd.DataFrame(grid)
ad=pd.DataFrame(audit)
ct.to_csv(OUT/'ctifl_common_flux_responses.csv',index=False,float_format='%.12g')
gd.to_csv(OUT/'d2_common_flux_grid.csv',index=False,float_format='%.12g')
ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')

summary={
    'model':'M1','primary_events':40,'strict_events':14,'d2_events':28,
    'ctifl_rows':len(ct),'d2_rows':len(gd),
    'H_target':H_TARGET,'vent_ratio_target':VENT_PROJECTED_RATIO,
    'all_finite':bool(ad.finite.all()),
    'max_init_error':float(ad.init_err.max()),
    'max_flux_identity_error':float(ad.flux_identity_error.max()),
    'CF0_max_common_DV_identity_error':float(np.max(np.abs(
        ad.loc[ad.stage=='CF0','DV_total'].to_numpy(float)-
        ad.loc[ad.stage=='CF0','q_common'].to_numpy(float)*900.0
    )))
}
summary['gate_pass']=bool(
    len(ct)==(40+14)*2 and len(gd)==28*5 and summary['all_finite']
    and summary['max_init_error']<=1e-8
    and summary['max_flux_identity_error']<=1e-12
    and summary['CF0_max_common_DV_identity_error']<=1e-12
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('R4 M1 runtime gate failed')
