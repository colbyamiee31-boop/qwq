import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import gl_gym
import casadi as ca

from gl_gym.models.GreenLight.ode import ODE
from gl_gym.models.GreenLight import aux_states

ROOT=Path(__file__).resolve().parents[2]
FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
OUT=ROOT/'EXP1_3_M1_FLUX_DECOMPOSITION.json'

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0
LOW=0.1; HIGH=0.9
ROOT_BRACKET=(435.0,455.0)
SENSITIVITY=[425.0,455.0]


def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def full_action(v):
    return np.array([0,0,0,v,0,0],dtype=float)

f=pd.read_csv(FORCING)
init=json.loads(INIT.read_text())
row=f.iloc[0]

# Benchmark disturbance row in the frozen M1 native representation.
d0=np.zeros(10,dtype=float)
d0[0]=row['global_radiation_w_m2']
d0[1]=row['outdoor_temperature_c']
d0[2]=row['outdoor_vapor_pressure_pa']
d0[3]=ppm_to_mg_m3(row['outdoor_temperature_c'],row['outdoor_co2_ppm'])
d0[4]=row['wind_speed_m_s']
d0[5]=row['sky_temperature_c']
d0[6]=row['soil_boundary_temperature_c']
d0[7]=row['global_radiation_w_m2']*900.0/1e6
d0[8]=1.0 if row['global_radiation_w_m2']>0 else 0.0
d0[9]=d0[8]

env=gym.make(
    'gl_gym/GreenLightTomato-v0',
    controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],
    normalize_actions=False,
    parameter_provider='fixed'
)
env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
e=env.unwrapped
base_x=e.x.copy()
base_x[2]=init['air_temperature_c']
base_x[15]=init['air_vapor_pressure_pa']
base_x[0]=ppm_to_mg_m3(init['air_temperature_c'],init['air_co2_ppm'])
base_x[4]=init['canopy_temperature_c']
p_native=np.asarray(e.p,dtype=float).copy()
combined_p=np.concatenate([d0,p_native])

# Augment the native integrator with read-only quadratures of the already-existing CO2 ODE terms.
x=ca.SX.sym('x',e.nx)
u=ca.SX.sym('u',e.nu)
d=ca.SX.sym('d',e.nd)
p=ca.SX.sym('p',len(p_native))
a=aux_states.update(x,u,d,p)
dx=ODE(x,u,d,p)

# Signed contributions to d(CO2 density in main air)/dt [mg m^-3 s^-1].
q_canopy=-a[216]/p[122]
q_main_top=-a[217]/p[122]
q_main_out=-a[219]/p[122]
q_injection= a[222]/p[122]
q_blower= a[223]/p[122]
q_pad= a[224]/p[122]
q_terms=ca.vertcat(q_canopy,q_main_top,q_main_out,q_injection,q_blower,q_pad,dx[0])
FQ=ca.integrator(
    'EXP1_3_M1_FQ','cvodes',
    {'x':x,'u':u,'p':ca.vertcat(d,p),'ode':dx,'quad':q_terms},
    0.0,900.0,
    {'abstol':1e-4,'reltol':1e-4,'max_num_steps':int(7e4)}
)
TERM_NAMES=['canopy_net','main_to_top','main_to_outside','co2_injection','blower_source','pad_source']


def x0_for_co2(c0):
    z=base_x.copy()
    z[0]=ppm_to_mg_m3(z[2],float(c0))
    return z

def native_arm(c0,v):
    z=x0_for_co2(c0); act=full_action(v)
    res=e.F(x0=z,u=act,p=combined_p)
    xf=np.asarray(res['xf'].full()).reshape(-1)
    return xf,float(mg_m3_to_ppm(xf[2],xf[0]))

def response(c0):
    _,lo=native_arm(c0,LOW); _,hi=native_arm(c0,HIGH)
    return hi-lo

def bisect(lo,hi):
    flo=response(lo); fhi=response(hi)
    if flo==0: return lo,flo,0
    if fhi==0: return hi,fhi,0
    if flo*fhi>0:
        raise AssertionError(f'Inherited M1 root bracket lost sign change: {flo}, {fhi}')
    for it in range(80):
        mid=0.5*(lo+hi); fm=response(mid)
        if abs(fm)<1e-9 or (hi-lo)<1e-7:
            return mid,fm,it+1
        if flo*fm<=0:
            hi=mid; fhi=fm
        else:
            lo=mid; flo=fm
    return 0.5*(lo+hi),response(0.5*(lo+hi)),80

def decomp_arm(c0,v):
    z=x0_for_co2(c0); act=full_action(v)
    r=FQ(x0=z,u=act,p=combined_p)
    xf=np.asarray(r['xf'].full()).reshape(-1)
    q=np.asarray(r['qf'].full()).reshape(-1)
    native_xf,_=native_arm(c0,v)
    state_err=float(np.max(np.abs(xf-native_xf)))
    term_integrals={TERM_NAMES[i]:float(q[i]) for i in range(len(TERM_NAMES))}
    qsum=float(sum(term_integrals.values()))
    qtotal=float(q[-1])
    density_delta=float(xf[0]-z[0])
    return {
        'ventilation_command_fraction':float(v),
        'initial_co2_ppm':float(c0),
        'initial_density_mg_m3':float(z[0]),
        'final_density_mg_m3':float(xf[0]),
        'final_air_temperature_c':float(xf[2]),
        'final_co2_ppm':float(mg_m3_to_ppm(xf[2],xf[0])),
        'integrated_density_contributions_mg_m3':term_integrals,
        'integrated_total_ode_density_change_mg_m3':qtotal,
        'sum_named_density_contributions_mg_m3':qsum,
        'actual_density_change_mg_m3':density_delta,
        'named_vs_total_ode_closure_mg_m3':float(qsum-qtotal),
        'total_ode_vs_actual_closure_mg_m3':float(qtotal-density_delta),
        'augmented_vs_native_final_state_max_abs_diff':state_err,
    }

def paired_decomp(c0):
    lo=decomp_arm(c0,LOW); hi=decomp_arm(c0,HIGH)
    CH=hi['final_density_mg_m3']; CL=lo['final_density_mg_m3']
    TH=hi['final_air_temperature_c']; TL=lo['final_air_temperature_c']
    kH=float(R*(TH+K)/(P*MCO2)); kL=float(R*(TL+K)/(P*MCO2))
    kbar=0.5*(kH+kL); Cbar=0.5*(CH+CL)
    density_parts={}
    for name in TERM_NAMES:
        dq=hi['integrated_density_contributions_mg_m3'][name]-lo['integrated_density_contributions_mg_m3'][name]
        density_parts[name]={
            'high_minus_low_density_mg_m3':float(dq),
            'exact_symmetric_ppm_contribution':float(kbar*dq),
        }
    density_contrast=float(CH-CL)
    density_ppm=float(kbar*density_contrast)
    temp_conversion_ppm=float(Cbar*(kH-kL))
    actual_ppm=float(hi['final_co2_ppm']-lo['final_co2_ppm'])
    reconstructed=float(sum(v['exact_symmetric_ppm_contribution'] for v in density_parts.values())+temp_conversion_ppm)
    return {
        'initial_co2_ppm':float(c0),
        'LOW':lo,'HIGH':hi,
        'high_minus_low':{
            'final_density_mg_m3':density_contrast,
            'final_co2_ppm':actual_ppm,
            'density_related_ppm_total':density_ppm,
            'temperature_conversion_ppm':temp_conversion_ppm,
            'flux_contributions':density_parts,
            'reconstructed_final_co2_ppm':reconstructed,
            'ppm_closure_error':float(reconstructed-actual_ppm),
            'density_flux_closure_error_mg_m3':float(sum(v['high_minus_low_density_mg_m3'] for v in density_parts.values())-density_contrast),
        }
    }

root,root_response,iters=bisect(*ROOT_BRACKET)
root_dec=paired_decomp(root)
sensitivity=[paired_decomp(x) for x in SENSITIVITY]

max_state_err=max(
    [root_dec['LOW']['augmented_vs_native_final_state_max_abs_diff'],root_dec['HIGH']['augmented_vs_native_final_state_max_abs_diff']]
    +[a[k]['augmented_vs_native_final_state_max_abs_diff'] for a in sensitivity for k in ['LOW','HIGH']]
)
max_density_closure=max(
    abs(root_dec['high_minus_low']['density_flux_closure_error_mg_m3']),
    *[abs(a['high_minus_low']['density_flux_closure_error_mg_m3']) for a in sensitivity]
)
max_ppm_closure=max(
    abs(root_dec['high_minus_low']['ppm_closure_error']),
    *[abs(a['high_minus_low']['ppm_closure_error']) for a in sensitivity]
)

result={
    'experiment':'PhysBench-GH EXP1.3',
    'model_id':'M1','model':'GreenLight-Gym2',
    'frozen_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
    'benchmark_forcing_sha256':sha(FORCING),
    'matched_initial_state_sha256':sha(INIT),
    'root_refinement':{
        'inherited_bracket_ppm':list(ROOT_BRACKET),
        'refined_boundary_ppm':float(root),
        'residual_high_minus_low_ppm':float(root_response),
        'iterations':int(iters),
    },
    'native_co2_state':'main-air CO2 mass density [mg m^-3]',
    'flux_term_names':TERM_NAMES,
    'boundary_decomposition':root_dec,
    'sensitivity_decompositions':sensitivity,
    'runtime_audit':{
        'max_augmented_vs_native_final_state_abs_diff':float(max_state_err),
        'max_density_flux_closure_error_mg_m3':float(max_density_closure),
        'max_final_ppm_decomposition_closure_error':float(max_ppm_closure),
        'all_finite':bool(np.isfinite(root) and np.isfinite(root_response)),
    },
    'interpretation_limit':'Read-only flux accounting of the frozen M1 ODE. No flux was disabled, repaired, replayed, or retuned.'
}
result['run_pass']=bool(
    result['runtime_audit']['all_finite']
    and max_state_err<=1e-7
    and max_density_closure<=1e-5
    and max_ppm_closure<=1e-5
    and ROOT_BRACKET[0]<=root<=ROOT_BRACKET[1]
)
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
env.close()
if not result['run_pass']:
    raise SystemExit('M1 EXP1.3 runtime/closure gate failed')
