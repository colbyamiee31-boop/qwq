import json, hashlib, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CSG=ROOT/'CSGtom'
sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape, BVtomato_fun

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
OUT=ROOT/'EXP1_3_M2_FLUX_DECOMPOSITION.json'

LOW=0.1; HIGH=0.9
ROOT_BRACKET=(355.0,375.0)
SENSITIVITY=[355.0,415.0]
DT=30.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(f'Expected scalar-like value, got {np.asarray(x).shape}')
    return float(a[0])

def const(v):
    return lambda t,v=float(v): v

f=pd.read_csv(FORCING)
init=json.loads(INIT.read_text())
row=f.iloc[0]


def build_model(c0,v):
    p=example.parameters()
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T00:15'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(row['soil_boundary_temperature_c'])
    sv=list(p['StateVariable']); iv=np.asarray(p['InitialValues'],dtype=float).copy()
    for name,val in {
        'T_air':init['air_temperature_c'],
        'VP':init['air_vapor_pressure_pa'],
        'CO2':float(c0),
        'T_can':init['canopy_temperature_c'],
    }.items():
        iv[sv.index(name)]=val
    p['InitialValues']=iv
    tsim=np.asarray([0.0,900.0])
    x0={name:p['InitialValues'][i] for i,name in enumerate(sv)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    model.p['StartTime']='2017-09-01T08:00'; model.p['EndTime']='2017-09-01T08:15'
    model.D=csg_shape.csg_shape(model.p)
    model.d={
        'f_Rad':const(row['global_radiation_w_m2']),
        'f_Tem':const(row['outdoor_temperature_c']),
        'f_RH':const(row['outdoor_rh_pct']/100.0),
        'f_CO2':const(row['outdoor_co2_ppm']),
        'f_Wind':const(row['wind_speed_m_s']),
        'f_Tsky':const(row['sky_temperature_c']),
    }
    model.U={
        'u_blanket':const(0.0),'u_vent':const(v),'u_venttop':const(1.0),
        'u_ventside':const(0.0),'u_venttopbot':const(0.0),
    }
    return model,sv

TERM_NAMES=['ventilation_exchange','photosynthesis_uptake','organic_respiration','soil_respiration','external_co2_source','residual']


def run_arm(c0,v,collect=True):
    model,sv=build_model(c0,v)
    records=[]; action_values=[]; ctx={}
    orig_diff=model.diff
    orig_ctl=csg_fun.ctl_csg1
    orig_photo=BVtomato_fun.BramVanthoorPhotoSynthesis
    orig_smooth2=BVtomato_fun.BoSmoth2

    def ctl_wrap(p_,D,d,Tair,t,U):
        res=orig_ctl(p_,D,d,Tair,t,U)
        ctx['Vent']=scalar(res[4])
        action_values.append(scalar(res[1]))
        return res
    def photo_wrap(*args,**kwargs):
        res=orig_photo(*args,**kwargs)
        ctx['MCairbuf']=scalar(res)
        return res
    def smooth2_wrap(*args,**kwargs):
        res=orig_smooth2(*args,**kwargs)
        ctx['MCorgair_m']=scalar(res[3])
        return res
    def diff_wrap(t,y):
        ctx.clear()
        dy=np.asarray(orig_diff(t,y),dtype=float).reshape(-1)
        Vent=float(ctx['Vent'])
        MCairbuf=float(ctx['MCairbuf'])
        MCorgair=float(ctx['MCorgair_m'])
        CO2=float(model.CO2)
        co2out=float(model.d['f_CO2'](t))
        rad=float(model.d['f_Rad'](t))
        fac=float(model.p['MCO2']/model.p['MCH2O']*model.D.area_floor/model.D.Vair/model.p['eta_ppm_mgm3'])
        vent=float(Vent*model.D.area_floor/model.D.Vair*(co2out-CO2))
        photo=float(-MCairbuf*fac)
        resp=float(MCorgair*fac)
        sw=scalar(csg_fun.switch01(np.asarray([CO2-900.0]),1))
        soil_res=float((0.64*rad+57.0)*sw)
        soil=float(soil_res*1000000/10000/86400*model.D.area_floor/model.p['MC']*model.p['MCO2']/model.D.Vair/model.p['eta_ppm_mgm3'])
        ext=float(model.ext['ext_co2']/model.D.Vair/model.p['eta_ppm_mgm3'])
        residual=float(dy[2]-(vent+photo+resp+soil+ext))
        if collect:
            records.append({
                'time_s':float(t),'co2_ppm':CO2,'ventilation_exchange':vent,
                'photosynthesis_uptake':photo,'organic_respiration':resp,
                'soil_respiration':soil,'external_co2_source':ext,
                'residual':residual,'total_dco2_dt':float(dy[2])
            })
        return dy

    csg_fun.ctl_csg1=ctl_wrap
    BVtomato_fun.BramVanthoorPhotoSynthesis=photo_wrap
    BVtomato_fun.BoSmoth2=smooth2_wrap
    model.diff=diff_wrap
    try:
        y=model.run((0.0,900.0))
    finally:
        model.diff=orig_diff
        csg_fun.ctl_csg1=orig_ctl
        BVtomato_fun.BramVanthoorPhotoSynthesis=orig_photo
        BVtomato_fun.BoSmoth2=orig_smooth2

    final=float(np.asarray(y['CO2'])[-1])
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in sv)
    action_err=float(np.max(np.abs(np.asarray(action_values,dtype=float)-v)))
    out={'initial_co2_ppm':float(c0),'ventilation_command_fraction':float(v),'final_co2_ppm':final,
         'finite':bool(finite),'max_requested_applied_error':action_err}
    if collect:
        if len(records)!=30:
            raise AssertionError(f'Expected 30 Euler flux evaluations, got {len(records)}')
        ints={name:float(sum(r[name] for r in records)*DT) for name in TERM_NAMES}
        total_int=float(sum(r['total_dco2_dt'] for r in records)*DT)
        named_sum=float(sum(ints[name] for name in TERM_NAMES))
        actual=float(final-c0)
        out.update({
            'integrated_flux_contributions_ppm':ints,
            'integrated_total_ode_change_ppm':total_int,
            'sum_named_flux_contributions_ppm':named_sum,
            'actual_co2_change_ppm':actual,
            'named_vs_total_ode_closure_ppm':float(named_sum-total_int),
            'total_ode_vs_actual_closure_ppm':float(total_int-actual),
            'max_abs_instantaneous_residual_ppm_s':float(max(abs(r['residual']) for r in records)),
            'flux_trace':records,
        })
    return out


def response(c0):
    lo=run_arm(c0,LOW,collect=False); hi=run_arm(c0,HIGH,collect=False)
    return hi['final_co2_ppm']-lo['final_co2_ppm']

def bisect(lo,hi):
    flo=response(lo); fhi=response(hi)
    if flo==0: return lo,flo,0
    if fhi==0: return hi,fhi,0
    if flo*fhi>0:
        raise AssertionError(f'Inherited M2 root bracket lost sign change: {flo}, {fhi}')
    for it in range(80):
        mid=0.5*(lo+hi); fm=response(mid)
        if abs(fm)<1e-9 or (hi-lo)<1e-7:
            return mid,fm,it+1
        if flo*fm<=0:
            hi=mid; fhi=fm
        else:
            lo=mid; flo=fm
    return 0.5*(lo+hi),response(0.5*(lo+hi)),80

def paired_decomp(c0):
    lo=run_arm(c0,LOW,collect=True); hi=run_arm(c0,HIGH,collect=True)
    parts={}
    for name in TERM_NAMES:
        parts[name]=float(hi['integrated_flux_contributions_ppm'][name]-lo['integrated_flux_contributions_ppm'][name])
    actual=float(hi['final_co2_ppm']-lo['final_co2_ppm'])
    reconstructed=float(sum(parts.values()))
    return {
        'initial_co2_ppm':float(c0),'LOW':lo,'HIGH':hi,
        'high_minus_low':{
            'final_co2_ppm':actual,
            'flux_contributions_ppm':parts,
            'reconstructed_final_co2_ppm':reconstructed,
            'closure_error_ppm':float(reconstructed-actual),
        }
    }

root,root_response,iters=bisect(*ROOT_BRACKET)
root_dec=paired_decomp(root)
sensitivity=[paired_decomp(x) for x in SENSITIVITY]

all_arms=[root_dec['LOW'],root_dec['HIGH']]+[a[k] for a in sensitivity for k in ['LOW','HIGH']]
max_action_err=max(a['max_requested_applied_error'] for a in all_arms)
all_finite=all(a['finite'] for a in all_arms)
max_arm_closure=max(abs(a['total_ode_vs_actual_closure_ppm']) for a in all_arms)
max_pair_closure=max(abs(root_dec['high_minus_low']['closure_error_ppm']),*[abs(a['high_minus_low']['closure_error_ppm']) for a in sensitivity])
max_instant_res=max(a['max_abs_instantaneous_residual_ppm_s'] for a in all_arms)

result={
    'experiment':'PhysBench-GH EXP1.3',
    'model_id':'M2','model':'CSGtom','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
    'benchmark_forcing_sha256':sha(FORCING),
    'matched_initial_state_sha256':sha(INIT),
    'root_refinement':{
        'inherited_bracket_ppm':list(ROOT_BRACKET),
        'refined_boundary_ppm':float(root),
        'residual_high_minus_low_ppm':float(root_response),
        'iterations':int(iters),
    },
    'native_co2_state':'CO2 concentration [ppm]',
    'flux_term_names':TERM_NAMES,
    'boundary_decomposition':root_dec,
    'sensitivity_decompositions':sensitivity,
    'runtime_audit':{
        'max_requested_applied_error':float(max_action_err),
        'all_states_finite':bool(all_finite),
        'max_arm_total_ode_vs_actual_closure_ppm':float(max_arm_closure),
        'max_pair_decomposition_closure_error_ppm':float(max_pair_closure),
        'max_abs_instantaneous_residual_ppm_s':float(max_instant_res),
        'native_integration_step_s':30,
    },
    'interpretation_limit':'Read-only flux accounting of the frozen M2 ODE. No flux was disabled, repaired, replayed, or retuned.'
}
result['run_pass']=bool(
    max_action_err==0.0 and all_finite
    and max_arm_closure<=1e-8 and max_pair_closure<=1e-8
    and ROOT_BRACKET[0]<=root<=ROOT_BRACKET[1]
)
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']:
    raise SystemExit('M2 EXP1.3 runtime/closure gate failed')
