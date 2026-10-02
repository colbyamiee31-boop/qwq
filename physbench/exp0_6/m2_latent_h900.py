import json, sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, pandas as pd

from common import *
from generate_prehistory import generate_prehistory

CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
OUT=ROOT/'evidence/EXP0_6A_M2'; OUT.mkdir(parents=True,exist_ok=True)

LOW=float(CFG['low_vent_command']); NEUTRAL=float(CFG['neutral_prehistory_vent_command']); HIGH=float(CFG['high_vent_command'])
SCAN=np.arange(CFG['root_scan']['min_ppm'],CFG['root_scan']['max_ppm']+0.1,CFG['root_scan']['step_ppm'])
WIDTH_TOL=float(CFG['root_scan']['bracket_width_tol_ppm'])
RESP_TOL=float(CFG['root_scan']['response_abs_tol_ppm'])
MAX_IT=int(CFG['root_scan']['max_iterations'])
DEDUP=float(CFG['root_scan']['dedup_tol_ppm'])
DT=30.0

forcing=pd.read_csv(FORCING)
target=json.loads(INIT.read_text())
assert sha256(ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv')=='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'

def const(v):
    return lambda t,vv=float(v):vv

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])

def common_state(x,sv):
    x=np.asarray(x,dtype=float)
    T=float(x[sv.index('T_air')]); VP=float(x[sv.index('VP')])
    return {
      'air_temperature_c':T,'air_vapor_pressure_pa':VP,'air_rh_pct':float(rh_from_t_vp(T,VP)),
      'air_co2_ppm':float(x[sv.index('CO2')]),'canopy_temperature_c':float(x[sv.index('T_can')])
    }

def base_parameters():
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    # Constructor-compatible window. It is replaced immediately after construction.
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    return p

p0=base_parameters()
SV=list(p0['StateVariable'])
native_x=np.asarray(p0['InitialValues'],dtype=float).copy()

def pwc(values):
    vals=np.asarray(values,dtype=float)
    def fn(t):
        sec=float(np.asarray(t).reshape(-1)[0])
        idx=int(np.floor(sec/900.0+1e-12))
        idx=max(0,min(idx,len(vals)-1))
        return float(vals[idx])
    return fn

def prehistory_state(hid):
    hist=generate_prehistory(hid,CFG['primary_prehistory_h'],CFG['anchor_end_local_hour'])
    intervals=hist.iloc[:-1].reset_index(drop=True)
    duration=float(CFG['primary_prehistory_h']*3600)
    p=base_parameters()
    p['T_soilbound']=float(intervals.soil_boundary_temperature_c.iloc[0])
    p['InitialValues']=native_x.copy()
    tsim=np.arange(0,duration+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:p['InitialValues'][i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)

    actual_end=datetime(2017,9,1,8,0)
    actual_start=actual_end-timedelta(hours=float(CFG['primary_prehistory_h']))
    model.p['StartTime']=actual_start.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=actual_end.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':pwc(intervals.global_radiation_w_m2),
      'f_Tem':pwc(intervals.outdoor_temperature_c),
      'f_RH':pwc(intervals.outdoor_rh_pct.to_numpy(float)/100.0),
      'f_CO2':pwc(intervals.outdoor_co2_ppm),
      'f_Wind':pwc(intervals.wind_speed_m_s),
      'f_Tsky':pwc(intervals.sky_temperature_c)
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(NEUTRAL),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0)
    }
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        rec.append(scalar(res[1]))
        return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,duration))
    finally:
        csg_fun.ctl_csg1=orig
    x=np.asarray([float(np.asarray(y[name])[-1]) for name in SV],dtype=float)
    finite=bool(all(np.all(np.isfinite(np.asarray(y[name],dtype=float))) for name in SV))
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-NEUTRAL))) if rec else float('inf')
    return x,{
      'history_id':hid,'native_state_sha256':state_sha(x),'all_states_finite':finite,
      'max_requested_applied_error':err,'pre_projection_common':common_state(x,SV),
      'end_hour_of_day':8.0,'forcing_rows_used':int(len(intervals))
    }

def projected_x(latent,c0):
    z=np.asarray(latent,dtype=float).copy()
    z[SV.index('T_air')]=float(target['air_temperature_c'])
    z[SV.index('VP')]=float(target['air_vapor_pressure_pa'])
    z[SV.index('T_can')]=float(target['canopy_temperature_c'])
    z[SV.index('CO2')]=float(c0)
    return z

def run_arm(latent,c0,vent):
    p=base_parameters()
    p['T_soilbound']=float(forcing.iloc[0].soil_boundary_temperature_c)
    z=projected_x(latent,c0)
    p['InitialValues']=z.copy()
    tsim=np.asarray([0.0,900.0])
    x0={name:z[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    st=datetime(2017,9,1,8,0); en=st+timedelta(seconds=900)
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    r=forcing.iloc[0]
    model.d={
      'f_Rad':const(r.global_radiation_w_m2),'f_Tem':const(r.outdoor_temperature_c),
      'f_RH':const(r.outdoor_rh_pct/100.0),'f_CO2':const(r.outdoor_co2_ppm),
      'f_Wind':const(r.wind_speed_m_s),'f_Tsky':const(r.sky_temperature_c)
    }
    model.U={
      'u_blanket':const(0.0),'u_vent':const(vent),'u_venttop':const(1.0),
      'u_ventside':const(0.0),'u_venttopbot':const(0.0)
    }
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); rec.append(scalar(res[1])); return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,900.0))
    finally:
        csg_fun.ctl_csg1=orig
    xf=np.asarray([float(np.asarray(y[name])[-1]) for name in SV],dtype=float)
    if not np.all(np.isfinite(xf)): raise RuntimeError('non-finite M2 intervention state')
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-vent))) if rec else float('inf')
    if err!=0.0: raise RuntimeError(f'M2 action application error {err}')
    T=float(xf[SV.index('T_air')]); VP=float(xf[SV.index('VP')])
    return {'T':T,'VP':VP,'AH':float(ah_g_m3_from_vp_pa(T,VP)),'CO2':float(xf[SV.index('CO2')])}

def response(latent,c0):
    lo=run_arm(latent,c0,LOW); hi=run_arm(latent,c0,HIGH)
    return float(hi['CO2']-lo['CO2'])

def refine(latent,lo,hi,flo,fhi):
    ilo,ihi=float(lo),float(hi)
    if abs(flo)<=RESP_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if abs(fhi)<=RESP_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if flo*fhi>0: raise AssertionError('invalid bracket')
    for it in range(1,MAX_IT+1):
        mid=0.5*(lo+hi); fm=response(latent,mid)
        if abs(fm)<=RESP_TOL or (hi-lo)<=WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
        if flo*fm<=0: hi,fhi=mid,fm
        else: lo,flo=mid,fm
    mid=0.5*(lo+hi); fm=response(latent,mid)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':MAX_IT,'final_bracket_width_ppm':float(hi-lo),
            'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,
            'converged':bool((hi-lo)<=WIDTH_TOL or abs(fm)<=RESP_TOL)}

latent_states={'HNR':native_x.copy()}
meta=[{
  'history_id':'HNR','native_state_sha256':state_sha(native_x),'all_states_finite':True,
  'max_requested_applied_error':0.0,'pre_projection_common':common_state(native_x,SV),
  'end_hour_of_day':None,'forcing_rows_used':0
}]
for hid in CFG['anchor_primary_history_ids']:
    x,m=prehistory_state(hid); latent_states[hid]=x; meta.append(m)

np.savez_compressed(OUT/'latent_states.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in latent_states.items()})

scan_rows=[]; root_rows=[]; summary_rows=[]
for hid in ['HNR']+CFG['anchor_primary_history_ids']:
    latent=latent_states[hid]
    pre=common_state(latent,SV); p500=projected_x(latent,500.0); post=common_state(p500,SV)
    correction={k:float(post[k]-pre[k]) for k in ['air_temperature_c','air_vapor_pressure_pa','air_co2_ppm','canopy_temperature_c']}
    ys=[]
    for c0 in SCAN:
        y=response(latent,float(c0)); ys.append(y)
        scan_rows.append({'model_id':'M2','history_id':hid,'initial_co2_ppm':float(c0),'high_minus_low_final_co2_ppm':y})
    candidates=[]
    for i,y in enumerate(ys):
        if abs(y)<=RESP_TOL: candidates.append(refine(latent,SCAN[i],SCAN[i],y,y))
    for i in range(len(SCAN)-1):
        if ys[i]*ys[i+1]<0: candidates.append(refine(latent,SCAN[i],SCAN[i+1],ys[i],ys[i+1]))
    roots=dedup_roots(candidates,DEDUP)
    unique=roots[0]['root_ppm'] if len(roots)==1 else None
    for j,r in enumerate(roots):
        root_rows.append({'model_id':'M2','history_id':hid,'root_index':j,**r})
    lo500=run_arm(latent,500.0,LOW); hi500=run_arm(latent,500.0,HIGH)
    row={
      'model_id':'M2','history_id':hid,'history_derived':hid!='HNR',
      'native_state_sha256':state_sha(latent),'root_count':len(roots),'primary_boundary_ppm':unique,
      'delta_T_C_at_500ppm':float(hi500['T']-lo500['T']),
      'delta_AH_g_m3_at_500ppm':float(hi500['AH']-lo500['AH']),
      'delta_CO2_ppm_at_500ppm':float(hi500['CO2']-lo500['CO2']),
      'projection_T_C':correction['air_temperature_c'],'projection_VP_Pa':correction['air_vapor_pressure_pa'],
      'projection_CO2_ppm_at_500':correction['air_co2_ppm'],'projection_canopy_T_C':correction['canopy_temperature_c'],
      'all_roots_converged':bool(all(r['converged'] for r in roots)),
      'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))
    }
    summary_rows.append(row)
    print(hid,'root=',unique,'n=',len(roots),'dCO2@500=',row['delta_CO2_ppm_at_500ppm'],flush=True)

md={m['history_id']:m for m in meta}
for row in summary_rows:
    row['prehistory_finite']=bool(md[row['history_id']]['all_states_finite'])
    row['prehistory_max_action_error']=float(md[row['history_id']]['max_requested_applied_error'])

sdf=pd.DataFrame(summary_rows)
pd.DataFrame(scan_rows).to_csv(OUT/'boundary_scan.csv',index=False,float_format='%.12g')
pd.DataFrame(root_rows).to_csv(OUT/'refined_roots.csv',index=False,float_format='%.12g')
sdf.to_csv(OUT/'history_summary.csv',index=False,float_format='%.12g')
(OUT/'latent_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

hnr=sdf[sdf.history_id=='HNR'].iloc[0]
hnr_err=abs(float(hnr.primary_boundary_ppm)-float(CFG['hnr_reference']['M2_root_ppm'])) if int(hnr.root_count)==1 else float('inf')
hist=sdf[sdf.history_derived]
audit={
 'experiment':'PhysBench-GH EXP0.6A','model_id':'M2','model':'CSGtom',
 'frozen_commit':CFG['frozen_models']['M2'],'history_count':int(len(hist)),
 'hnr_root_ppm':None if int(hnr.root_count)!=1 else float(hnr.primary_boundary_ppm),
 'hnr_root_abs_error_ppm':float(hnr_err),
 'all_prehistory_finite':bool(hist.prehistory_finite.all()),
 'max_prehistory_action_error':float(hist.prehistory_max_action_error.max()),
 'all_root_scans_finite':bool(np.isfinite(pd.DataFrame(scan_rows).high_minus_low_final_co2_ppm.to_numpy(float)).all()),
 'all_roots_converged':bool(sdf.all_roots_converged.all()),
 'all_roots_inside_bracket':bool(sdf.all_roots_inside_bracket.all())
}
audit['gate_pass']=bool(
    audit['history_count']==17 and audit['all_prehistory_finite'] and audit['max_prehistory_action_error']==0.0
    and audit['all_root_scans_finite'] and audit['all_roots_converged'] and audit['all_roots_inside_bracket']
    and audit['hnr_root_abs_error_ppm']<=float(CFG['hnr_reference']['root_abs_tol_ppm'])
)
(OUT/'summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
if not audit['gate_pass']: raise SystemExit('M2 EXP0.6A runtime/reconstruction gate failed')
