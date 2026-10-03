import json, sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from common import *
from generate_prehistory import generate_prehistory

CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape, BVtomato_fun

FORCING=ROOT/'physbench/exp0_5/benchmark_forcing.csv'
INIT=ROOT/'physbench/exp0_5/matched_initial_state.json'
REG=HERE/'exp0_6a_72h_state_hashes.json'
OUT=ROOT/'evidence/EXP0_6B_M2'; OUT.mkdir(parents=True,exist_ok=True)

LOW=float(CFG['low_vent_command']); NEUTRAL=float(CFG['neutral_prehistory_vent_command']); HIGH=float(CFG['high_vent_command'])
HORIZONS=[300,900,1800]
HISTORY_IDS=list(CFG['decision_primary_history_ids'])
SCAN=np.arange(CFG['root_scan']['min_ppm'],CFG['root_scan']['max_ppm']+0.1,CFG['root_scan']['step_ppm'])
WIDTH_TOL=float(CFG['root_scan']['bracket_width_tol_ppm'])
RESP_TOL=float(CFG['root_scan']['response_abs_tol_ppm'])
MAX_IT=int(CFG['root_scan']['max_iterations'])
DEDUP=float(CFG['root_scan']['dedup_tol_ppm'])
ACTIVE_SHARE=0.10
DT=30.0

TERM_NAMES=['ventilation_exchange','photosynthesis_uptake','organic_respiration','soil_respiration','external_co2_source','residual']
VECTOR_NAMES=TERM_NAMES.copy()

forcing=pd.read_csv(FORCING)
target=json.loads(INIT.read_text())
reg=json.loads(REG.read_text())

def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1: raise ValueError(np.asarray(x).shape)
    return float(a[0])

def const(v):
    return lambda t,vv=float(v):vv

# Frozen EXP1.4/1.5 compatibility adapter: only guarantees two solar-geometry support points.
_original_tran_cover=csg_shape.csg_shape.tran_cover
def _tran_cover_with_minimum_support(self):
    start=datetime.strptime(self.p['StartTime'],'%Y-%m-%dT%H:%M')
    end=datetime.strptime(self.p['EndTime'],'%Y-%m-%dT%H:%M')
    original_end=self.p['EndTime']
    if (end-start).total_seconds()<=600:
        self.p['EndTime']=(start+timedelta(seconds=900)).strftime('%Y-%m-%dT%H:%M')
        try:
            return _original_tran_cover(self)
        finally:
            self.p['EndTime']=original_end
    return _original_tran_cover(self)
csg_shape.csg_shape.tran_cover=_tran_cover_with_minimum_support

def base_parameters():
    p=example.parameters()
    p['outdoorDataFileURL']=str(CSG/'data/example_data.xls')
    p['UFileURL']=str(CSG/'data/example_u.xls')
    p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'
    p['dtsim']=900; p['dt']=30
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    return p

p0=base_parameters(); SV=list(p0['StateVariable']); native_x=np.asarray(p0['InitialValues'],dtype=float).copy()

def common_state(x):
    x=np.asarray(x,dtype=float)
    T=float(x[SV.index('T_air')]); VP=float(x[SV.index('VP')])
    return {'air_temperature_c':T,'air_vapor_pressure_pa':VP,'air_rh_pct':float(rh_from_t_vp(T,VP)),
            'air_co2_ppm':float(x[SV.index('CO2')]),'canopy_temperature_c':float(x[SV.index('T_can')])}

def pwc(values):
    vals=np.asarray(values,dtype=float)
    def fn(t):
        sec=float(np.asarray(t).reshape(-1)[0]); idx=int(np.floor(sec/900.0+1e-12))
        return float(vals[max(0,min(idx,len(vals)-1))])
    return fn

def prehistory_state(hid,duration_h):
    hist=generate_prehistory(hid,duration_h,CFG['anchor_end_local_hour'])
    intervals=hist.iloc[:-1].reset_index(drop=True)
    duration=float(duration_h*3600)
    p=base_parameters(); p['T_soilbound']=float(intervals.soil_boundary_temperature_c.iloc[0]); p['InitialValues']=native_x.copy()
    tsim=np.arange(0,duration+p['dtsim'],p['dtsim'],dtype=float)
    x0={name:p['InitialValues'][i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    actual_end=datetime(2017,9,1,8,0); actual_start=actual_end-timedelta(hours=float(duration_h))
    model.p['StartTime']=actual_start.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=actual_end.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
      'f_Rad':pwc(intervals.global_radiation_w_m2),'f_Tem':pwc(intervals.outdoor_temperature_c),
      'f_RH':pwc(intervals.outdoor_rh_pct.to_numpy(float)/100.0),'f_CO2':pwc(intervals.outdoor_co2_ppm),
      'f_Wind':pwc(intervals.wind_speed_m_s),'f_Tsky':pwc(intervals.sky_temperature_c)
    }
    model.U={'u_blanket':const(0.0),'u_vent':const(NEUTRAL),'u_venttop':const(1.0),
             'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); rec.append(scalar(res[1])); return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,duration))
    finally:
        csg_fun.ctl_csg1=orig
    x=np.asarray([float(np.asarray(y[name])[-1]) for name in SV],dtype=float)
    finite=bool(all(np.all(np.isfinite(np.asarray(y[name],dtype=float))) for name in SV))
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-NEUTRAL))) if rec else float('inf')
    return x,{'history_id':hid,'duration_h':int(duration_h),'native_state_sha256':state_sha(x),
              'all_states_finite':finite,'max_requested_applied_error':err,
              'pre_projection_common':common_state(x),'end_hour_of_day':8.0,'forcing_rows_used':int(len(intervals))}

def projected_x(latent,c0):
    z=np.asarray(latent,dtype=float).copy()
    z[SV.index('T_air')]=float(target['air_temperature_c'])
    z[SV.index('VP')]=float(target['air_vapor_pressure_pa'])
    z[SV.index('T_can')]=float(target['canopy_temperature_c'])
    z[SV.index('CO2')]=float(c0)
    return z

def build_model(h,latent,c0,vent):
    p=base_parameters()
    p['dtsim']=int(h); p['dt']=30
    p['T_soilbound']=float(forcing.iloc[0].soil_boundary_temperature_c)
    z=projected_x(latent,c0); p['InitialValues']=z.copy()
    tsim=np.asarray([0.0,float(h)]); x0={name:z[i] for i,name in enumerate(SV)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    st=datetime(2017,9,1,8,0); en=st+timedelta(seconds=float(h))
    model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    r=forcing.iloc[0]
    model.d={'f_Rad':const(r.global_radiation_w_m2),'f_Tem':const(r.outdoor_temperature_c),
             'f_RH':const(r.outdoor_rh_pct/100.0),'f_CO2':const(r.outdoor_co2_ppm),
             'f_Wind':const(r.wind_speed_m_s),'f_Tsky':const(r.sky_temperature_c)}
    model.U={'u_blanket':const(0.0),'u_vent':const(vent),'u_venttop':const(1.0),
             'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    return model,z

arm_cache={}
def native_arm(h,latent,c0,vent,cache_key):
    key=(int(h),cache_key,round(float(c0),10),float(vent))
    if key in arm_cache: return arm_cache[key]
    model,z=build_model(h,latent,c0,vent)
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); rec.append(scalar(res[1])); return res
    csg_fun.ctl_csg1=logged
    try:
        y=model.run((0.0,float(h)))
    finally:
        csg_fun.ctl_csg1=orig
    xf=np.asarray([float(np.asarray(y[name])[-1]) for name in SV],dtype=float)
    if not np.all(np.isfinite(xf)): raise RuntimeError(f'non-finite M2 state {key}')
    err=float(np.max(np.abs(np.asarray(rec,dtype=float)-vent))) if rec else float('inf')
    if err!=0.0: raise RuntimeError(f'M2 action application error {err}')
    out={'xf':xf,'final_co2_ppm':float(xf[SV.index('CO2')]),'max_requested_applied_error':err}
    arm_cache[key]=out
    return out

def response(h,latent,c0,cache_key):
    lo=native_arm(h,latent,c0,LOW,cache_key); hi=native_arm(h,latent,c0,HIGH,cache_key)
    return float(hi['final_co2_ppm']-lo['final_co2_ppm'])

def refine(h,latent,cache_key,lo,hi,flo,fhi):
    ilo,ihi=float(lo),float(hi)
    if abs(flo)<=RESP_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if abs(fhi)<=RESP_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
    if flo*fhi>0: raise AssertionError('invalid bracket')
    for it in range(1,MAX_IT+1):
        mid=.5*(lo+hi); fm=response(h,latent,mid,cache_key)
        if abs(fm)<=RESP_TOL or (hi-lo)<=WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,'converged':True}
        if flo*fm<=0: hi,fhi=mid,fm
        else: lo,flo=mid,fm
    mid=.5*(lo+hi); fm=response(h,latent,mid,cache_key)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':MAX_IT,'final_bracket_width_ppm':float(hi-lo),
            'initial_bracket_low_ppm':ilo,'initial_bracket_high_ppm':ihi,
            'converged':bool((hi-lo)<=WIDTH_TOL or abs(fm)<=RESP_TOL)}

def scan_root(h,latent,cache_key):
    ys=[response(h,latent,float(c0),cache_key) for c0 in SCAN]
    candidates=[]
    for i,y in enumerate(ys):
        if abs(y)<=RESP_TOL: candidates.append(refine(h,latent,cache_key,SCAN[i],SCAN[i],y,y))
    for i in range(len(SCAN)-1):
        if ys[i]*ys[i+1]<0: candidates.append(refine(h,latent,cache_key,SCAN[i],SCAN[i+1],ys[i],ys[i+1]))
    return ys,dedup_roots(candidates,DEDUP)

def flux_arm(latent,c0,vent,cache_key):
    model,z=build_model(900,latent,c0,vent)
    records=[]; actions=[]; ctx={}
    orig_diff=model.diff; orig_ctl=csg_fun.ctl_csg1
    orig_photo=BVtomato_fun.BramVanthoorPhotoSynthesis; orig_smooth2=BVtomato_fun.BoSmoth2
    def ctl_wrap(p_,D,d,Tair,t,U):
        res=orig_ctl(p_,D,d,Tair,t,U); ctx['Vent']=scalar(res[4]); actions.append(scalar(res[1])); return res
    def photo_wrap(*args,**kwargs):
        res=orig_photo(*args,**kwargs); ctx['MCairbuf']=scalar(res); return res
    def smooth2_wrap(*args,**kwargs):
        res=orig_smooth2(*args,**kwargs); ctx['MCorgair_m']=scalar(res[3]); return res
    def diff_wrap(t,y):
        ctx.clear(); dy=np.asarray(orig_diff(t,y),dtype=float).reshape(-1)
        Vent=float(ctx['Vent']); MCairbuf=float(ctx['MCairbuf']); MCorgair=float(ctx['MCorgair_m'])
        CO2=float(model.CO2); co2out=float(model.d['f_CO2'](t)); rad=float(model.d['f_Rad'](t))
        fac=float(model.p['MCO2']/model.p['MCH2O']*model.D.area_floor/model.D.Vair/model.p['eta_ppm_mgm3'])
        vent_flux=float(Vent*model.D.area_floor/model.D.Vair*(co2out-CO2))
        photo=float(-MCairbuf*fac); resp=float(MCorgair*fac)
        sw=scalar(csg_fun.switch01(np.asarray([CO2-900.0]),1))
        soil_res=float((0.64*rad+57.0)*sw)
        soil=float(soil_res*1000000/10000/86400*model.D.area_floor/model.p['MC']*model.p['MCO2']/model.D.Vair/model.p['eta_ppm_mgm3'])
        ext=float(model.ext['ext_co2']/model.D.Vair/model.p['eta_ppm_mgm3'])
        residual=float(dy[2]-(vent_flux+photo+resp+soil+ext))
        records.append({'ventilation_exchange':vent_flux,'photosynthesis_uptake':photo,'organic_respiration':resp,
                        'soil_respiration':soil,'external_co2_source':ext,'residual':residual,'total_dco2_dt':float(dy[2])})
        return dy
    csg_fun.ctl_csg1=ctl_wrap; BVtomato_fun.BramVanthoorPhotoSynthesis=photo_wrap; BVtomato_fun.BoSmoth2=smooth2_wrap; model.diff=diff_wrap
    try:
        y=model.run((0.0,900.0))
    finally:
        model.diff=orig_diff; csg_fun.ctl_csg1=orig_ctl
        BVtomato_fun.BramVanthoorPhotoSynthesis=orig_photo; BVtomato_fun.BoSmoth2=orig_smooth2
    final=float(np.asarray(y['CO2'])[-1])
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in SV))
    action_err=float(np.max(np.abs(np.asarray(actions,dtype=float)-vent))) if actions else float('inf')
    if len(records)!=30: raise AssertionError(f'Expected 30 Euler flux evaluations, got {len(records)}')
    ints={name:float(sum(r[name] for r in records)*DT) for name in TERM_NAMES}
    total_int=float(sum(r['total_dco2_dt'] for r in records)*DT)
    named=float(sum(ints.values())); actual=float(final-c0)
    return {'final_co2_ppm':final,'finite':finite,'max_requested_applied_error':action_err,
            'integrated_flux_contributions_ppm':ints,'integrated_total_ode_change_ppm':total_int,
            'sum_named_flux_contributions_ppm':named,'actual_co2_change_ppm':actual,
            'named_vs_total_ode_closure_ppm':float(named-total_int),
            'total_ode_vs_actual_closure_ppm':float(total_int-actual),
            'max_abs_instantaneous_residual_ppm_s':float(max(abs(r['residual']) for r in records))}

def paired_mechanism(latent,c0,cache_key):
    lo=flux_arm(latent,c0,LOW,cache_key); hi=flux_arm(latent,c0,HIGH,cache_key)
    v={name:float(hi['integrated_flux_contributions_ppm'][name]-lo['integrated_flux_contributions_ppm'][name]) for name in TERM_NAMES}
    actual=float(hi['final_co2_ppm']-lo['final_co2_ppm']); reconstructed=float(sum(v.values()))
    return {'LOW':lo,'HIGH':hi,'vector':v,'actual_high_minus_low_ppm':actual,
            'reconstructed_ppm':reconstructed,'closure_error_ppm':float(reconstructed-actual)}

def vector_metrics(v,anchor_v,closure_error_ppm):
    arr=np.asarray([v[k] for k in VECTOR_NAMES],dtype=float); l1=float(np.sum(np.abs(arr)))
    norm=arr/l1 if l1>0 else np.zeros_like(arr); idx=int(np.argmax(np.abs(arr)))
    order=np.argsort(-np.abs(arr)); top=float(abs(arr[order[0]])); second=float(abs(arr[order[1]]))
    resolved=bool((top-second)>abs(float(closure_error_ppm)))
    aa=np.asarray([anchor_v[k] for k in VECTOR_NAMES],dtype=float); al1=float(np.sum(np.abs(aa))); an=aa/al1 if al1>0 else np.zeros_like(aa)
    den=float(np.linalg.norm(norm)*np.linalg.norm(an))
    shared=[i for i in range(len(VECTOR_NAMES)) if abs(norm[i])>=ACTIVE_SHARE and abs(an[i])>=ACTIVE_SHARE]
    flips=[VECTOR_NAMES[i] for i in shared if np.sign(norm[i])!=np.sign(an[i])]
    out={'dominant_term':VECTOR_NAMES[idx] if l1>0 else 'NONE','dominance_share':float(np.max(np.abs(norm))) if l1>0 else 0.0,
         'active_terms':'|'.join(VECTOR_NAMES[i] for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
         'active_sign_pattern':'|'.join(f"{VECTOR_NAMES[i]}:{'+' if x>0 else '-'}" for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
         'anchor_cosine_similarity':float(np.dot(norm,an)/den) if den>0 else float('nan'),
         'anchor_l1_distance':float(np.sum(np.abs(norm-an))),
         'dominant_switch_from_hnr':bool(VECTOR_NAMES[idx]!=VECTOR_NAMES[int(np.argmax(np.abs(aa)))]) if l1>0 and al1>0 else False,
         'active_term_sign_flips_vs_hnr':'|'.join(flips),'n_active_term_sign_flips_vs_hnr':int(len(flips)),
         'l1_total_abs_ppm':l1,'dominant_resolved_under_closure_bound':resolved}
    for i,k in enumerate(VECTOR_NAMES): out[f'norm_{k}']=float(norm[i])
    return out

states={('HNR',0):native_x.copy()}
cross_run_hash_mismatches=[]
meta=[{'history_id':'HNR','duration_h':0,'native_state_sha256':state_sha(native_x),'all_states_finite':True,
       'max_requested_applied_error':0.0,'pre_projection_common':common_state(native_x)}]
for dur in [72,168]:
    for hid in HISTORY_IDS:
        x,m=prehistory_state(hid,dur)
        if dur==72:
            exp=reg['M2'][hid]
            if state_sha(x)!=exp:
                cross_run_hash_mismatches.append({'history_id':hid,'expected_sha256':exp,'observed_sha256':state_sha(x)})
        states[(hid,dur)]=x; meta.append(m)
state_file=OUT/'latent_states_72_168.npz'
np.savez_compressed(state_file,**{f'{hid}_{dur}h':np.asarray(v,dtype=np.float64) for (hid,dur),v in states.items()})
_reloaded=np.load(state_file)
same_run_state_restore_exact=True
for (hid,dur),v in states.items():
    key=f'{hid}_{dur}h'
    same_run_state_restore_exact = same_run_state_restore_exact and bool(np.array_equal(np.asarray(v,dtype=np.float64),_reloaded[key]))
if not same_run_state_restore_exact:
    raise AssertionError('M2 same-run latent-state serialization/restoration mismatch')

root_rows=[]; scan_rows=[]
for h in HORIZONS:
    ys,roots=scan_root(h,native_x,f'HNR_{h}')
    for c0,y in zip(SCAN,ys):
        scan_rows.append({'model_id':'M2','duration_h':0,'history_id':'HNR','horizon_s':h,'initial_co2_ppm':float(c0),'response_ppm':float(y)})
    unique=roots[0]['root_ppm'] if len(roots)==1 else np.nan
    root_rows.append({'model_id':'M2','duration_h':0,'history_id':'HNR','horizon_s':h,'root_count':len(roots),
                      'primary_boundary_ppm':unique,'all_roots_converged':bool(all(r['converged'] for r in roots)),
                      'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))})
for dur,horizons in [(72,HORIZONS),(168,[900])]:
    for hid in HISTORY_IDS:
        latent=states[(hid,dur)]
        for h in horizons:
            ys,roots=scan_root(h,latent,f'{hid}_{dur}_{h}')
            for c0,y in zip(SCAN,ys):
                scan_rows.append({'model_id':'M2','duration_h':dur,'history_id':hid,'horizon_s':h,'initial_co2_ppm':float(c0),'response_ppm':float(y)})
            unique=roots[0]['root_ppm'] if len(roots)==1 else np.nan
            root_rows.append({'model_id':'M2','duration_h':dur,'history_id':hid,'horizon_s':h,'root_count':len(roots),
                              'primary_boundary_ppm':unique,'all_roots_converged':bool(all(r['converged'] for r in roots)),
                              'all_roots_inside_bracket':bool(all(r['initial_bracket_low_ppm']-1e-12<=r['root_ppm']<=r['initial_bracket_high_ppm']+1e-12 for r in roots))})
            print('M2',dur,hid,h,'root',unique,'n',len(roots),flush=True)

rdf=pd.DataFrame(root_rows); sdf=pd.DataFrame(scan_rows)
hnr900=rdf[(rdf.duration_h==0)&(rdf.history_id=='HNR')&(rdf.horizon_s==900)].iloc[0]
if int(hnr900.root_count)!=1: raise AssertionError('M2 HNR H900 root not unique')
anchor_pair=paired_mechanism(native_x,float(hnr900.primary_boundary_ppm),'HNR_900')
anchor_v=anchor_pair['vector']

mechanism_rows=[]; term_rows=[]
def add_mechanism(duration_h,hid,latent,root):
    pair=paired_mechanism(latent,float(root),f'{hid}_{duration_h}_900')
    m=vector_metrics(pair['vector'],anchor_v,pair['closure_error_ppm'])
    max_action=max(pair['LOW']['max_requested_applied_error'],pair['HIGH']['max_requested_applied_error'])
    max_arm_closure=max(abs(pair['LOW']['total_ode_vs_actual_closure_ppm']),abs(pair['HIGH']['total_ode_vs_actual_closure_ppm']))
    row={'model_id':'M2','duration_h':duration_h,'history_id':hid,'root_ppm':float(root),
         'actual_high_minus_low_ppm':pair['actual_high_minus_low_ppm'],
         'pair_decomposition_closure_error_ppm':pair['closure_error_ppm'],
         'max_arm_total_ode_vs_actual_closure_ppm':float(max_arm_closure),
         'max_requested_applied_error':float(max_action),**m}
    mechanism_rows.append(row)
    for k in VECTOR_NAMES:
        term_rows.append({'model_id':'M2','duration_h':duration_h,'history_id':hid,'root_ppm':float(root),'term':k,
                          'contribution_ppm':float(pair['vector'][k])})

add_mechanism(0,'HNR',native_x,float(hnr900.primary_boundary_ppm))
for dur in [72,168]:
    for hid in HISTORY_IDS:
        rr=rdf[(rdf.duration_h==dur)&(rdf.history_id==hid)&(rdf.horizon_s==900)].iloc[0]
        if int(rr.root_count)==1:
            add_mechanism(dur,hid,states[(hid,dur)],float(rr.primary_boundary_ppm))

mdf=pd.DataFrame(mechanism_rows); tdf=pd.DataFrame(term_rows)

hnr_err=abs(float(hnr900.primary_boundary_ppm)-float(CFG['hnr_reference']['M2_root_ppm']))
reg_errors=[]
for hid in HISTORY_IDS:
    rr=rdf[(rdf.duration_h==72)&(rdf.history_id==hid)&(rdf.horizon_s==900)].iloc[0]
    reg_errors.append(abs(float(rr.primary_boundary_ppm)-float(reg['M2_H900_root_ppm'][hid])) if int(rr.root_count)==1 else float('inf'))
max_reg_err=float(max(reg_errors))
all_mech_finite=bool(np.isfinite(mdf.select_dtypes(include=[np.number]).to_numpy()).all())
max_action=float(mdf.max_requested_applied_error.max())
max_arm_closure=float(mdf.max_arm_total_ode_vs_actual_closure_ppm.max())
max_pair_closure=float(np.max(np.abs(mdf.pair_decomposition_closure_error_ppm.to_numpy(float))))

rdf.to_csv(OUT/'root_results.csv',index=False,float_format='%.12g')
sdf.to_csv(OUT/'root_scan.csv',index=False,float_format='%.12g')
mdf.to_csv(OUT/'mechanism_results.csv',index=False,float_format='%.12g')
tdf.to_csv(OUT/'mechanism_term_contributions.csv',index=False,float_format='%.12g')
(OUT/'latent_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

summary={'experiment':'PhysBench-GH EXP0.6B','model_id':'M2','model':'CSGtom','frozen_commit':CFG['frozen_models']['M2'],
 'history_ids':HISTORY_IDS,'hnr_h900_root_ppm':float(hnr900.primary_boundary_ppm),'hnr_h900_abs_error_ppm':float(hnr_err),
 'max_72h_h900_root_regression_error_ppm':max_reg_err,
 'all_root_scan_values_finite':bool(np.isfinite(sdf.response_ppm.to_numpy(float)).all()),
 'all_detected_roots_converged':bool(rdf.all_roots_converged.all()),
 'all_detected_roots_inside_bracket':bool(rdf.all_roots_inside_bracket.all()),
 'all_mechanism_outputs_finite':all_mech_finite,'max_mechanism_requested_applied_error':max_action,
 'max_mechanism_arm_total_ode_vs_actual_closure_ppm':max_arm_closure,
 'max_mechanism_pair_decomposition_closure_error_ppm':max_pair_closure,
 'mechanism_rows':int(len(mdf)),
 'cross_run_72h_state_hash_mismatch_count':int(len(cross_run_hash_mismatches)),
 'cross_run_72h_state_hash_mismatches':cross_run_hash_mismatches,
 'same_run_state_restore_exact':bool(same_run_state_restore_exact),
 'prehistory_all_finite':bool(all(x['all_states_finite'] for x in meta)),
 'prehistory_max_action_error':float(max(x['max_requested_applied_error'] for x in meta))}
summary['gate_pass']=bool(
    summary['hnr_h900_abs_error_ppm']<=float(CFG['hnr_reference']['root_abs_tol_ppm'])
    and summary['max_72h_h900_root_regression_error_ppm']<=float(reg['root_regression_tolerance_ppm'])
    and summary['all_root_scan_values_finite'] and summary['all_detected_roots_converged']
    and summary['all_detected_roots_inside_bracket'] and summary['all_mechanism_outputs_finite']
    and summary['max_mechanism_requested_applied_error']==0.0
    and summary['max_mechanism_arm_total_ode_vs_actual_closure_ppm']<=1e-8
    and summary['max_mechanism_pair_decomposition_closure_error_ppm']<=1e-8
    and summary['same_run_state_restore_exact']
    and summary['prehistory_all_finite'] and summary['prehistory_max_action_error']==0.0
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M2 EXP0.6B runtime/reconstruction gate failed')
