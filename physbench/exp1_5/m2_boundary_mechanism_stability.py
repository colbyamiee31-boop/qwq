import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSG = ROOT / 'CSGtom'
sys.path.insert(0, str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape, BVtomato_fun

FORCING = ROOT / 'physbench/exp0_5/benchmark_forcing.csv'
INIT = ROOT / 'physbench/exp0_5/matched_initial_state.json'
OUT_JSON = ROOT / 'EXP1_5_M2_BOUNDARY_MECHANISM_STABILITY.json'
OUT_CELLS = ROOT / 'EXP1_5_M2_CELL_SUMMARY.csv'
OUT_ROOTS = ROOT / 'EXP1_5_M2_ROOT_MECHANISMS.csv'
OUT_EDGES = ROOT / 'EXP1_5_M2_EDGE_MECHANISMS.csv'
OUT_TERMS = ROOT / 'EXP1_5_M2_TERM_CONTRIBUTIONS.csv'
OUT_SCAN = ROOT / 'EXP1_5_M2_SURFACE_REGRESSION_SCAN.csv'
OUT_TRACE = ROOT / 'EXP1_5_M2_FLUX_TRACE.csv'

LOW = 0.1
HIGH = 0.9
DT = 30.0
HORIZONS = [300, 900, 1800]
FORCING_ROWS = [0, 6, 10]
SCAN_GRID = np.arange(250.0, 600.0 + 0.1, 25.0)
ROOT_WIDTH_TOL = 1e-4
RESPONSE_TOL = 1e-7
ROOT_DEDUP_TOL = 1e-3
ANCHOR_CONDITION = 'H900_S_NOMINAL_FROW0'
EXP1_4_ANCHOR = 358.2789194211364
ANCHOR_TOL = 0.02
EXPECTED_UNIQUE = 18
EXPECTED_NO_ROOT = 9
EXPECTED_MULTIPLE = 0
ACTIVE_SHARE = 0.10

TERM_NAMES = ['ventilation_exchange','photosynthesis_uptake','organic_respiration','soil_respiration','external_co2_source','residual']
VECTOR_NAMES = TERM_NAMES.copy()
GROUP_NAMES = ['direct_exchange','canopy_biology','soil_biology','external_source','residual']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sat_vp(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))


def scalar(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.size!=1:
        raise ValueError(f'Expected scalar-like value, got {np.asarray(x).shape}')
    return float(a[0])


def const(v):
    return lambda t,v=float(v):v

# EXP1.4 compatibility adapter: minimum support only for solar-geometry interpolation.
_original_tran_cover = csg_shape.csg_shape.tran_cover

def _tran_cover_with_minimum_support(self):
    start=datetime.strptime(self.p['StartTime'],'%Y-%m-%dT%H:%M')
    end=datetime.strptime(self.p['EndTime'],'%Y-%m-%dT%H:%M')
    original_end=self.p['EndTime']
    if (end-start).total_seconds() <= 600:
        self.p['EndTime']=(start+timedelta(seconds=900)).strftime('%Y-%m-%dT%H:%M')
        try:
            return _original_tran_cover(self)
        finally:
            self.p['EndTime']=original_end
    return _original_tran_cover(self)

csg_shape.csg_shape.tran_cover = _tran_cover_with_minimum_support

forcing=pd.read_csv(FORCING)
init=json.loads(INIT.read_text())
STATE_CASES=[
    {'state_id':'S_COOL_HUMID','air_temperature_c':20.0,'air_rh_pct':80.0,
     'air_vapor_pressure_pa':float(0.80*sat_vp(20.0)),'canopy_temperature_c':20.5},
    {'state_id':'S_NOMINAL','air_temperature_c':24.0,'air_rh_pct':70.0,
     'air_vapor_pressure_pa':float(init['air_vapor_pressure_pa']),'canopy_temperature_c':24.5},
    {'state_id':'S_WARM_DRY','air_temperature_c':30.0,'air_rh_pct':55.0,
     'air_vapor_pressure_pa':float(0.55*sat_vp(30.0)),'canopy_temperature_c':30.5},
]
STATE_MAP={s['state_id']:s for s in STATE_CASES}


def build_model(horizon_s,state_id,forcing_row,c0,vent):
    row=forcing.iloc[int(forcing_row)]
    s=STATE_MAP[state_id]
    p=example.parameters()
    p['StartTime']='2017-09-01T00:00'
    p['EndTime']=(datetime.fromisoformat('2017-09-01T00:00')+timedelta(seconds=float(horizon_s))).strftime('%Y-%m-%dT%H:%M')
    p['dtsim']=int(horizon_s); p['dt']=int(DT)
    p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'
    p['T_soilbound']=float(row['soil_boundary_temperature_c'])
    sv=list(p['StateVariable']); iv=np.asarray(p['InitialValues'],dtype=float).copy()
    for name,val in {'T_air':s['air_temperature_c'],'VP':s['air_vapor_pressure_pa'],
                     'CO2':float(c0),'T_can':s['canopy_temperature_c']}.items():
        iv[sv.index(name)]=val
    p['InitialValues']=iv
    tsim=np.asarray([0.0,float(horizon_s)])
    x0={name:p['InitialValues'][i] for i,name in enumerate(sv)}
    model=CSG_Climate(tsim,p['dt'],x0,p)
    start=datetime.fromisoformat('2017-09-01T08:00:00')+timedelta(seconds=float(row['time_s']))
    end=start+timedelta(seconds=float(horizon_s))
    model.p['StartTime']=start.strftime('%Y-%m-%dT%H:%M')
    model.p['EndTime']=end.strftime('%Y-%m-%dT%H:%M')
    model.D=csg_shape.csg_shape(model.p)
    model.d={
        'f_Rad':const(row['global_radiation_w_m2']),
        'f_Tem':const(row['outdoor_temperature_c']),
        'f_RH':const(row['outdoor_rh_pct']/100.0),
        'f_CO2':const(row['outdoor_co2_ppm']),
        'f_Wind':const(row['wind_speed_m_s']),
        'f_Tsky':const(row['sky_temperature_c']),
    }
    model.U={'u_blanket':const(0.0),'u_vent':const(vent),'u_venttop':const(1.0),
             'u_ventside':const(0.0),'u_venttopbot':const(0.0)}
    return model,sv


arm_cache={}; runtime_errors=[]

def native_arm(h,s,fr,c0,vent):
    key=(int(h),s,int(fr),round(float(c0),10),float(vent))
    if key in arm_cache:
        return arm_cache[key]
    model=None; orig_ctl=csg_fun.ctl_csg1; action_values=[]
    try:
        model,sv=build_model(h,s,fr,c0,vent)
        def ctl_wrap(p_,D,d,Tair,t,U):
            res=orig_ctl(p_,D,d,Tair,t,U)
            action_values.append(scalar(res[1]))
            return res
        csg_fun.ctl_csg1=ctl_wrap
        y=model.run((0.0,float(h)))
        final=float(np.asarray(y['CO2'])[-1])
        finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in sv))
        action_err=float(np.max(np.abs(np.asarray(action_values,dtype=float)-vent))) if action_values else float('inf')
        out={'final_co2_ppm':final,'finite':finite,'max_requested_applied_error':action_err,'error':None}
    except Exception as exc:
        out={'final_co2_ppm':float('nan'),'finite':False,'max_requested_applied_error':float('inf'),'error':repr(exc)}
        runtime_errors.append({'key':list(key),'error':repr(exc)})
    finally:
        csg_fun.ctl_csg1=orig_ctl
    arm_cache[key]=out
    return out


def response(h,s,fr,c0):
    lo=native_arm(h,s,fr,c0,LOW); hi=native_arm(h,s,fr,c0,HIGH)
    if not (lo['finite'] and hi['finite']):
        return float('nan')
    return float(hi['final_co2_ppm']-lo['final_co2_ppm'])


def refine_bracket(h,s,fr,lo,hi,flo,fhi):
    initial_lo,initial_hi=float(lo),float(hi)
    if abs(flo)<=RESPONSE_TOL:
        return {'root_ppm':float(lo),'response_ppm':float(flo),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':initial_lo,'initial_bracket_high_ppm':initial_hi,'converged':True}
    if abs(fhi)<=RESPONSE_TOL:
        return {'root_ppm':float(hi),'response_ppm':float(fhi),'iterations':0,'final_bracket_width_ppm':0.0,
                'initial_bracket_low_ppm':initial_lo,'initial_bracket_high_ppm':initial_hi,'converged':True}
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo*fhi>0:
        raise AssertionError('Invalid sign-changing bracket')
    for it in range(1,51):
        mid=0.5*(lo+hi); fm=response(h,s,fr,mid)
        if not np.isfinite(fm):
            return {'root_ppm':float('nan'),'response_ppm':float('nan'),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':initial_lo,'initial_bracket_high_ppm':initial_hi,'converged':False}
        if abs(fm)<=RESPONSE_TOL or (hi-lo)<=ROOT_WIDTH_TOL:
            return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':it,'final_bracket_width_ppm':float(hi-lo),
                    'initial_bracket_low_ppm':initial_lo,'initial_bracket_high_ppm':initial_hi,'converged':True}
        if flo*fm<=0:
            hi,fhi=mid,fm
        else:
            lo,flo=mid,fm
    mid=0.5*(lo+hi); fm=response(h,s,fr,mid)
    return {'root_ppm':float(mid),'response_ppm':float(fm),'iterations':50,'final_bracket_width_ppm':float(hi-lo),
            'initial_bracket_low_ppm':initial_lo,'initial_bracket_high_ppm':initial_hi,
            'converged':bool((hi-lo)<=ROOT_WIDTH_TOL or abs(fm)<=RESPONSE_TOL)}


def dedup_roots(roots):
    good=[r for r in roots if np.isfinite(r['root_ppm'])]; good.sort(key=lambda r:r['root_ppm'])
    out=[]
    for r in good:
        if not out or abs(r['root_ppm']-out[-1]['root_ppm'])>ROOT_DEDUP_TOL:
            out.append(r)
        elif abs(r['response_ppm'])<abs(out[-1]['response_ppm']):
            out[-1]=r
    return out


def scan_cell(h,s,fr):
    ys=[response(h,s,fr,float(c0)) for c0 in SCAN_GRID]
    candidates=[]
    for i,yv in enumerate(ys):
        if np.isfinite(yv) and abs(yv)<=RESPONSE_TOL:
            candidates.append(refine_bracket(h,s,fr,SCAN_GRID[i],SCAN_GRID[i],yv,yv))
    for i in range(len(SCAN_GRID)-1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i+1]) and ys[i]*ys[i+1]<0:
            candidates.append(refine_bracket(h,s,fr,SCAN_GRID[i],SCAN_GRID[i+1],ys[i],ys[i+1]))
    return ys,dedup_roots(candidates)


trace_rows=[]

def flux_arm(h,s,fr,c0,vent,condition_id,point_type):
    model,sv=build_model(h,s,fr,c0,vent)
    records=[]; action_values=[]; ctx={}
    orig_diff=model.diff; orig_ctl=csg_fun.ctl_csg1
    orig_photo=BVtomato_fun.BramVanthoorPhotoSynthesis; orig_smooth2=BVtomato_fun.BoSmoth2
    def ctl_wrap(p_,D,d,Tair,t,U):
        res=orig_ctl(p_,D,d,Tair,t,U)
        ctx['Vent']=scalar(res[4]); action_values.append(scalar(res[1])); return res
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
        records.append({'time_s':float(t),'co2_ppm':CO2,'ventilation_exchange':vent_flux,
                        'photosynthesis_uptake':photo,'organic_respiration':resp,'soil_respiration':soil,
                        'external_co2_source':ext,'residual':residual,'total_dco2_dt':float(dy[2])})
        return dy
    csg_fun.ctl_csg1=ctl_wrap; BVtomato_fun.BramVanthoorPhotoSynthesis=photo_wrap; BVtomato_fun.BoSmoth2=smooth2_wrap; model.diff=diff_wrap
    try:
        y=model.run((0.0,float(h)))
    finally:
        model.diff=orig_diff; csg_fun.ctl_csg1=orig_ctl
        BVtomato_fun.BramVanthoorPhotoSynthesis=orig_photo; BVtomato_fun.BoSmoth2=orig_smooth2
    final=float(np.asarray(y['CO2'])[-1])
    finite=bool(all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in sv))
    action_err=float(np.max(np.abs(np.asarray(action_values,dtype=float)-vent))) if action_values else float('inf')
    expected_steps=int(round(float(h)/DT))
    if len(records)!=expected_steps:
        raise AssertionError(f'Expected {expected_steps} Euler flux evaluations, got {len(records)}')
    ints={name:float(sum(r[name] for r in records)*DT) for name in TERM_NAMES}
    total_int=float(sum(r['total_dco2_dt'] for r in records)*DT)
    named_sum=float(sum(ints.values())); actual=float(final-c0)
    arm_label='LOW' if abs(vent-LOW)<1e-12 else 'HIGH'
    for r in records:
        trace_rows.append({'model_id':'M2','condition_id':condition_id,'point_type':point_type,
                           'horizon_s':int(h),'state_id':s,'forcing_row':int(fr),'initial_co2_ppm':float(c0),
                           'arm':arm_label,**r})
    return {'initial_co2_ppm':float(c0),'ventilation_command_fraction':float(vent),'final_co2_ppm':final,
            'finite':finite,'max_requested_applied_error':action_err,
            'integrated_flux_contributions_ppm':ints,'integrated_total_ode_change_ppm':total_int,
            'sum_named_flux_contributions_ppm':named_sum,'actual_co2_change_ppm':actual,
            'named_vs_total_ode_closure_ppm':float(named_sum-total_int),
            'total_ode_vs_actual_closure_ppm':float(total_int-actual),
            'max_abs_instantaneous_residual_ppm_s':float(max(abs(r['residual']) for r in records))}


def paired_decomp(h,s,fr,c0,condition_id,point_type):
    lo=flux_arm(h,s,fr,c0,LOW,condition_id,point_type); hi=flux_arm(h,s,fr,c0,HIGH,condition_id,point_type)
    parts={name:float(hi['integrated_flux_contributions_ppm'][name]-lo['integrated_flux_contributions_ppm'][name]) for name in TERM_NAMES}
    actual=float(hi['final_co2_ppm']-lo['final_co2_ppm']); reconstructed=float(sum(parts.values()))
    return {'horizon_s':int(h),'state_id':s,'forcing_row':int(fr),'initial_co2_ppm':float(c0),
            'LOW':lo,'HIGH':hi,'high_minus_low':{'final_co2_ppm':actual,'flux_contributions_ppm':parts,
            'reconstructed_final_co2_ppm':reconstructed,'closure_error_ppm':float(reconstructed-actual)}}


def group_vector(v):
    return {'direct_exchange':float(v['ventilation_exchange']),
            'canopy_biology':float(v['photosynthesis_uptake']+v['organic_respiration']),
            'soil_biology':float(v['soil_respiration']),'external_source':float(v['external_co2_source']),
            'residual':float(v['residual'])}


def vector_metrics(v,anchor=None):
    arr=np.asarray([v[k] for k in VECTOR_NAMES],dtype=float); l1=float(np.sum(np.abs(arr)))
    norm=arr/l1 if l1>0 else np.zeros_like(arr); idx=int(np.argmax(np.abs(arr))) if l1>0 else 0
    out={'l1_total_abs_ppm':l1,'dominant_term':VECTOR_NAMES[idx] if l1>0 else 'NONE',
         'dominance_share':float(np.max(np.abs(norm))) if l1>0 else 0.0,
         'active_terms':'|'.join(VECTOR_NAMES[i] for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE),
         'active_sign_pattern':'|'.join(f"{VECTOR_NAMES[i]}:{'+' if x>0 else '-'}" for i,x in enumerate(norm) if abs(x)>=ACTIVE_SHARE)}
    for i,k in enumerate(VECTOR_NAMES): out[f'norm_{k}']=float(norm[i])
    g=group_vector(v); garr=np.asarray([g[k] for k in GROUP_NAMES],dtype=float); gl1=float(np.sum(np.abs(garr)))
    if gl1>0:
        gi=int(np.argmax(np.abs(garr))); out['dominant_group']=GROUP_NAMES[gi]; out['dominant_group_share']=float(abs(garr[gi])/gl1)
    else:
        out['dominant_group']='NONE'; out['dominant_group_share']=0.0
    if anchor is not None:
        aa=np.asarray([anchor[k] for k in VECTOR_NAMES],dtype=float); al1=float(np.sum(np.abs(aa))); an=aa/al1 if al1>0 else np.zeros_like(aa)
        den=float(np.linalg.norm(norm)*np.linalg.norm(an)); out['anchor_cosine_similarity']=float(np.dot(norm,an)/den) if den>0 else float('nan')
        out['anchor_l1_distance']=float(np.sum(np.abs(norm-an)))
        aidx=int(np.argmax(np.abs(aa))) if al1>0 else 0
        out['dominant_switch_from_anchor']=bool(out['dominant_term']!=(VECTOR_NAMES[aidx] if al1>0 else 'NONE'))
        shared=[i for i in range(len(VECTOR_NAMES)) if abs(norm[i])>=ACTIVE_SHARE and abs(an[i])>=ACTIVE_SHARE]
        flips=[VECTOR_NAMES[i] for i in shared if np.sign(norm[i])!=np.sign(an[i])]
        out['active_term_sign_flips_vs_anchor']='|'.join(flips); out['n_active_term_sign_flips_vs_anchor']=int(len(flips))
    return out


cell_rows=[]; scan_rows=[]; root_objects=[]; edge_objects=[]; all_pairs=[]
for h in HORIZONS:
    for st in STATE_CASES:
        sid=st['state_id']
        for fr in FORCING_ROWS:
            cid=f'H{h}_{sid}_FROW{fr}'
            ys,roots=scan_cell(h,sid,fr)
            for c0,yv in zip(SCAN_GRID,ys):
                scan_rows.append({'model_id':'M2','condition_id':cid,'horizon_s':h,'state_id':sid,'forcing_row':fr,
                                  'initial_co2_ppm':float(c0),'high_minus_low_final_co2_ppm':float(yv)})
            topology='UNIQUE_CROSSING_IN_WINDOW' if len(roots)==1 else ('NO_CROSSING_IN_WINDOW' if len(roots)==0 else 'MULTIPLE_CROSSINGS_IN_WINDOW')
            primary=float(roots[0]['root_ppm']) if len(roots)==1 else float('nan')
            cell_rows.append({'model_id':'M2','condition_id':cid,'horizon_s':h,'state_id':sid,'forcing_row':fr,
                              'root_count':len(roots),'topology':topology,'primary_boundary_ppm':primary,
                              'response_at_250_ppm':float(ys[0]),'response_at_600_ppm':float(ys[-1])})
            if len(roots)==1:
                pair=paired_decomp(h,sid,fr,primary,cid,'ROOT'); pair['condition_id']=cid; pair['point_type']='ROOT'; pair['root_meta']=roots[0]
                root_objects.append(pair); all_pairs.append(pair)
            elif len(roots)==0:
                for edge in [250.0,600.0]:
                    pair=paired_decomp(h,sid,fr,edge,cid,'EDGE_AUDIT'); pair['condition_id']=cid; pair['point_type']='EDGE_AUDIT'; pair['edge_ppm']=edge
                    edge_objects.append(pair); all_pairs.append(pair)
            else:
                for r in roots:
                    pair=paired_decomp(h,sid,fr,float(r['root_ppm']),cid,'MULTIPLE_ROOT_AUDIT'); pair['condition_id']=cid; pair['point_type']='MULTIPLE_ROOT_AUDIT'; pair['root_meta']=r
                    root_objects.append(pair); all_pairs.append(pair)

anchor_matches=[x for x in root_objects if x['condition_id']==ANCHOR_CONDITION and x['point_type']=='ROOT']
if len(anchor_matches)!=1: raise AssertionError(f'Anchor root count != 1: {len(anchor_matches)}')
anchor_obj=anchor_matches[0]; anchor_vec=anchor_obj['high_minus_low']['flux_contributions_ppm']

root_rows=[]; edge_rows=[]; term_rows=[]
for obj in root_objects+edge_objects:
    v=obj['high_minus_low']['flux_contributions_ppm']; m=vector_metrics(v,anchor_vec)
    base={'model_id':'M2','condition_id':obj['condition_id'],'point_type':obj['point_type'],'horizon_s':obj['horizon_s'],
          'state_id':obj['state_id'],'forcing_row':obj['forcing_row'],'initial_co2_ppm':obj['initial_co2_ppm'],
          'final_high_minus_low_ppm':obj['high_minus_low']['final_co2_ppm'],'ppm_closure_error':obj['high_minus_low']['closure_error_ppm'],**m}
    if obj['point_type'] in ['ROOT','MULTIPLE_ROOT_AUDIT']: root_rows.append(base)
    else: edge_rows.append(base)
    g=group_vector(v); gl1=sum(abs(x) for x in g.values())
    for k in VECTOR_NAMES:
        term_rows.append({**{kk:base[kk] for kk in ['model_id','condition_id','point_type','horizon_s','state_id','forcing_row','initial_co2_ppm']},
                          'term':k,'contribution_ppm':float(v[k]),'abs_share':float(abs(v[k])/m['l1_total_abs_ppm']) if m['l1_total_abs_ppm']>0 else 0.0})
    for k in GROUP_NAMES:
        term_rows.append({**{kk:base[kk] for kk in ['model_id','condition_id','point_type','horizon_s','state_id','forcing_row','initial_co2_ppm']},
                          'term':f'GROUP::{k}','contribution_ppm':float(g[k]),'abs_share':float(abs(g[k])/gl1) if gl1>0 else 0.0})

cells_df=pd.DataFrame(cell_rows); roots_df=pd.DataFrame(root_rows); edges_df=pd.DataFrame(edge_rows); terms_df=pd.DataFrame(term_rows); scan_df=pd.DataFrame(scan_rows); trace_df=pd.DataFrame(trace_rows)
cells_df.to_csv(OUT_CELLS,index=False); roots_df.to_csv(OUT_ROOTS,index=False); edges_df.to_csv(OUT_EDGES,index=False)
terms_df.to_csv(OUT_TERMS,index=False); scan_df.to_csv(OUT_SCAN,index=False); trace_df.to_csv(OUT_TRACE,index=False)

all_arms=[obj[k] for obj in all_pairs for k in ['LOW','HIGH']]
max_action=max(a['max_requested_applied_error'] for a in all_arms); all_finite=all(a['finite'] for a in all_arms)
max_arm_closure=max(abs(a['total_ode_vs_actual_closure_ppm']) for a in all_arms)
max_pair=max(abs(obj['high_minus_low']['closure_error_ppm']) for obj in all_pairs)
max_inst=max(a['max_abs_instantaneous_residual_ppm_s'] for a in all_arms)
unique_count=int((cells_df.root_count==1).sum()); no_count=int((cells_df.root_count==0).sum()); multiple_count=int((cells_df.root_count>1).sum())
anchor_root=float(anchor_obj['initial_co2_ppm']); anchor_err=abs(anchor_root-EXP1_4_ANCHOR)
all_roots_inside=all(obj.get('root_meta',{}).get('initial_bracket_low_ppm',-np.inf)-1e-12<=obj['initial_co2_ppm']<=obj.get('root_meta',{}).get('initial_bracket_high_ppm',np.inf)+1e-12 for obj in root_objects)
dominant_counts=roots_df['dominant_term'].value_counts().to_dict() if len(roots_df) else {}; switch_count=int(roots_df['dominant_switch_from_anchor'].sum()) if len(roots_df) else 0

result={'experiment':'PhysBench-GH EXP1.5','model_id':'M2','model':'CSGtom','frozen_commit':'bea8c3b0a1324162a4b5487db578aa674c8b587c',
        'benchmark_forcing_sha256':sha(FORCING),'matched_initial_state_sha256':sha(INIT),
        'surface_regression':{'unique_root_cells':unique_count,'no_crossing_cells':no_count,'multiple_root_cells':multiple_count,
                              'expected':[EXPECTED_UNIQUE,EXPECTED_NO_ROOT,EXPECTED_MULTIPLE],
                              'anchor_condition':ANCHOR_CONDITION,'anchor_root_ppm':anchor_root,'anchor_abs_error_ppm':anchor_err,
                              'all_roots_inside_detected_brackets':bool(all_roots_inside)},
        'mechanism_summary':{'native_vector_names':VECTOR_NAMES,'anchor_vector_ppm':anchor_vec,'anchor_metrics':vector_metrics(anchor_vec,anchor_vec),
                             'dominant_term_counts_across_root_cells':dominant_counts,'dominant_switch_count_from_anchor':switch_count,
                             'n_root_mechanisms':int(len(roots_df)),'n_edge_audits':int(len(edges_df))},
        'runtime_audit':{'all_finite':bool(all_finite),'runtime_errors':runtime_errors,'max_requested_applied_error':float(max_action),
                         'max_arm_total_ode_vs_actual_closure_ppm':float(max_arm_closure),'max_pair_decomposition_closure_error_ppm':float(max_pair),
                         'max_abs_instantaneous_residual_ppm_s':float(max_inst),'native_integration_step_s':30,
                         'short_horizon_geometry_adapter':'900-s support construction only; requested dynamics horizon unchanged'},
        'interpretation_limit':'Read-only flux accounting across the frozen EXP1.4 surface. No flux was disabled, repaired, replayed, or retuned.'}
result['run_pass']=bool(max_action==0.0 and all_finite and len(runtime_errors)==0 and max_arm_closure<=1e-8 and max_pair<=1e-8
                        and unique_count==EXPECTED_UNIQUE and no_count==EXPECTED_NO_ROOT and multiple_count==EXPECTED_MULTIPLE
                        and anchor_err<=ANCHOR_TOL and all_roots_inside)
OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['run_pass']: raise SystemExit('M2 EXP1.5 runtime/surface/closure gate failed')
