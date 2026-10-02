import json, sys, datetime as _dt
from pathlib import Path
import numpy as np, pandas as pd
from common import *
ROOT=Path(__file__).resolve().parents[2]; CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun, csg_shape
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'; MED=ROOT/'physbench/exp2_1d/PRE15_FORCING_MEDIANS.csv'; CFG=ROOT/'physbench/exp2_1d/scenario_config.json'; OUT=ROOT/'evidence/M2'; OUT.mkdir(parents=True,exist_ok=True)
EVENTS=pd.read_csv(IN).merge(pd.read_csv(MED),on='event_id',validate='one_to_one'); config=json.loads(CFG.read_text())

def const(v): return lambda t,vv=float(v): vv

def vent_pair(r,m):
    k=m['kind']
    if k=='fixed': lo=float(m['low']); hi=float(m['high'])
    elif k=='absolute_linear': lo=float(r.status_before)/100.; hi=float(r.status_after)/100.
    elif k=='absolute_compressed': lo=float(m['offset'])+float(m['scale'])*float(r.status_before)/100.; hi=float(m['offset'])+float(m['scale'])*float(r.status_after)/100.
    elif k=='delta_from_baseline': lo=float(m['baseline']); hi=min(1.0,lo+float(r.status_delta)/100.)
    else: raise ValueError(k)
    if not (0<=lo<=1 and 0<=hi<=1 and hi>lo): raise AssertionError((r.event_id,m['id'],lo,hi))
    return lo,hi

def forcing(r,fid):
    if fid=='SNAPSHOT_T0': return dict(rad=float(r.event_Iglob),tout=float(r.event_Tout),vp=float(r.out_vp_pa),wind=float(r.event_Windsp))
    if fid=='PRE15_MEDIAN': return dict(rad=float(r.pre15_Iglob_median),tout=float(r.pre15_Tout_median),vp=float(r.pre15_out_vp_pa_median),wind=float(r.pre15_Windsp_median))
    raise ValueError(fid)

def run_arm(r,vent,coff,fid):
    f=forcing(r,fid); p=example.parameters(); p['outdoorDataFileURL']=str(CSG/'data/example_data.xls'); p['UFileURL']=str(CSG/'data/example_u.xls'); p['StartTime']='2017-09-01T00:00'; p['EndTime']='2017-09-01T03:00'; p['dtsim']=900; p['dt']=30; p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'; p['T_soilbound']=float(r.soil_boundary_temperature_c_fixed)
    sv=list(p['StateVariable']); iv=p['InitialValues'].copy(); vals={'T_air':float(r.event_Tair),'VP':float(r.in_vp_pa),'CO2':float(r.CO2_pre_ppm),'T_can':float(r.event_Tair)+float(coff)}
    for name,val in vals.items(): iv[sv.index(name)]=val
    p['InitialValues']=iv; tsim=np.arange(0,1800+p['dtsim'],p['dtsim'],dtype=float); x0={name:p['InitialValues'][i] for i,name in enumerate(p['StateVariable'])}; model=CSG_Climate(tsim,p['dt'],x0,p)
    hh=int(float(r.hour_decimal)); mm=int(round((float(r.hour_decimal)-hh)*60))
    if mm==60: hh=(hh+1)%24; mm=0
    st=_dt.datetime(2017,int(r.month),int(r.day),hh,mm); en=st+_dt.timedelta(minutes=30); model.p['StartTime']=st.strftime('%Y-%m-%dT%H:%M'); model.p['EndTime']=en.strftime('%Y-%m-%dT%H:%M'); model.D=csg_shape.csg_shape(model.p)
    model.d={'f_Rad':const(f['rad']),'f_Tem':const(f['tout']),'f_RH':const(f['vp']/float(sat_vp_pa(f['tout']))),'f_CO2':const(r.outdoor_co2_ppm_fixed),'f_Wind':const(f['wind']),'f_Tsky':const(f['tout'])}
    model.U={'u_blanket':lambda t:0.0,'u_vent':lambda t,v=vent:v,'u_venttop':lambda t:1.0,'u_ventside':lambda t:0.0,'u_venttopbot':lambda t:0.0}
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U); rec.append(float(np.asarray(res[1]).reshape(-1)[0])); return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,1800.0))
    finally: csg_fun.ctl_csg1=orig
    init={'T':float(iv[sv.index('T_air')]),'VP':float(iv[sv.index('VP')]),'CO2ppm':float(iv[sv.index('CO2')]),'Tcan':float(iv[sv.index('T_can')])}; t=np.asarray(y['t'],dtype=float); trace=[]
    for sec in (900.,1800.):
        hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-9))[0]
        if len(hits)!=1: raise AssertionError((r.event_id,sec,hits))
        j=int(hits[0]); T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j]); trace.append({'horizon_min':int(sec/60),'T':T,'VP':VP,'AH':float(ah_from_t_vp(T,VP))})
    finite=all(np.all(np.isfinite(np.asarray(y[k],dtype=float))) for k in p['StateVariable']); err=max(abs(v-vent) for v in rec) if rec else float('inf'); return init,trace,float(err),bool(finite)

rows=[]; audits=[]
for m in config['mapping_scenarios']:
  for coff in config['canopy_offsets_c']:
    for fr in config['forcing_representations']:
      sid=f"{m['id']}__TCAN_{coff:+.1f}__{fr['id']}"
      for _,r in EVENTS.iterrows():
        lo_cmd,hi_cmd=vent_pair(r,m); li,lo,le,lf=run_arm(r,lo_cmd,coff,fr['id']); hi,ho,he,hf=run_arm(r,hi_cmd,coff,fr['id']); target_tcan=float(r.event_Tair)+float(coff)
        init_err=max(abs(li['T']-float(r.event_Tair)),abs(li['VP']-float(r.in_vp_pa)),abs(li['CO2ppm']-float(r.CO2_pre_ppm)),abs(li['Tcan']-target_tcan),abs(hi['T']-float(r.event_Tair)),abs(hi['VP']-float(r.in_vp_pa)),abs(hi['CO2ppm']-float(r.CO2_pre_ppm)),abs(hi['Tcan']-target_tcan))
        audits.append({'scenario_id':sid,'mapping':m['id'],'canopy_offset_c':coff,'forcing':fr['id'],'event_id':int(r.event_id),'low_command':lo_cmd,'high_command':hi_cmd,'init_max_abs_error':init_err,'action_max_abs_error':max(le,he),'finite':bool(lf and hf)})
        for l,h in zip(lo,ho):
          dt=float(h['T']-l['T']); dah=float(h['AH']-l['AH']); rows.append({'scenario_id':sid,'mapping':m['id'],'canopy_offset_c':coff,'forcing':fr['id'],'event_id':int(r.event_id),'team':r.team,'timestamp':r.timestamp,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':int(h['horizon_min']),'low_command':lo_cmd,'high_command':hi_cmd,'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'model_T_high_minus_low':dt,'model_AH_high_minus_low':dah,'model_T_sign':sign_model(dt,h['T'],l['T']),'model_AH_sign':sign_model(dah,h['AH'],l['AH']),'model_T_norm':(-np.sign(float(r.T_gradient_C))*dt)/abs(float(r.T_gradient_C)),'model_AH_norm':(-np.sign(float(r.AH_gradient_g_m3))*dah)/abs(float(r.AH_gradient_g_m3)),'low_T':l['T'],'high_T':h['T'],'low_AH':l['AH'],'high_AH':h['AH']})

df=pd.DataFrame(rows).sort_values(['scenario_id','event_id','horizon_min']); ad=pd.DataFrame(audits).sort_values(['scenario_id','event_id']); df.to_csv(OUT/'robust_event_responses.csv',index=False,float_format='%.12g'); ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
anchor_sid='STD_0p1_0p9__TCAN_+0.0__SNAPSHOT_T0'; got=df[df.scenario_id.eq(anchor_sid)].copy(); cols=['event_id','horizon_min','model_T_high_minus_low','model_AH_high_minus_low','model_T_norm','model_AH_norm','low_T','high_T','low_AH','high_AH']; canonical=got[cols].sort_values(['event_id','horizon_min']).to_csv(index=False,float_format='%.12g',lineterminator='\n'); import hashlib; anchor_sha=hashlib.sha256(canonical.encode()).hexdigest(); anchor_ok=(anchor_sha==config['exp2_1c_anchor_response_sha256']['M2'])
summary={'model':'M2','frozen_commit':config['frozen_models']['M2'],'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),'scenario_count':int(df.scenario_id.nunique()),'rows':len(df),'all_finite':bool(ad.finite.all()),'max_init_error':float(ad.init_max_abs_error.max()),'max_action_error':float(ad.action_max_abs_error.max()),'anchor_response_sha256':anchor_sha,'anchor_reproduced':anchor_ok,'native_integration_step_s':30}
summary['gate_pass']=bool(summary['events']==61 and summary['scenario_count']==24 and summary['rows']==24*61*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0 and anchor_ok); (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M2 EXP2.1D runtime gate failed')
