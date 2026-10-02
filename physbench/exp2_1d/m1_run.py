import json
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym
import gl_gym
from common import *

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'; MED=ROOT/'physbench/exp2_1d/PRE15_FORCING_MEDIANS.csv'
CFG=ROOT/'physbench/exp2_1d/scenario_config.json'
OUT=ROOT/'evidence/M1'; OUT.mkdir(parents=True,exist_ok=True)
EVENTS=pd.read_csv(IN).merge(pd.read_csv(MED),on='event_id',validate='one_to_one'); config=json.loads(CFG.read_text())

def vent_pair(r,m):
    kind=m['kind']
    if kind=='fixed': lo=float(m['low']); hi=float(m['high'])
    elif kind=='absolute_linear': lo=float(r.status_before)/100.; hi=float(r.status_after)/100.
    elif kind=='absolute_compressed': lo=float(m['offset'])+float(m['scale'])*float(r.status_before)/100.; hi=float(m['offset'])+float(m['scale'])*float(r.status_after)/100.
    elif kind=='delta_from_baseline': lo=float(m['baseline']); hi=min(1.0,lo+float(r.status_delta)/100.)
    else: raise ValueError(kind)
    if not (0<=lo<=1 and 0<=hi<=1 and hi>lo): raise AssertionError((r.event_id,m['id'],lo,hi))
    return lo,hi

def forcing(r,fid):
    if fid=='SNAPSHOT_T0':
        return dict(rad=float(r.event_Iglob),tout=float(r.event_Tout),vp=float(r.out_vp_pa),wind=float(r.event_Windsp))
    if fid=='PRE15_MEDIAN':
        return dict(rad=float(r.pre15_Iglob_median),tout=float(r.pre15_Tout_median),vp=float(r.pre15_out_vp_pa_median),wind=float(r.pre15_Windsp_median))
    raise ValueError(fid)

def build_weather(r,fid):
    f=forcing(r,fid); d=np.zeros((3,10),dtype=float)
    d[:,0]=f['rad']; d[:,1]=f['tout']; d[:,2]=f['vp']; d[:,3]=ppm_to_mg_m3(f['tout'],float(r.outdoor_co2_ppm_fixed)); d[:,4]=f['wind']; d[:,5]=f['tout']; d[:,6]=float(r.soil_boundary_temperature_c_fixed)
    d[:,7]=np.cumsum(np.full(3,f['rad'])*900.0)/1e6; d[:,8]=1.0 if f['rad']>0 else 0.0; d[:,9]=d[:,8]
    return d

def run_arm(r,vent,canopy_offset,fid):
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=3407,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped; e.weather_data=build_weather(r,fid); e.day_of_year=int(r.day_of_year); e.hour_of_day=float(r.hour_decimal)
    x=e.x.copy(); x[2]=float(r.event_Tair); x[15]=float(r.in_vp_pa); x[0]=float(ppm_to_mg_m3(float(r.event_Tair),float(r.CO2_pre_ppm))); x[4]=float(r.event_Tair)+float(canopy_offset)
    e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs()
    init={'T':float(e.x[2]),'VP':float(e.x[15]),'CO2ppm':float(mg_m3_to_ppm(e.x[2],e.x[0])),'Tcan':float(e.x[4])}
    a=np.array([0,0,0,vent,0,0],dtype=np.float32); trace=[]; errs=[]; finite=True
    for k in range(2):
        obs,reward,terminated,truncated,info=env.step(a); applied=np.asarray(info['controls'],dtype=float); errs.append(float(np.max(np.abs(applied-a)))); finite=finite and bool(np.all(np.isfinite(e.x)))
        T=float(e.x[2]); VP=float(e.x[15]); trace.append({'horizon_min':15*(k+1),'T':T,'VP':VP,'AH':float(ah_from_t_vp(T,VP))})
    env.close(); return init,trace,max(errs),finite

rows=[]; audits=[]
for m in config['mapping_scenarios']:
  for coff in config['canopy_offsets_c']:
    for fr in config['forcing_representations']:
      sid=f"{m['id']}__TCAN_{coff:+.1f}__{fr['id']}"
      for _,r in EVENTS.iterrows():
        lo_cmd,hi_cmd=vent_pair(r,m); li,lo,le,lf=run_arm(r,lo_cmd,coff,fr['id']); hi,ho,he,hf=run_arm(r,hi_cmd,coff,fr['id'])
        target_tcan=float(r.event_Tair)+float(coff)
        init_err=max(abs(li['T']-float(r.event_Tair)),abs(li['VP']-float(r.in_vp_pa)),abs(li['CO2ppm']-float(r.CO2_pre_ppm)),abs(li['Tcan']-target_tcan),abs(hi['T']-float(r.event_Tair)),abs(hi['VP']-float(r.in_vp_pa)),abs(hi['CO2ppm']-float(r.CO2_pre_ppm)),abs(hi['Tcan']-target_tcan))
        audits.append({'scenario_id':sid,'mapping':m['id'],'canopy_offset_c':coff,'forcing':fr['id'],'event_id':int(r.event_id),'low_command':lo_cmd,'high_command':hi_cmd,'init_max_abs_error':init_err,'action_max_abs_error':max(le,he),'finite':bool(lf and hf)})
        for l,h in zip(lo,ho):
          dt=float(h['T']-l['T']); dah=float(h['AH']-l['AH'])
          rows.append({'scenario_id':sid,'mapping':m['id'],'canopy_offset_c':coff,'forcing':fr['id'],'event_id':int(r.event_id),'team':r.team,'timestamp':r.timestamp,'event_date':r.event_date,'daynight':r.daynight,'horizon_min':int(h['horizon_min']),'low_command':lo_cmd,'high_command':hi_cmd,'T_gradient_C':float(r.T_gradient_C),'AH_gradient_g_m3':float(r.AH_gradient_g_m3),'model_T_high_minus_low':dt,'model_AH_high_minus_low':dah,'model_T_sign':sign_model(dt,h['T'],l['T']),'model_AH_sign':sign_model(dah,h['AH'],l['AH']),'model_T_norm':(-np.sign(float(r.T_gradient_C))*dt)/abs(float(r.T_gradient_C)),'model_AH_norm':(-np.sign(float(r.AH_gradient_g_m3))*dah)/abs(float(r.AH_gradient_g_m3)),'low_T':l['T'],'high_T':h['T'],'low_AH':l['AH'],'high_AH':h['AH']})

df=pd.DataFrame(rows).sort_values(['scenario_id','event_id','horizon_min']); ad=pd.DataFrame(audits).sort_values(['scenario_id','event_id'])
df.to_csv(OUT/'robust_event_responses.csv',index=False,float_format='%.12g'); ad.to_csv(OUT/'runtime_audit.csv',index=False,float_format='%.12g')
anchor_sid='STD_0p1_0p9__TCAN_+0.0__SNAPSHOT_T0'; got=df[df.scenario_id.eq(anchor_sid)].copy(); cols=['event_id','horizon_min','model_T_high_minus_low','model_AH_high_minus_low','model_T_norm','model_AH_norm','low_T','high_T','low_AH','high_AH']; canonical=got[cols].sort_values(['event_id','horizon_min']).to_csv(index=False,float_format='%.12g',lineterminator='\n'); import hashlib; anchor_sha=hashlib.sha256(canonical.encode()).hexdigest(); anchor_ok=(anchor_sha==config['exp2_1c_anchor_response_sha256']['M1'])
summary={'model':'M1','frozen_commit':config['frozen_models']['M1'],'input_sha256':sha256(IN),'events':int(EVENTS.event_id.nunique()),'scenario_count':int(df.scenario_id.nunique()),'rows':len(df),'all_finite':bool(ad.finite.all()),'max_init_error':float(ad.init_max_abs_error.max()),'max_action_error':float(ad.action_max_abs_error.max()),'anchor_response_sha256':anchor_sha,'anchor_reproduced':anchor_ok}
summary['gate_pass']=bool(summary['events']==61 and summary['scenario_count']==24 and summary['rows']==24*61*2 and summary['all_finite'] and summary['max_init_error']<=1e-8 and summary['max_action_error']==0.0 and anchor_ok)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('M1 EXP2.1D runtime gate failed')
