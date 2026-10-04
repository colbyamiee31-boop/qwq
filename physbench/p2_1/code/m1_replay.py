import argparse,json,traceback
from pathlib import Path
import numpy as np,pandas as pd
import gymnasium as gym
import gl_gym
from common import *

def weather(r):
    Tout=np.asarray(r.forcing_Tout_C,float); RH=np.asarray(r.forcing_RHout_pct,float); G=np.asarray(r.forcing_Gout_W_m2,float); W=np.asarray(r.forcing_WS2M_m_s,float)
    d=np.zeros((len(Tout),10),dtype=float)
    d[:,0]=G; d[:,1]=Tout; d[:,2]=rh_to_vp_pa(Tout,RH); d[:,3]=ppm_to_mg_m3(Tout,415.0); d[:,4]=W
    d[:,5]=Tout; d[:,6]=18.0
    d[:,7]=np.cumsum(G*900.0)/1e6
    d[:,8]=(G>=20.0).astype(float); d[:,9]=d[:,8]
    return d

def make_state(e,r,init):
    x=e.x.copy(); tin=float(r.Tin0_C); vp=float(r.VP0_Pa); co2=float(r.CO20_ppm)
    if init=='I0':
        for i in [2,3,4,5,6,7,8,9,17,18,19,20,21]: x[i]=tin
        x[10:15]=np.linspace(tin,18.0,5)
        x[15]=vp; x[16]=vp
        x[0]=ppm_to_mg_m3(tin,co2); x[1]=x[0]
    elif init=='I1':
        x[2]=tin; x[15]=vp; x[0]=ppm_to_mg_m3(tin,co2); x[4]=tin
    else: raise ValueError(init)
    return x

def run_arm(r,arm_name,cfg):
    env=gym.make('gl_gym/GreenLightTomato-v0',controlled_inputs=['uBoil','uCO2','uThScr','uVent','uLamp','uBlScr'],normalize_actions=False,parameter_provider='fixed')
    env.reset(seed=SEED,options={'scenario':{'location':'Amsterdam','growth_year':2010,'start_day':244}})
    e=env.unwrapped; e.weather_data=weather(r)
    ts=pd.Timestamp(r.timestamp_UTCplus6); e.day_of_year=int(ts.dayofyear); e.hour_of_day=ts.hour+ts.minute/60.0
    x=make_state(e,r,cfg['init']); e.x=x.copy(); e.x_prev=x.copy(); e.obs=e._get_obs(); e.timestep=0
    init_err=max(abs(float(e.x[2])-float(r.Tin0_C)),abs(float(e.x[15])-float(r.VP0_Pa)),abs(float(mg_m3_to_ppm(e.x[2],e.x[0]))-float(r.CO20_ppm)))
    a=np.asarray([0.0,0.0,float(cfg['enclosure']),float(cfg['vent']),0.0,0.0],dtype=np.float32)
    out=[]; action_err=0.0; finite=True
    for k in range(24):
        _,_,terminated,truncated,info=env.step(a)
        applied=np.asarray(info['controls'],dtype=float); action_err=max(action_err,float(np.max(np.abs(applied-a))))
        finite=finite and bool(np.all(np.isfinite(e.x)))
        if terminated or truncated: raise RuntimeError(f'terminated/truncated step={k+1}')
        if (k+1)%4==0:
            T=float(e.x[2]); VP=float(e.x[15]); Top=float(e.x[3])
            out.append({'horizon_h':(k+1)//4,'T_C':T,'VP_Pa':VP,'AH_g_m3':float(ah_from_t_vp(T,VP)),'top_minus_air_C':Top-T})
    env.close()
    return out,init_err,action_err,finite

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--outdir',required=True); args=ap.parse_args()
    E=load_compact(args.npz,args.shard); od=Path(args.outdir); od.mkdir(parents=True,exist_ok=True)
    rows=[]; audits=[]
    for r in E:
      for arm,cfg in ARMS.items():
        try:
          tr,ie,ae,fin=run_arm(r,arm,cfg)
          status='OK' if fin else 'NONFINITE'
          for q in tr:
            rows.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'init':cfg['init'],'vent':cfg['vent'],'enclosure':cfg['enclosure'],**q})
          audits.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'status':status,'init_max_abs_error':ie,'action_max_abs_error':ae,'finite':int(fin),'traceback':''})
        except Exception as ex:
          audits.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'status':'RUNTIME_FAIL','init_max_abs_error':np.nan,'action_max_abs_error':np.nan,'finite':0,'traceback':repr(ex)+' | '+traceback.format_exc(limit=2).replace('\n',' ')})
    R=pd.DataFrame(rows); A=pd.DataFrame(audits)
    R.to_csv(od/'m1_replay.csv',index=False,float_format='%.12g'); A.to_csv(od/'m1_audit.csv',index=False,float_format='%.12g')
    expected=len(E)*len(ARMS)*6
    summary={'model':'M1','commit':M1_COMMIT,'atlas_rows':len(E),'arms':len(ARMS),'output_rows':len(R),'expected_rows':expected,'runtime_failures':int((A.status!='OK').sum()),'max_init_error':float(A.init_max_abs_error.max()) if A.init_max_abs_error.notna().any() else None,'max_action_error':float(A.action_max_abs_error.max()) if A.action_max_abs_error.notna().any() else None}
    summary['engineering_gate_pass']=bool(len(R)==expected and (A.status=='OK').all() and summary['max_init_error']<=1e-7 and summary['max_action_error']<=1e-7)
    (od/'m1_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    if not summary['engineering_gate_pass']: raise SystemExit('M1 P2.1 engineering gate failed')
if __name__=='__main__': main()
