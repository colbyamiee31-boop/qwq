import argparse,json,sys,datetime as dt,traceback
from pathlib import Path
import numpy as np,pandas as pd
from common import *
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; CSG=ROOT/'CSGtom'; sys.path.insert(0,str(CSG))
from models.csg_climate import CSG_Climate
from parameters import example
from functions import csg_fun,csg_shape

THERMAL=['T_floor','T_soil1','T_soil2','T_soil3','T_soil4','T_soil5','T_can','T_wali','T_wal1','T_wal2','T_wal3','T_wal4','T_wal5','T_wale','T_rofi','T_rof1','T_rofe','T_cov','T_blanki','T_blanke']

def init_values(p,r,init):
    sv=list(p['StateVariable']); iv=np.asarray(p['InitialValues'],dtype=float).copy(); tin=float(r.Tin0_C)
    def put(name,val): iv[sv.index(name)]=float(val)
    if init=='I0':
        for name in THERMAL: put(name,tin)
        soils=['T_soil1','T_soil2','T_soil3','T_soil4','T_soil5']
        for name,val in zip(soils,np.linspace(tin,18.0,5)): put(name,val)
        put('T_air',tin); put('VP',r.VP0_Pa); put('CO2',r.CO20_ppm); put('T_can',tin); put('T_can24',tin)
    elif init=='I1':
        put('T_air',tin); put('VP',r.VP0_Pa); put('CO2',r.CO20_ppm); put('T_can',tin)
    else: raise ValueError(init)
    return sv,iv

def one_try(r,arm_name,cfg,step_s):
    p=example.parameters(); p['outdoorDataFileURL']=str(CSG/'data/example_data.xls'); p['UFileURL']=str(CSG/'data/example_u.xls')
    p['dtsim']=900; p['dt']=int(step_s); p['ctl_vent_type']='timebasedControl'; p['ctl_blank_type']='timebasedControl'; p['T_soilbound']=18.0
    ts=pd.Timestamp(r.timestamp_UTCplus6); start=dt.datetime(2017,ts.month,ts.day,ts.hour,ts.minute); end=start+dt.timedelta(hours=6)
    p['StartTime']=start.strftime('%Y-%m-%dT%H:%M'); p['EndTime']=end.strftime('%Y-%m-%dT%H:%M')
    sv,iv=init_values(p,r,cfg['init']); p['InitialValues']=iv
    tsim=np.arange(0,21600+p['dtsim'],p['dtsim'],dtype=float); x0={name:iv[i] for i,name in enumerate(sv)}; model=CSG_Climate(tsim,p['dt'],x0,p)
    model.p['StartTime']=p['StartTime']; model.p['EndTime']=p['EndTime']; model.D=csg_shape.csg_shape(model.p)
    Tout=np.asarray(r.forcing_Tout_C,float); RH=np.asarray(r.forcing_RHout_pct,float); G=np.asarray(r.forcing_Gout_W_m2,float); W=np.asarray(r.forcing_WS2M_m_s,float)
    model.d={'f_Rad':interp_fun(G),'f_Tem':interp_fun(Tout),'f_RH':interp_fun(RH/100.0),'f_CO2':lambda t:415.0,'f_Wind':interp_fun(W),'f_Tsky':interp_fun(Tout)}
    model.U={'u_blanket':lambda t,v=float(cfg['enclosure']):v,'u_vent':lambda t,v=float(cfg['vent']):v,'u_venttop':lambda t:1.0,'u_ventside':lambda t:0.0,'u_venttopbot':lambda t:0.0}
    rec=[]; orig=csg_fun.ctl_csg1
    def logged(p_,D,d,Tair,t,U):
        res=orig(p_,D,d,Tair,t,U)
        try: rec.append(tuple(float(np.asarray(z).reshape(-1)[0]) for z in res[:2]))
        except Exception: pass
        return res
    csg_fun.ctl_csg1=logged
    try: y=model.run((0.0,21600.0))
    finally: csg_fun.ctl_csg1=orig
    init_err=max(abs(iv[sv.index('T_air')]-float(r.Tin0_C)),abs(iv[sv.index('VP')]-float(r.VP0_Pa)),abs(iv[sv.index('CO2')]-float(r.CO20_ppm)))
    t=np.asarray(y['t'],float); out=[]
    for h in HOURS:
        sec=float(h*3600); hits=np.where(np.isclose(t,sec,rtol=0,atol=1e-7))[0]
        if len(hits)!=1: raise RuntimeError(f'missing output h={h}, hits={hits}')
        j=int(hits[0]); T=float(np.asarray(y['T_air'])[j]); VP=float(np.asarray(y['VP'])[j])
        out.append({'horizon_h':h,'T_C':T,'VP_Pa':VP,'AH_g_m3':float(ah_from_t_vp(T,VP)),'top_minus_air_C':np.nan})
    finite=all(np.all(np.isfinite(np.asarray(y[k],float))) for k in sv)
    vent_err=max([abs(v[1]-float(cfg['vent'])) for v in rec],default=np.nan)
    return out,init_err,vent_err,finite

def run_arm(r,arm,cfg):
    try:
        tr,ie,ae,fin=one_try(r,arm,cfg,30)
        if fin: return tr,ie,ae,fin,30,'OK'
    except Exception as ex30:
        first=repr(ex30)
    else:
        first='NONFINITE_30S'
    tr,ie,ae,fin=one_try(r,arm,cfg,10)
    if not fin: raise RuntimeError('nonfinite after 10s retry; first='+first)
    return tr,ie,ae,fin,10,'NUMERICAL_RETRY_10S'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--outdir',required=True); args=ap.parse_args()
    E=load_compact(args.npz,args.shard); od=Path(args.outdir); od.mkdir(parents=True,exist_ok=True); rows=[]; audits=[]
    for r in E:
      for arm,cfg in ARMS.items():
        try:
          tr,ie,ae,fin,step,status=run_arm(r,arm,cfg)
          for q in tr: rows.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'init':cfg['init'],'vent':cfg['vent'],'enclosure':cfg['enclosure'],'integration_step_s':step,**q})
          audits.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'status':status,'integration_step_s':step,'init_max_abs_error':ie,'vent_log_max_abs_error':ae,'finite':int(fin),'traceback':''})
        except Exception as ex:
          audits.append({'atlas_id':r.atlas_id,'greenhouse':r.greenhouse,'arm':arm,'status':'RUNTIME_FAIL','integration_step_s':np.nan,'init_max_abs_error':np.nan,'vent_log_max_abs_error':np.nan,'finite':0,'traceback':repr(ex)+' | '+traceback.format_exc(limit=2).replace('\n',' ')})
    R=pd.DataFrame(rows); A=pd.DataFrame(audits); R.to_csv(od/'m2_replay.csv',index=False,float_format='%.12g'); A.to_csv(od/'m2_audit.csv',index=False,float_format='%.12g')
    expected=len(E)*len(ARMS)*6
    summary={'model':'M2','commit':M2_COMMIT,'atlas_rows':len(E),'arms':len(ARMS),'output_rows':len(R),'expected_rows':expected,'runtime_failures':int((A.status=='RUNTIME_FAIL').sum()),'numerical_retry_10s':int((A.status=='NUMERICAL_RETRY_10S').sum()),'max_init_error':float(A.init_max_abs_error.max()) if A.init_max_abs_error.notna().any() else None}
    summary['engineering_gate_pass']=bool(len(R)==expected and (A.status!='RUNTIME_FAIL').all() and summary['max_init_error']<=1e-7)
    (od/'m2_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    if not summary['engineering_gate_pass']: raise SystemExit('M2 P2.1 engineering gate failed')
if __name__=='__main__': main()
