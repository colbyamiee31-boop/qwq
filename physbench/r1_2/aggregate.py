import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from common import sign_obs
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'evidence/R1_2/FINAL'; OUT.mkdir(parents=True,exist_ok=True)
PIN=pd.read_csv(ROOT/'physbench/r1_2/generated/PRIMARY_97_MODEL_INPUT.csv'); SIN=pd.read_csv(ROOT/'physbench/r1_2/generated/STRICT_36_MODEL_INPUT.csv')
M={m:pd.read_csv(ROOT/f'evidence/R1_2/{m}/model_event_responses.csv') for m in ['M1','M2']}

def kappa(a,b):
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    if n==0:return np.nan
    p0=np.mean(a==b); pa=np.mean(a==1); pb=np.mean(b==1); pe=pa*pb+(1-pa)*(1-pb)
    return np.nan if abs(1-pe)<1e-15 else (p0-pe)/(1-pe)

def metric(frame,var):
    ms=frame[f'model_{var}_sign'].to_numpy(int); od=frame[f'obs_{var}_delta'].to_numpy(float); os=np.array([sign_obs(x) for x in od])
    keep=(ms!=0)&(os!=0); ms=ms[keep]; os=os[keep]; f=frame.loc[keep].copy()
    conc=float(np.mean(ms==os)) if len(ms) else np.nan; kap=float(kappa(ms,os)) if len(ms) else np.nan
    grad=np.abs(f[f'{var}_gradient'].to_numpy(float)); du=np.abs(f.delta_u.to_numpy(float)); denom=grad*du; good=denom>1e-12
    rho=float(spearmanr(f.loc[good,f'model_{var}_delta']/denom[good],f.loc[good,f'obs_{var}_delta']/denom[good]).statistic) if good.sum()>=3 else np.nan
    return {'n_resolved':int(len(ms)),'concordance':conc,'kappa':kap,'spearman_norm':rho,'unresolved':int(len(frame)-len(ms))}

def bootstrap(frame,var,seed=20261004,B=2000):
    dates=np.array(sorted(frame.event_date.unique())); groups={d:frame[frame.event_date==d] for d in dates}; rng=np.random.default_rng(seed); vals=[]
    for _ in range(B):
        sample=rng.choice(dates,size=len(dates),replace=True); b=pd.concat([groups[d] for d in sample],ignore_index=True); q=metric(b,var); vals.append([q['concordance'],q['kappa'],q['spearman_norm']])
    A=np.asarray(vals,float); out=[]
    for j,name in enumerate(['concordance','kappa','spearman_norm']):
        v=A[:,j]; v=v[np.isfinite(v)]; out.append((name,float(np.percentile(v,2.5)) if len(v) else np.nan,float(np.percentile(v,97.5)) if len(v) else np.nan))
    return dict((n,{'lo':lo,'hi':hi}) for n,lo,hi in out)

def merge(cohort,model):
    inp=PIN if cohort=='primary' else SIN; mm=M[model]; mm=mm[mm.cohort==cohort].copy(); q=mm.merge(inp,on='event_id',suffixes=('','_in'),validate='many_to_one')
    q['obs_T_delta']=np.where(q.horizon_min==15,q.obs_T_matched_15,q.obs_T_matched_30); q['obs_AH_delta']=np.where(q.horizon_min==15,q.obs_AH_matched_15,q.obs_AH_matched_30); q['T_gradient']=q.T_gradient_C; q['AH_gradient']=q.AH_gradient_g_m3
    return q

rows=[]; event_tables=[]
for cohort in ['primary','strict']:
  for model in ['M1','M2']:
    q=merge(cohort,model); q['model']=model; event_tables.append(q)
    for h in ([15,30] if cohort=='primary' else [15]):
      base=q[q.horizon_min==h]
      groups=[('all',base)]
      if cohort=='primary' and h==15:
        groups += [('opening',base[base.direction=='opening']),('closing',base[base.direction=='closing']),('day',base[base.daynight=='day']),('night',base[base.daynight=='night'])]
      for gname,g in groups:
        for var in ['T','AH']:
          met=metric(g,var); ci=bootstrap(g,var) if len(g)>=3 and g.event_date.nunique()>=2 else {}
          rows.append({'cohort':cohort,'model':model,'horizon_min':h,'group':gname,'variable':var,**met,'concordance_ci_low':ci.get('concordance',{}).get('lo',np.nan),'concordance_ci_high':ci.get('concordance',{}).get('hi',np.nan),'kappa_ci_low':ci.get('kappa',{}).get('lo',np.nan),'kappa_ci_high':ci.get('kappa',{}).get('hi',np.nan),'spearman_ci_low':ci.get('spearman_norm',{}).get('lo',np.nan),'spearman_ci_high':ci.get('spearman_norm',{}).get('hi',np.nan)})
metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'validation_metrics.csv',index=False,float_format='%.12g')
allq=pd.concat(event_tables,ignore_index=True); allq.to_csv(OUT/'event_level_validation.csv',index=False,float_format='%.12g')

comp=[]; rng=np.random.default_rng(20261004)
for var in ['T','AH']:
    a=merge('primary','M1'); b=merge('primary','M2'); a=a[a.horizon_min==15]; b=b[b.horizon_min==15]
    x=a[['event_id','event_date',f'model_{var}_sign',f'obs_{var}_delta']].merge(b[['event_id',f'model_{var}_sign']],on='event_id',suffixes=('_M1','_M2'))
    obs=np.array([sign_obs(z) for z in x[f'obs_{var}_delta']]); m1=x[f'model_{var}_sign_M1'].to_numpy(int); m2=x[f'model_{var}_sign_M2'].to_numpy(int); keep=(obs!=0)&(m1!=0)&(m2!=0); x=x.loc[keep].copy(); obs=obs[keep]; m1=m1[keep]; m2=m2[keep]; diff=float(np.mean(m1==obs)-np.mean(m2==obs))
    dates=np.array(sorted(x.event_date.unique())); vals=[]; tmp=pd.DataFrame({'date':x.event_date.to_numpy(),'obs':obs,'m1':m1,'m2':m2}); gd={d:tmp[tmp.date==d] for d in dates}
    for _ in range(2000):
        ss=rng.choice(dates,size=len(dates),replace=True); z=pd.concat([gd[d] for d in ss],ignore_index=True); vals.append(np.mean(z.m1==z.obs)-np.mean(z.m2==z.obs))
    comp.append({'variable':var,'n':len(tmp),'M1_minus_M2_concordance':diff,'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5))})
pd.DataFrame(comp).to_csv(OUT/'paired_model_concordance_difference.csv',index=False,float_format='%.12g')

s1=json.loads((ROOT/'evidence/R1_2/M1/summary.json').read_text()); s2=json.loads((ROOT/'evidence/R1_2/M2/summary.json').read_text())
summary={'primary_matched_events':97,'strict_matched_events':36,'M1_gate':s1['gate_pass'],'M2_gate':s2['gate_pass'],'primary_metrics_rows':int(len(metrics[(metrics.cohort=='primary')&(metrics.group=='all')])),'scientific_pass_fail_not_used':True}
summary['gate_pass']=bool(s1['gate_pass'] and s2['gate_pass'] and summary['primary_metrics_rows']==8)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R1.2 aggregate gate failed')
