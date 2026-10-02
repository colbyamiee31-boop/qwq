import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from common import sign_obs,sha256
ROOT=Path(__file__).resolve().parents[2]; IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'; CFG=json.loads((ROOT/'physbench/exp2_1d/scenario_config.json').read_text()); E=ROOT/'evidence'; OUT=E/'FINAL'; OUT.mkdir(parents=True,exist_ok=True)
obs=pd.read_csv(IN); models={m:pd.read_csv(E/m/'robust_event_responses.csv') for m in ['M1','M2']}; B=int(CFG['bootstrap']['resamples']); SEED=int(CFG['bootstrap']['seed']); anchor='STD_0p1_0p9__TCAN_+0.0__SNAPSHOT_T0'

def sign_ag(g,v):
 a=g[f'model_{v}_sign'].to_numpy(int); b=g[f'obs_{v}_sign'].to_numpy(int); z=(a!=0)&(b!=0); return float(np.mean(a[z]==b[z])) if z.any() else np.nan
def rho(g,v):
 a=g[f'model_{v}_norm'].to_numpy(float); b=g[f'obs_{v}_norm'].to_numpy(float); z=np.isfinite(a)&np.isfinite(b); return float(spearmanr(a[z],b[z]).statistic) if z.sum()>=3 else np.nan
def weighted_midrank(vals,W):
 vals=np.asarray(vals,float); order=np.argsort(vals,kind='mergesort'); sv=vals[order]; R=np.zeros_like(W,dtype=float); prev=np.zeros(W.shape[0]); i=0
 while i<len(vals):
  j=i+1
  while j<len(vals) and sv[j]==sv[i]: j+=1
  idx=order[i:j]; gw=W[:,idx].sum(axis=1); rr=prev+(gw+1)/2; R[:,idx]=rr[:,None]; prev+=gw; i=j
 return R
def boot_primary(g,v,seedoff):
 dates=np.array(sorted(g.event_date.astype(str).unique())); di={d:i for i,d in enumerate(dates)}; eidx=np.array([di[x] for x in g.event_date.astype(str)]); rng=np.random.default_rng(SEED+seedoff); counts=rng.multinomial(len(dates),np.ones(len(dates))/len(dates),size=B); W=counts[:,eidx].astype(float)
 ms=g[f'model_{v}_sign'].to_numpy(int); os=g[f'obs_{v}_sign'].to_numpy(int); resolved=((ms!=0)&(os!=0)).astype(float); corr=((ms==os)&(ms!=0)&(os!=0)).astype(float); den=(W*resolved).sum(1); frac=np.divide((W*corr).sum(1),den,out=np.full(B,np.nan),where=den>0)
 x=g[f'model_{v}_norm'].to_numpy(float); y=g[f'obs_{v}_norm'].to_numpy(float); Rx=weighted_midrank(x,W); Ry=weighted_midrank(y,W); sw=W.sum(1); mx=(W*Rx).sum(1)/sw; my=(W*Ry).sum(1)/sw; dx=Rx-mx[:,None]; dy=Ry-my[:,None]; cov=(W*dx*dy).sum(1); vx=(W*dx*dx).sum(1); vy=(W*dy*dy).sum(1); rr=np.divide(cov,np.sqrt(vx*vy),out=np.full(B,np.nan),where=(vx>0)&(vy>0))
 return np.nanquantile(frac,[.025,.975]).tolist(),np.nanquantile(rr,[.025,.975]).tolist()

metrics=[]; eventout=[]; seedoff=0
for m,d in models.items():
 for sid,ds in d.groupby('scenario_id',sort=True):
  for h in [15,30]:
   g=ds[ds.horizon_min.eq(h)].merge(obs[['event_id','obs_matched_T_15','obs_matched_AH_15','obs_T_norm_15','obs_AH_norm_15','obs_matched_T_30','obs_matched_AH_30','obs_T_norm_30','obs_AH_norm_30']],on='event_id',validate='one_to_one')
   for v in ['T','AH']:
    g[f'obs_{v}_delta']=g[f'obs_matched_{v}_{h}']; g[f'obs_{v}_norm']=g[f'obs_{v}_norm_{h}']; g[f'obs_{v}_sign']=g[f'obs_{v}_delta'].map(sign_obs)
   g['model']=m; eventout.append(g)
   strata=['all','day','night'] if h==15 else ['all']
   for st in strata:
    s=g if st=='all' else g[g.daynight.eq(st)]
    for v in ['T','AH']:
     ci=[np.nan,np.nan]; rci=[np.nan,np.nan]
     if h==15 and st=='all': ci,rci=boot_primary(s,v,seedoff); seedoff+=1
     metrics.append({'model':m,'scenario_id':sid,'mapping':s.mapping.iloc[0],'canopy_offset_c':s.canopy_offset_c.iloc[0],'forcing':s.forcing.iloc[0],'horizon_min':h,'stratum':st,'variable':v,'n_events':len(s),'sign_concordance_fraction':sign_ag(s,v),'sign_ci_low':ci[0],'sign_ci_high':ci[1],'spearman_rho':rho(s,v),'rho_ci_low':rci[0],'rho_ci_high':rci[1],'median_model_norm':float(np.nanmedian(s[f'model_{v}_norm'])),'median_obs_norm':float(np.nanmedian(s[f'obs_{v}_norm']))})
met=pd.DataFrame(metrics); prim=met[(met.horizon_min==15)&(met.stratum=='all')].copy(); a=prim[prim.scenario_id.eq(anchor)][['model','variable','sign_concordance_fraction','spearman_rho','median_model_norm']].rename(columns={'sign_concordance_fraction':'anchor_sign','spearman_rho':'anchor_rho','median_model_norm':'anchor_median_norm'}); prim=prim.merge(a,on=['model','variable']); prim['sign_diff_from_anchor']=prim.sign_concordance_fraction-prim.anchor_sign; prim['rho_diff_from_anchor']=prim.spearman_rho-prim.anchor_rho; prim['median_norm_diff_from_anchor']=prim.median_model_norm-prim.anchor_median_norm
summ=[]
for (m,v),g in prim.groupby(['model','variable']):
 summ.append({'model':m,'variable':v,'scenario_count':len(g),'anchor_sign':float(g.anchor_sign.iloc[0]),'sign_min':float(g.sign_concordance_fraction.min()),'sign_max':float(g.sign_concordance_fraction.max()),'n_sign_below_0p5':int((g.sign_concordance_fraction<.5).sum()),'n_sign_above_0p5':int((g.sign_concordance_fraction>.5).sum()),'n_sign_ci_contains_0p5':int(((g.sign_ci_low<=.5)&(g.sign_ci_high>=.5)).sum()),'anchor_rho':float(g.anchor_rho.iloc[0]),'rho_min':float(g.spearman_rho.min()),'rho_max':float(g.spearman_rho.max()),'n_rho_ci_contains_0':int(((g.rho_ci_low<=0)&(g.rho_ci_high>=0)).sum()),'anchor_median_norm':float(g.anchor_median_norm.iloc[0]),'median_norm_min':float(g.median_model_norm.min()),'median_norm_max':float(g.median_model_norm.max()),'max_abs_sign_diff_from_anchor':float(g.sign_diff_from_anchor.abs().max()),'max_abs_rho_diff_from_anchor':float(g.rho_diff_from_anchor.abs().max())})
rob=pd.DataFrame(summ)
cross=[]
for sid in sorted(models['M1'].scenario_id.unique()):
 for h in [15,30]:
  x=models['M1'][(models['M1'].scenario_id==sid)&(models['M1'].horizon_min==h)].set_index('event_id'); y=models['M2'][(models['M2'].scenario_id==sid)&(models['M2'].horizon_min==h)].set_index('event_id')
  for v in ['T','AH']:
   z=(x[f'model_{v}_sign']!=0)&(y[f'model_{v}_sign']!=0); cross.append({'scenario_id':sid,'horizon_min':h,'variable':v,'n_resolved':int(z.sum()),'M1_M2_sign_agreement':float((x.loc[z,f'model_{v}_sign']==y.loc[z,f'model_{v}_sign']).mean())})
all_df=pd.concat(eventout,ignore_index=True); all_df.to_csv(OUT/'event_level_robustness.csv',index=False,float_format='%.12g'); met.to_csv(OUT/'all_metrics.csv',index=False,float_format='%.12g'); prim.to_csv(OUT/'primary_15min_scenario_metrics.csv',index=False,float_format='%.12g'); rob.to_csv(OUT/'robustness_summary.csv',index=False,float_format='%.12g'); pd.DataFrame(cross).to_csv(OUT/'cross_model_sign_agreement.csv',index=False,float_format='%.12g')
fac=[]
for col in ['mapping','canopy_offset_c','forcing']:
 for keys,g in prim.groupby(['model','variable',col]):
  fac.append({'model':keys[0],'variable':keys[1],'factor':col,'level':keys[2],'n_scenarios':len(g),'sign_mean':float(g.sign_concordance_fraction.mean()),'sign_min':float(g.sign_concordance_fraction.min()),'sign_max':float(g.sign_concordance_fraction.max()),'rho_mean':float(g.spearman_rho.mean()),'rho_min':float(g.spearman_rho.min()),'rho_max':float(g.spearman_rho.max()),'median_norm_mean':float(g.median_model_norm.mean())})
pd.DataFrame(fac).to_csv(OUT/'factor_level_summary.csv',index=False,float_format='%.12g')
sums={m:json.loads((E/m/'summary.json').read_text()) for m in ['M1','M2']}; summary={'experiment':'EXP2.1D','status':'PASS_REPRESENTATION_MAPPING_ROBUSTNESS','realized_action_validation':'HOLD','primary_cohort_n':61,'scenario_count_per_model':24,'primary_horizon_min':15,'models':sums,'input_sha256':sha256(IN),'robustness_summary':rob.to_dict(orient='records')}; summary['gate_pass']=bool(all(x['gate_pass'] for x in sums.values()) and len(rob)==4); (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('EXP2.1D aggregate gate failed')
