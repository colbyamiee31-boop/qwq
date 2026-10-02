import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import sign_obs, sha256

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
E=ROOT/'evidence'; OUT=E/'FINAL'; OUT.mkdir(parents=True,exist_ok=True)
obs=pd.read_csv(IN)
models={m:pd.read_csv(E/m/'model_event_responses.csv') for m in ['M1','M2']}
for m,d in models.items():
    assert set(d.event_id)==set(obs.event_id) and len(d)==122

SEED=20261002; B=2000
rng=np.random.default_rng(SEED)

def cluster_boot(df, value_fn):
    dates=np.array(sorted(df.event_date.astype(str).unique())); vals=[]
    for _ in range(B):
        samp=rng.choice(dates,size=len(dates),replace=True); parts=[]
        for i,dt in enumerate(samp):
            g=df[df.event_date.astype(str).eq(dt)].copy(); g['_cluster_copy']=i; parts.append(g)
        b=pd.concat(parts,ignore_index=True); v=value_fn(b)
        if np.isfinite(v): vals.append(float(v))
    if not vals: return [None,None]
    return [float(np.quantile(vals,0.025)),float(np.quantile(vals,0.975))]

def agreement(g,var):
    modsign=g[f'model_{var}_sign'].astype(int).to_numpy(); obssign=g[f'obs_{var}_sign'].astype(int).to_numpy()
    mask=(modsign!=0)&(obssign!=0)
    return float(np.mean(modsign[mask]==obssign[mask])) if mask.any() else np.nan

def rho(g,var):
    a=g[f'model_{var}_norm'].to_numpy(float); b=g[f'obs_{var}_norm'].to_numpy(float)
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<3 or np.unique(a[mask]).size<2 or np.unique(b[mask]).size<2: return np.nan
    return float(spearmanr(a[mask],b[mask]).statistic)

all_events=[]; metrics=[]
for m,d in models.items():
  for h in [15,30]:
    dm=d[d.horizon_min.eq(h)].copy()
    ocols=['event_id','obs_matched_T_15','obs_matched_AH_15','obs_T_norm_15','obs_AH_norm_15','obs_matched_T_30','obs_matched_AH_30','obs_T_norm_30','obs_AH_norm_30']
    g=dm.merge(obs[ocols],on='event_id',how='left',validate='one_to_one')
    for var in ['T','AH']:
        g[f'obs_{var}_delta']=g[f'obs_matched_{var}_{h}']; g[f'obs_{var}_norm']=g[f'obs_{var}_norm_{h}']; g[f'obs_{var}_sign']=g[f'obs_{var}_delta'].map(sign_obs)
        g[f'{var}_sign_concordant']=((g[f'model_{var}_sign']!=0)&(g[f'obs_{var}_sign']!=0)&(g[f'model_{var}_sign']==g[f'obs_{var}_sign'])).astype(int)
    g['model']=m; all_events.append(g)
    strata=['all'] if h==30 else ['all','day','night']
    for stratum in strata:
      s=g if stratum=='all' else g[g.daynight.eq(stratum)]
      for var in ['T','AH']:
        ag=agreement(s,var); rr=rho(s,var)
        ag_ci=cluster_boot(s,lambda x,v=var: agreement(x,v))
        rr_ci=cluster_boot(s,lambda x,v=var: rho(x,v))
        metrics.append({'model':m,'horizon_min':h,'stratum':stratum,'variable':var,'n_events':len(s),
                        'n_resolved_sign_pairs':int(((s[f'model_{var}_sign']!=0)&(s[f'obs_{var}_sign']!=0)).sum()),
                        'sign_concordance_fraction':ag,'sign_concordance_ci_low':ag_ci[0],'sign_concordance_ci_high':ag_ci[1],
                        'spearman_rho':rr,'spearman_ci_low':rr_ci[0],'spearman_ci_high':rr_ci[1],
                        'median_model_norm':float(np.nanmedian(s[f'model_{var}_norm'])),'median_obs_norm':float(np.nanmedian(s[f'obs_{var}_norm']))})

all_df=pd.concat(all_events,ignore_index=True).sort_values(['model','event_id','horizon_min']); met=pd.DataFrame(metrics)
cross=[]
for h in [15,30]:
    a=models['M1'][models['M1'].horizon_min.eq(h)][['event_id','model_T_sign','model_AH_sign']].set_index('event_id')
    b=models['M2'][models['M2'].horizon_min.eq(h)][['event_id','model_T_sign','model_AH_sign']].set_index('event_id')
    j=a.join(b,lsuffix='_M1',rsuffix='_M2')
    for var in ['T','AH']:
        mask=(j[f'model_{var}_sign_M1']!=0)&(j[f'model_{var}_sign_M2']!=0)
        cross.append({'horizon_min':h,'variable':var,'n_resolved':int(mask.sum()),'M1_M2_sign_agreement':float(np.mean(j.loc[mask,f'model_{var}_sign_M1']==j.loc[mask,f'model_{var}_sign_M2'])) if mask.any() else np.nan})

all_df.to_csv(OUT/'event_level_model_observation_alignment.csv',index=False,float_format='%.12g')
met.to_csv(OUT/'alignment_metrics.csv',index=False,float_format='%.12g')
pd.DataFrame(cross).to_csv(OUT/'cross_model_sign_agreement.csv',index=False,float_format='%.12g')
summaries={m:json.loads((E/m/'summary.json').read_text()) for m in ['M1','M2']}
primary=met[(met.horizon_min.eq(15))&(met.stratum.eq('all'))].to_dict(orient='records')
summary={'experiment':'EXP2.1C','status':'PASS_OBSERVATIONAL_MODEL_ALIGNMENT','realized_action_validation':'HOLD','primary_cohort_n':61,'day_n':17,'night_n':44,'primary_horizon_min':15,'models':summaries,'primary_metrics':primary,'input_sha256':sha256(IN),'interpretation':'agreement with confounding-reduced matched observational contrasts under a standardized stronger-ventilation probe; not causal or realized-actuator validation'}
summary['gate_pass']=bool(all(v.get('gate_pass') for v in summaries.values()) and len(primary)==4)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('EXP2.1C aggregate gate failed')
