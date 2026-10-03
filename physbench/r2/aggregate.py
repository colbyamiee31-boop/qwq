import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
GEN=ROOT/'physbench/r2/generated'; OUT=ROOT/'evidence/R2/FINAL'; OUT.mkdir(parents=True,exist_ok=True)
M1=pd.read_csv(ROOT/'evidence/R2/M1/dose_arms.csv'); M2=pd.read_csv(ROOT/'evidence/R2/M2/dose_arms.csv')
P=pd.read_csv(GEN/'PRIMARY_97_R2_INPUT.csv'); S=pd.read_csv(GEN/'STRICT_36_R2_INPUT.csv')
SEED=20261004; B=2000
def contrasts(d,model):
    piv=d.pivot(index=['cohort','event_id','event_date'],columns='arm',values=['DV','DN','action']).reset_index()
    piv.columns=['_'.join([str(x) for x in c if x!='']).rstrip('_') if isinstance(c,tuple) else c for c in piv.columns]
    piv['model']=model; piv['delta_DV']=piv['DV_post']-piv['DV_pre']; piv['delta_DN']=piv['DN_post']-piv['DN_pre']; piv['delta_U']=piv['action_post']-piv['action_pre']
    return piv
C=pd.concat([contrasts(M1,'M1'),contrasts(M2,'M2')],ignore_index=True)
def source(cohort):
    x=P.copy() if cohort=='primary' else S.copy()
    return x
def build(cohort,maskcol,anchor='23_23'):
    x=source(cohort); x=x[x[maskcol].astype(bool)].copy()
    z=C[C.cohort==cohort].merge(x,on=['event_id','event_date'],how='inner',validate='many_to_one')
    z['emp_DV']=z[f'emp_delta_DV_{anchor}']; z['emp_DN']=z[f'emp_delta_DN_{anchor}']
    return z
def metric(z,coord,model='pooled'):
    q=z if model=='pooled' else z[z.model==model]
    emp=q[f'emp_{coord}'].to_numpy(float); mod=q[f'delta_{coord}'].to_numpy(float)
    se=np.sign(emp); sm=np.sign(mod); nz=(np.abs(emp)>1e-12)&(np.abs(mod)>1e-12)
    same=nz&(se==sm)
    conc=float(np.mean(se[nz]==sm[nz])) if nz.any() else np.nan
    rho=float(spearmanr(emp,mod).statistic) if len(q)>2 else np.nan
    logdist=float(np.median(np.abs(np.log(np.abs(mod[same])/np.abs(emp[same]))))) if same.any() else np.nan
    slope=float(np.sum(emp*mod)/np.sum(emp*emp)) if np.sum(emp*emp)>0 else np.nan
    den=float(np.median(np.abs(emp)))
    nae=float(np.median(np.abs(mod-emp))/den) if den>0 else np.nan
    return {'coordinate':coord,'model':model,'n_rows':len(q),'n_events':q.event_id.nunique(),'sign_concordance':conc,'spearman':rho,'median_abs_log_ratio':logdist,'through_origin_slope':slope,'normalized_abs_error':nae,'same_sign_nonzero_n':int(same.sum())}
def native_metric(x):
    emp=x.emp_delta_DV_23_23.to_numpy(float); u=x.delta_u.to_numpy(float)
    nz=(np.abs(emp)>1e-12)&(np.abs(u)>1e-12)
    return {'coordinate':'U_native','model':'common','n_rows':len(x),'n_events':x.event_id.nunique(),'sign_concordance':float(np.mean(np.sign(emp[nz])==np.sign(u[nz]))),'spearman':float(spearmanr(emp,u).statistic),'median_abs_log_ratio':np.nan,'through_origin_slope':np.nan,'normalized_abs_error':np.nan,'same_sign_nonzero_n':int(nz.sum())}
def eval_case(name,cohort,mask,anchor):
    z=build(cohort,mask,anchor)
    rows=[]
    for c in ['DV','DN']:
        for m in ['M1','M2','pooled']: rows.append({'case':name,**metric(z,c,m)})
    x=source(cohort); x=x[x[mask].astype(bool)].copy()
    nr=native_metric(x); nr['case']=name; rows.append(nr)
    return z,rows
cases=[
 ('primary_3_6_23_23','primary','anchor_primary_3_6','23_23'),
 ('expanded_2_8_23_23','primary','anchor_expanded_2_8','23_23'),
 ('strict_3_6_23_23','strict','anchor_primary_3_6','23_23'),
 ('primary_3_6_22_24','primary','anchor_primary_3_6','22_24'),
 ('primary_3_6_24_22','primary','anchor_primary_3_6','24_22')]
allrows=[]; zs={}
for args in cases:
    z,r=eval_case(*args); zs[args[0]]=z; allrows.extend(r)
metrics=pd.DataFrame(allrows); metrics.to_csv(OUT/'coordinate_metrics.csv',index=False,float_format='%.12g')
# Event-level primary table
zp=zs['primary_3_6_23_23'].copy()
zp[['cohort','event_id','event_date','model','delta_U','delta_DV','delta_DN','emp_DV','emp_DN','event_Windsp','Wind direction']].to_csv(OUT/'primary_event_coordinate_contrasts.csv',index=False,float_format='%.12g')
# Date-cluster bootstrap for pooled DN-DV distortion difference
rng=np.random.default_rng(SEED); dates=np.array(sorted(zp.event_date.unique()))
boots=[]
def pooled_dist(q,c):
    r=metric(q,c,'pooled'); return r['median_abs_log_ratio'],r['normalized_abs_error']
for b in range(B):
    samp=rng.choice(dates,size=len(dates),replace=True)
    pieces=[]
    for k,d in enumerate(samp):
        a=zp[zp.event_date==d].copy(); a['_rep']=k; pieces.append(a)
    q=pd.concat(pieces,ignore_index=True)
    ldv,adv=pooled_dist(q,'DV'); ldn,adn=pooled_dist(q,'DN')
    boots.append((ldv,ldn,ldn-ldv,adv,adn,adn-adv))
bd=pd.DataFrame(boots,columns=['DV_logdist','DN_logdist','DN_minus_DV_logdist','DV_NAE','DN_NAE','DN_minus_DV_NAE'])
bd.to_csv(OUT/'date_cluster_bootstrap.csv',index=False,float_format='%.12g')
def ci(col):
    a=bd[col].dropna().to_numpy(float); return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
prim=metrics[(metrics.case=='primary_3_6_23_23')&(metrics.model=='pooled')]
dv=prim[prim.coordinate=='DV'].iloc[0]; dn=prim[prim.coordinate=='DN'].iloc[0]
ldiff=float(dn.median_abs_log_ratio-dv.median_abs_log_ratio); adiff=float(dn.normalized_abs_error-dv.normalized_abs_error)
lci=ci('DN_minus_DV_logdist'); aci=ci('DN_minus_DV_NAE')
if lci[0]>0 and aci[0]>0: verdict='DV_LESS_DISTORTED'
elif lci[1]<0 and aci[1]<0: verdict='DN_LESS_DISTORTED'
else: verdict='UNRESOLVED_OR_MIXED'
# summaries
m1s=json.loads((ROOT/'evidence/R2/M1/summary.json').read_text()); m2s=json.loads((ROOT/'evidence/R2/M2/summary.json').read_text())
summary={
 'experiment':'PhysBench-GH R2 empirically anchored action coordinate',
 'primary_events':int(zp.event_id.nunique()),'primary_rows_pooled':int(len(zp)),
 'M1_H_eff_m':m1s['H_eff_m'],'M2_H_eff_m':m2s['H_eff_m'],'CTIFL_H_eff_m':7.375,
 'primary_pooled_DV':{k:float(dv[k]) for k in ['sign_concordance','spearman','median_abs_log_ratio','through_origin_slope','normalized_abs_error']},
 'primary_pooled_DN':{k:float(dn[k]) for k in ['sign_concordance','spearman','median_abs_log_ratio','through_origin_slope','normalized_abs_error']},
 'DN_minus_DV_logdist':ldiff,'DN_minus_DV_logdist_CI95':lci,
 'DN_minus_DV_NAE':adiff,'DN_minus_DV_NAE_CI95':aci,
 'coordinate_verdict':verdict,
 'bootstrap_reps':B,'bootstrap_seed':SEED
}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
# Fail closed only execution/data integrity, not verdict
if not (len(zp)==80 and zp.event_id.nunique()==40 and np.isfinite(zp[['delta_DV','delta_DN','emp_DV','emp_DN']].to_numpy(float)).all()):
    raise SystemExit('R2 aggregate integrity gate failed')
