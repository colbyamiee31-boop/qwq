import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1DIR=ROOT/'evidence/EXP3_4B_M1'
M2DIR=ROOT/'evidence/EXP3_4B_M2'
OLD=ROOT/'evidence/EXP3_1_LOCKED_FINAL'
OUT=ROOT/'evidence/EXP3_4B_FINAL'; OUT.mkdir(parents=True,exist_ok=True)

Q=np.array([0.0,0.25,0.5,0.75,1.0],dtype=float)
COST=Q**2
ANCHORS=[0.0,0.05,0.10,0.25,0.50,1.00]
SEED=20261002; NBOOT=2000
METRICS=[
 'decision_disagreement_measure',
 'normalized_action_gap_integral',
 'symmetric_cross_model_regret_integral',
 'symmetric_outcome_divergence_integral'
]

m1=pd.read_csv(M1DIR/'matched_dose_responses.csv')
m2=pd.read_csv(M2DIR/'matched_dose_responses.csv')
s1=json.loads((M1DIR/'summary.json').read_text())
s2=json.loads((M2DIR/'summary.json').read_text())
old=pd.read_csv(OLD/'event_decision_consequence.csv')
assert s1['gate_pass'] and s2['gate_pass']
assert s1['target_sha256']==s2['target_sha256']=='e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292'
assert len(m1)==len(m2)==1060
REQ={'model_id','event_id','team','event_date','daynight','horizon_min','coordinate','q',
     'target_dose','achieved_dose','dose_error','dose_tolerance','native_command',
     'T','AH','T_gradient_C','AH_gradient_g_m3'}
for name,df in [('M1',m1),('M2',m2)]:
    miss=REQ-set(df.columns); assert not miss,(name,miss)
    assert not df.duplicated(['event_id','horizon_min','coordinate','q']).any()
    assert df.groupby(['event_id','horizon_min','coordinate']).size().eq(5).all()
    assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.all(np.abs(df.dose_error.to_numpy(float))<=df.dose_tolerance.to_numpy(float)+1e-15)
    assert np.all((df.native_command>=0.1-1e-8)&(df.native_command<=0.9+1e-8))

# Exact cohort gates.
expected={('DV',15):42,('DN',15):61,('DV',30):48,('DN',30):61}
for key,n in expected.items():
    c,h=key
    ids=set(m1[(m1.coordinate==c)&(m1.horizon_min==h)].event_id.astype(int))
    ids2=set(m2[(m2.coordinate==c)&(m2.horizon_min==h)].event_id.astype(int))
    assert ids==ids2 and len(ids)==n,(key,len(ids),len(ids2))
assert {27,86,89}.issubset(set(m1[(m1.coordinate=='DV')&(m1.horizon_min==15)].event_id.astype(int)))

# Cross-model target/achievement audit.
match=m1.merge(
    m2,on=['event_id','horizon_min','coordinate','q','target_dose'],
    suffixes=('_M1','_M2'),validate='one_to_one'
)
assert len(match)==1060
match['cross_model_achieved_dose_abs_gap']=np.abs(match.achieved_dose_M1-match.achieved_dose_M2)
match['combined_tolerance']=match.dose_tolerance_M1+match.dose_tolerance_M2
match['cross_model_gap_fraction_of_combined_tol']=(
    match.cross_model_achieved_dose_abs_gap/match.combined_tolerance
)
assert np.all(match.cross_model_achieved_dose_abs_gap<=match.combined_tolerance+1e-15)
match.to_csv(OUT/'matched_dose_cross_model_audit.csv',index=False,float_format='%.12g')

def benefit_table(df,event_id,horizon,coordinate):
    x=df[(df.event_id==event_id)&(df.horizon_min==horizon)&(df.coordinate==coordinate)].sort_values('q')
    assert len(x)==5 and np.allclose(x.q.to_numpy(float),Q,rtol=0,atol=1e-12)
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0])
    assert abs(gt)>0 and abs(ga)>0
    T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt)
    bA=-np.sign(ga)*(AH-AH[0])/abs(ga)
    B=0.5*(bT+bA)
    return {
      'q':Q.copy(),'T':T,'AH':AH,'B':B,'bT':bT,'bAH':bA,'GT':gt,'GAH':ga,
      'event_date':x.event_date.iloc[0],'daynight':x.daynight.iloc[0],'team':x.team.iloc[0],
      'target_dose':x.target_dose.to_numpy(float),'native_command':x.native_command.to_numpy(float)
    }

def breakpoints(B,lam_max):
    vals=[0.0,float(lam_max)]
    for i in range(len(Q)):
        for j in range(i+1,len(Q)):
            dc=COST[i]-COST[j]
            if abs(dc)<1e-15: continue
            lam=(B[i]-B[j])/dc
            if 0.0<lam<lam_max: vals.append(float(lam))
    vals=sorted(vals); out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,lam):
    util=B-lam*COST
    mx=np.max(util)
    idx=np.where(util>=mx-1e-12)[0]
    return int(idx[0])

def norm_regret(B,chosen_idx,opt_idx,lam):
    util=B-lam*COST
    rg=float(util[opt_idx]-util[chosen_idx])
    span=float(np.max(util)-np.min(util))
    return 0.0 if span<=1e-15 else max(0.0,rg/span)

def event_metrics(event_id,horizon,coordinate,lam_max):
    a=benefit_table(m1,event_id,horizon,coordinate)
    b=benefit_table(m2,event_id,horizon,coordinate)
    # Frozen targets must be identical by construction.
    assert np.allclose(a['target_dose'],b['target_dose'],rtol=0,atol=1e-9)
    bp=sorted(set(breakpoints(a['B'],lam_max)+breakpoints(b['B'],lam_max)))
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)
    cover=disagree=gap=sreg=odiv=0.0; segrows=[]
    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=0.5*(lo+hi)
        i1=choose(a['B'],mid); i2=choose(b['B'],mid)
        cover+=width
        dis=i1!=i2
        if dis: disagree+=width
        gap+=width*abs(Q[i1]-Q[i2])
        r21=norm_regret(b['B'],i1,i2,mid)
        r12=norm_regret(a['B'],i2,i1,mid)
        sreg+=width*0.5*(r21+r12)
        d1=0.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT']) + abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=0.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT']) + abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        odiv+=width*0.5*(d1+d2)
        segrows.append({
          'event_id':event_id,'horizon_min':horizon,'coordinate':coordinate,'lambda_max':lam_max,
          'lambda_lo':lo,'lambda_hi':hi,'m1_q':Q[i1],'m2_q':Q[i2],
          'm1_target_dose':a['target_dose'][i1],'m2_target_dose':b['target_dose'][i2],
          'm1_native_command':a['native_command'][i1],'m2_native_command':b['native_command'][i2],
          'disagree':dis,'symmetric_normalized_regret_mid':0.5*(r21+r12),
          'symmetric_outcome_divergence_mid':0.5*(d1+d2)
        })
    if abs(cover-lam_max)>1e-10: raise AssertionError((event_id,horizon,coordinate,lam_max,cover))
    return {
      'event_id':event_id,'team':a['team'],'event_date':a['event_date'],'daynight':a['daynight'],
      'horizon_min':horizon,'coordinate':coordinate,'lambda_max':lam_max,
      'decision_disagreement_measure':disagree/lam_max,
      'normalized_action_gap_integral':gap/lam_max,
      'symmetric_cross_model_regret_integral':sreg/lam_max,
      'symmetric_outcome_divergence_integral':odiv/lam_max,
      'any_disagreement':bool(disagree>1e-12)
    },segrows

event_rows=[]; seg_rows=[]
for coordinate,horizon in [('DV',15),('DN',15),('DV',30),('DN',30)]:
    ids=sorted(set(m1[(m1.coordinate==coordinate)&(m1.horizon_min==horizon)].event_id.astype(int)))
    for lm in (1.0,3.0):
        for eid in ids:
            r,s=event_metrics(eid,horizon,coordinate,lm)
            event_rows.append(r); seg_rows.extend(s)
ev=pd.DataFrame(event_rows); seg=pd.DataFrame(seg_rows)
ev.to_csv(OUT/'event_matched_decision_consequence.csv',index=False,float_format='%.12g')
seg.to_csv(OUT/'matched_decision_partitions.csv',index=False,float_format='%.12g')

# Lambda anchor decisions.
arows=[]
for coordinate,horizon in [('DV',15),('DN',15),('DV',30),('DN',30)]:
    ids=sorted(set(m1[(m1.coordinate==coordinate)&(m1.horizon_min==horizon)].event_id.astype(int)))
    for eid in ids:
        a=benefit_table(m1,eid,horizon,coordinate); b=benefit_table(m2,eid,horizon,coordinate)
        for lam in ANCHORS:
            i1=choose(a['B'],lam); i2=choose(b['B'],lam)
            arows.append({
              'event_id':eid,'event_date':a['event_date'],'daynight':a['daynight'],
              'horizon_min':horizon,'coordinate':coordinate,'lambda':lam,
              'm1_q':Q[i1],'m2_q':Q[i2],'disagree':i1!=i2,'q_gap':abs(Q[i1]-Q[i2]),
              'm1_target_dose':a['target_dose'][i1],'m2_target_dose':b['target_dose'][i2],
              'm1_native_command':a['native_command'][i1],'m2_native_command':b['native_command'][i2]
            })
anchors=pd.DataFrame(arows)
anchors.to_csv(OUT/'lambda_anchor_matched_decisions.csv',index=False,float_format='%.12g')

def cluster_boot_ci(df,col):
    dates=np.array(sorted(df.event_date.unique()))
    by={d:df[df.event_date==d][col].to_numpy(float) for d in dates}
    rng=np.random.default_rng(SEED); vals=np.empty(NBOOT)
    for k in range(NBOOT):
        ds=rng.choice(dates,size=len(dates),replace=True)
        vals[k]=np.mean(np.concatenate([by[d] for d in ds]))
    return np.quantile(vals,[0.025,0.975])

summary_rows=[]
for coordinate,horizon in [('DV',15),('DN',15),('DV',30),('DN',30)]:
  for lm in (1.0,3.0):
    for stratum in ('all','day','night'):
      x=ev[(ev.coordinate==coordinate)&(ev.horizon_min==horizon)&(ev.lambda_max==lm)]
      if stratum!='all': x=x[x.daynight==stratum]
      for col in METRICS:
        lo,hi=cluster_boot_ci(x,col)
        summary_rows.append({
          'coordinate':coordinate,'horizon_min':horizon,'lambda_max':lm,'stratum':stratum,
          'metric':col,'n_events':len(x),'mean':float(x[col].mean()),'median':float(x[col].median()),
          'ci_low':float(lo),'ci_high':float(hi),
          'fraction_any_disagreement':float(x.any_disagreement.mean())
        })
summary=pd.DataFrame(summary_rows)
summary.to_csv(OUT/'matched_decision_consequence_summary.csv',index=False,float_format='%.12g')

# Benefit audit.
brows=[]
for model,df in [('M1',m1),('M2',m2)]:
  for coordinate,horizon in [('DV',15),('DN',15),('DV',30),('DN',30)]:
    ids=sorted(set(df[(df.coordinate==coordinate)&(df.horizon_min==horizon)].event_id.astype(int)))
    for eid in ids:
      z=benefit_table(df,eid,horizon,coordinate)
      for j,q in enumerate(Q):
        brows.append({
          'model':model,'event_id':eid,'horizon_min':horizon,'coordinate':coordinate,'q':q,
          'target_dose':z['target_dose'][j],'native_command':z['native_command'][j],
          'benefit_T':z['bT'][j],'benefit_AH':z['bAH'][j],
          'benefit_equal_weight':z['B'][j],'cost_q_squared':COST[j]
        })
pd.DataFrame(brows).to_csv(OUT/'matched_action_benefit_audit.csv',index=False,float_format='%.12g')

# Paired comparison against locked EXP3.1 on the SAME cohort.
assert {'event_id','horizon_min','lambda_max',*METRICS}.issubset(old.columns)
paired=[]
paired_summary=[]
for coordinate,horizon in [('DV',15),('DN',15),('DV',30),('DN',30)]:
  for lm in (1.0,3.0):
    new=ev[(ev.coordinate==coordinate)&(ev.horizon_min==horizon)&(ev.lambda_max==lm)].copy()
    oldx=old[(old.horizon_min==horizon)&(old.lambda_max==lm)].copy()
    z=new.merge(oldx[['event_id','event_date',*METRICS,'any_disagreement']],
                on='event_id',suffixes=('_matched','_old'),validate='one_to_one')
    assert len(z)==len(new)
    for col in METRICS:
        z[f'delta_{col}']=z[f'{col}_matched']-z[f'{col}_old']
    z['old_any_disagreement']=z['any_disagreement_old'].astype(bool)
    z['matched_any_disagreement']=z['any_disagreement_matched'].astype(bool)
    z['disagreement_survives']=z.old_any_disagreement & z.matched_any_disagreement
    z['coordinate']=coordinate; z['horizon_min']=horizon; z['lambda_max']=lm
    paired.append(z)
    for col in METRICS:
        delta=f'delta_{col}'
        lo,hi=cluster_boot_ci(z.rename(columns={'event_date_matched':'event_date'}),delta)
        paired_summary.append({
          'coordinate':coordinate,'horizon_min':horizon,'lambda_max':lm,'metric':col,
          'n_events':len(z),
          'old_mean_same_cohort':float(z[f'{col}_old'].mean()),
          'matched_mean':float(z[f'{col}_matched'].mean()),
          'mean_delta_matched_minus_old':float(z[delta].mean()),
          'median_delta_matched_minus_old':float(z[delta].median()),
          'delta_ci_low':float(lo),'delta_ci_high':float(hi),
          'old_fraction_any_disagreement':float(z.old_any_disagreement.mean()),
          'matched_fraction_any_disagreement':float(z.matched_any_disagreement.mean())
        })
paired_df=pd.concat(paired,ignore_index=True)
paired_df.to_csv(OUT/'paired_EXP3_1_vs_EXP3_4B_same_cohort.csv',index=False,float_format='%.12g')
paired_summary=pd.DataFrame(paired_summary)
paired_summary.to_csv(OUT/'paired_comparison_summary.csv',index=False,float_format='%.12g')

# Locked event 27/86/89 trace on primary DV15.
locked_ids=[27,86,89]
trace=match[(match.coordinate=='DV')&(match.horizon_min==15)&(match.event_id.isin(locked_ids))].copy()
trace.to_csv(OUT/'locked_events_27_86_89_matched_grid.csv',index=False,float_format='%.12g')
locked_metrics=ev[(ev.coordinate=='DV')&(ev.horizon_min==15)&(ev.lambda_max==1.0)&(ev.event_id.isin(locked_ids))].copy()
assert set(locked_metrics.event_id.astype(int))==set(locked_ids)
locked_metrics.to_csv(OUT/'locked_events_27_86_89_decision_metrics.csv',index=False,float_format='%.12g')
locked_parts=seg[(seg.coordinate=='DV')&(seg.horizon_min==15)&(seg.lambda_max==1.0)&(seg.event_id.isin(locked_ids))].copy()
locked_parts.to_csv(OUT/'locked_events_27_86_89_decision_partitions.csv',index=False,float_format='%.12g')

primary=summary[(summary.coordinate=='DV')&(summary.horizon_min==15)&(summary.lambda_max==1.0)&(summary.stratum=='all')]
primary_pair=paired_summary[(paired_summary.coordinate=='DV')&(paired_summary.horizon_min==15)&(paired_summary.lambda_max==1.0)]
dn61=summary[(summary.coordinate=='DN')&(summary.horizon_min==15)&(summary.lambda_max==1.0)&(summary.stratum=='all')]
result={
 'experiment':'PhysBench-GH EXP3.4B — Physical-Dose-Matched Decision Consequence',
 'status':'PASS_EXP3_4B',
 'target_sha256':s1['target_sha256'],
 'primary':{'coordinate':'DV','horizon_min':15,'lambda_domain':[0,1],'events':42,
            'summary':primary.to_dict(orient='records')},
 'secondary_DN_15':{'events':61,'summary':dn61.to_dict(orient='records')},
 'paired_primary_same_42':primary_pair.to_dict(orient='records'),
 'locked_events_27_86_89':locked_metrics[
     ['event_id','decision_disagreement_measure','normalized_action_gap_integral',
      'symmetric_cross_model_regret_integral','symmetric_outcome_divergence_integral','any_disagreement']
 ].to_dict(orient='records'),
 'dose_matching':{
    'rows_per_model':1060,
    'max_cross_model_achieved_dose_gap':float(match.cross_model_achieved_dose_abs_gap.max()),
    'max_cross_model_gap_fraction_of_combined_tolerance':float(match.cross_model_gap_fraction_of_combined_tol.max()),
    'M1_summary':s1,'M2_summary':s2
 }
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
