import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1DIR=ROOT/'evidence/M1'
M2DIR=ROOT/'evidence/M2'
OUT=ROOT/'evidence/FINAL'; OUT.mkdir(parents=True,exist_ok=True)
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
COST=((ACTIONS-0.1)/0.8)**2
ANCHORS=[0.0,0.05,0.10,0.25,0.50,1.00]
SEED=20261002; NBOOT=2000

m1=pd.read_csv(M1DIR/'action_grid_responses.csv')
m2=pd.read_csv(M2DIR/'action_grid_responses.csv')
s1=json.loads((M1DIR/'summary.json').read_text())
s2=json.loads((M2DIR/'summary.json').read_text())
assert s1['gate_pass'] and s2['gate_pass']
assert s1['actions']==s2['actions']==ACTIONS.tolist()
assert s1['input_sha256']==s2['input_sha256']
assert set(m1.event_id)==set(m2.event_id) and len(set(m1.event_id))==61

def benefit_table(df,event_id,horizon):
    x=df[(df.event_id==event_id)&(df.horizon_min==horizon)].sort_values('action')
    assert np.allclose(x.action.to_numpy(float),ACTIONS,rtol=0,atol=1e-12)
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0])
    assert abs(gt)>0 and abs(ga)>0
    T=x.T.to_numpy(float); AH=x.AH.to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt)
    bA=-np.sign(ga)*(AH-AH[0])/abs(ga)
    B=0.5*(bT+bA)
    return {'actions':ACTIONS.copy(),'T':T,'AH':AH,'B':B,'bT':bT,'bAH':bA,'GT':gt,'GAH':ga,
            'event_date':x.event_date.iloc[0],'daynight':x.daynight.iloc[0],'team':x.team.iloc[0]}

def breakpoints(B,lam_max):
    vals=[0.0,float(lam_max)]
    for i in range(len(ACTIONS)):
        for j in range(i+1,len(ACTIONS)):
            dc=COST[i]-COST[j]
            if abs(dc)<1e-15: continue
            lam=(B[i]-B[j])/dc
            if 0.0 < lam < lam_max:
                vals.append(float(lam))
    vals=sorted(vals)
    out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,lam):
    q=B-lam*COST
    mx=np.max(q)
    idx=np.where(q>=mx-1e-12)[0]
    return int(idx[0])

def norm_regret(B,chosen_idx,opt_idx,lam):
    q=B-lam*COST
    rg=float(q[opt_idx]-q[chosen_idx])
    span=float(np.max(q)-np.min(q))
    if span<=1e-15: return 0.0
    return max(0.0,rg/span)

def event_metrics(event_id,horizon,lam_max):
    a=benefit_table(m1,event_id,horizon); b=benefit_table(m2,event_id,horizon)
    bp=sorted(set(breakpoints(a['B'],lam_max)+breakpoints(b['B'],lam_max)))
    # deduplicate
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)
    cover=0.0; disagree=0.0; gap=0.0; sreg=0.0; odiv=0.0
    segrows=[]
    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=(lo+hi)/2
        i1=choose(a['B'],mid); i2=choose(b['B'],mid)
        cover+=width
        dis=(i1!=i2)
        if dis: disagree+=width
        gap += width*abs(ACTIONS[i1]-ACTIONS[i2])/0.8
        r21=norm_regret(b['B'],i1,i2,mid)  # M1 action under M2 evaluator
        r12=norm_regret(a['B'],i2,i1,mid)  # M2 action under M1 evaluator
        sreg += width*0.5*(r21+r12)
        # normalized outcome difference induced by the two selected actions, under both evaluators
        d1=0.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT']) + abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=0.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT']) + abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        odiv += width*0.5*(d1+d2)
        segrows.append({'event_id':event_id,'horizon_min':horizon,'lambda_max':lam_max,'lambda_lo':lo,'lambda_hi':hi,
                        'm1_action':ACTIONS[i1],'m2_action':ACTIONS[i2],'disagree':dis,
                        'symmetric_normalized_regret_mid':0.5*(r21+r12),'symmetric_outcome_divergence_mid':0.5*(d1+d2)})
    if abs(cover-lam_max)>1e-10: raise AssertionError((event_id,horizon,lam_max,cover))
    return {
      'event_id':event_id,'team':a['team'],'event_date':a['event_date'],'daynight':a['daynight'],
      'horizon_min':horizon,'lambda_max':lam_max,
      'decision_disagreement_measure':disagree/lam_max,
      'normalized_action_gap_integral':gap/lam_max,
      'symmetric_cross_model_regret_integral':sreg/lam_max,
      'symmetric_outcome_divergence_integral':odiv/lam_max,
      'any_disagreement':bool(disagree>1e-12)
    },segrows

event_rows=[]; seg_rows=[]
for h in (15,30):
    for lm in (1.0,3.0):
        for eid in sorted(set(m1.event_id)):
            r,s=event_metrics(eid,h,lm); event_rows.append(r); seg_rows.extend(s)
ev=pd.DataFrame(event_rows)
seg=pd.DataFrame(seg_rows)
ev.to_csv(OUT/'event_decision_consequence.csv',index=False,float_format='%.12g')
seg.to_csv(OUT/'decision_partitions.csv',index=False,float_format='%.12g')

# Lambda anchor disagreement table
arows=[]
for h in (15,30):
    for eid in sorted(set(m1.event_id)):
        a=benefit_table(m1,eid,h); b=benefit_table(m2,eid,h)
        for lam in ANCHORS:
            i1=choose(a['B'],lam); i2=choose(b['B'],lam)
            arows.append({'event_id':eid,'event_date':a['event_date'],'daynight':a['daynight'],'horizon_min':h,'lambda':lam,
                          'm1_action':ACTIONS[i1],'m2_action':ACTIONS[i2],'disagree':i1!=i2,
                          'action_gap':abs(ACTIONS[i1]-ACTIONS[i2])})
anchors=pd.DataFrame(arows)
anchors.to_csv(OUT/'lambda_anchor_decisions.csv',index=False,float_format='%.12g')

def cluster_boot_ci(df,col):
    dates=np.array(sorted(df.event_date.unique()))
    by={d:df[df.event_date==d][col].to_numpy(float) for d in dates}
    rng=np.random.default_rng(SEED)
    vals=np.empty(NBOOT)
    for k in range(NBOOT):
        ds=rng.choice(dates,size=len(dates),replace=True)
        x=np.concatenate([by[d] for d in ds])
        vals[k]=np.mean(x)
    return np.quantile(vals,[0.025,0.975])

summary_rows=[]
for h in (15,30):
  for lm in (1.0,3.0):
    for stratum in ('all','day','night'):
      x=ev[(ev.horizon_min==h)&(ev.lambda_max==lm)]
      if stratum!='all': x=x[x.daynight==stratum]
      for col in ['decision_disagreement_measure','normalized_action_gap_integral','symmetric_cross_model_regret_integral','symmetric_outcome_divergence_integral']:
        lo,hi=cluster_boot_ci(x,col)
        summary_rows.append({'horizon_min':h,'lambda_max':lm,'stratum':stratum,'metric':col,'n_events':len(x),
                             'mean':float(x[col].mean()),'median':float(x[col].median()),
                             'ci_low':float(lo),'ci_high':float(hi),
                             'fraction_any_disagreement':float(x.any_disagreement.mean())})
summary=pd.DataFrame(summary_rows)
summary.to_csv(OUT/'decision_consequence_summary.csv',index=False,float_format='%.12g')

anchor_summary=anchors.groupby(['horizon_min','lambda'],as_index=False).agg(
    n_events=('event_id','count'),
    disagreement_fraction=('disagree','mean'),
    mean_action_gap=('action_gap','mean'),
    m1_mean_action=('m1_action','mean'),
    m2_mean_action=('m2_action','mean')
)
anchor_summary.to_csv(OUT/'lambda_anchor_summary.csv',index=False,float_format='%.12g')

# model benefit tables for audit
brows=[]
for model,df in [('M1',m1),('M2',m2)]:
  for h in (15,30):
    for eid in sorted(set(df.event_id)):
      z=benefit_table(df,eid,h)
      for j,u in enumerate(ACTIONS):
        brows.append({'model':model,'event_id':eid,'horizon_min':h,'action':u,'benefit_T':z['bT'][j],'benefit_AH':z['bAH'][j],'benefit_equal_weight':z['B'][j],'cost':COST[j]})
pd.DataFrame(brows).to_csv(OUT/'action_benefit_audit.csv',index=False,float_format='%.12g')

result={
 'experiment':'PhysBench-GH EXP3.1 Decision-Consequence Test',
 'status':'PASS_DECISION_CONSEQUENCE' if len(ev)==61*2*2 else 'FAIL',
 'primary_horizon_min':15,'primary_lambda_domain':[0,1],
 'events':61,'actions':ACTIONS.tolist(),
 'm1_summary':s1,'m2_summary':s2,
 'primary':summary[(summary.horizon_min==15)&(summary.lambda_max==1.0)&(summary.stratum=='all')].to_dict(orient='records')
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if result['status']!='PASS_DECISION_CONSEQUENCE': raise SystemExit('EXP3.1 aggregate failed')
