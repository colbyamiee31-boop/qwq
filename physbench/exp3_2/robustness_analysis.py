import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1DIR=ROOT/'evidence/EXP3_2_M1'
M2DIR=ROOT/'evidence/EXP3_2_M2'
OUT=ROOT/'evidence/EXP3_2_FINAL'; OUT.mkdir(parents=True,exist_ok=True)

GRIDS={
 'g3':np.array([0.1,0.5,0.9],float),
 'g5':np.array([0.1,0.3,0.5,0.7,0.9],float),
 'g9':np.round(np.arange(0.1,1.0,0.1),1)
}
WEIGHTS={
 'T75_AH25':(0.75,0.25),
 'equal':(0.50,0.50),
 'T25_AH75':(0.25,0.75)
}
COST_FORMS=('quadratic','linear')
HORIZONS=(15,30)
LAMBDA_MAXES=(1.0,3.0)
SEED=20261002; NBOOT=2000
TOL=1e-10

LOCKED_EXP31={
 'decision_disagreement_measure':{
   'mean':0.024552427907011506,'median':0.0,'ci_low':0.0,'ci_high':0.059165829930892304,
   'fraction_any_disagreement':0.04918032786885246},
 'normalized_action_gap_integral':{
   'mean':0.006138106976752877,'median':0.0,'ci_low':0.0,'ci_high':0.014791457482723078,
   'fraction_any_disagreement':0.04918032786885246},
 'symmetric_cross_model_regret_integral':{
   'mean':0.001645164794050128,'median':0.0,'ci_low':0.0,'ci_high':0.003910665621685459,
   'fraction_any_disagreement':0.04918032786885246},
 'symmetric_outcome_divergence_integral':{
   'mean':0.01691271675501554,'median':0.0,'ci_low':0.0,'ci_high':0.03967407752859917,
   'fraction_any_disagreement':0.04918032786885246}
}

m1=pd.read_csv(M1DIR/'action_grid_responses.csv')
m2=pd.read_csv(M2DIR/'action_grid_responses.csv')
s1=json.loads((M1DIR/'summary.json').read_text())
s2=json.loads((M2DIR/'summary.json').read_text())

assert s1['gate_pass'] and s2['gate_pass']
assert s1['actions']==s2['actions']==GRIDS['g9'].tolist()
assert s1['input_sha256']==s2['input_sha256']=='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
assert s1['frozen_commit']=='2d3febb1ea002b24b452e32293e990beb78d3ce1'
assert s2['frozen_commit']=='bea8c3b0a1324162a4b5487db578aa674c8b587c'
assert set(m1.event_id)==set(m2.event_id) and len(set(m1.event_id))==61

_REQUIRED_COLUMNS={
    'event_id','team','event_date','daynight','horizon_min','action',
    'T_gradient_C','AH_gradient_g_m3','T','AH'
}
for _name,_df in (('M1',m1),('M2',m2)):
    _missing=sorted(_REQUIRED_COLUMNS-set(_df.columns))
    assert not _missing, f'{_name} missing required columns: {_missing}'
    assert set(_df.horizon_min.to_numpy(int))=={15,30}
    assert not _df.duplicated(['event_id','horizon_min','action']).any()
    _counts=_df.groupby(['event_id','horizon_min']).size().to_numpy(int)
    assert len(_counts)==61*2 and np.all(_counts==9)
    for _col in ('action','T_gradient_C','AH_gradient_g_m3','T','AH'):
        _v=pd.to_numeric(_df[_col],errors='raise').to_numpy(float)
        assert np.all(np.isfinite(_v))

def cost(actions,form):
    z=(np.asarray(actions,float)-0.1)/0.8
    if form=='quadratic': return z*z
    if form=='linear': return z
    raise ValueError(form)

def benefit_table(df,event_id,horizon,actions,wT,wAH):
    x=df[(df.event_id==event_id)&(df.horizon_min==horizon)&(df.action.isin(actions))].sort_values('action')
    assert len(x)==len(actions)
    assert np.allclose(x['action'].to_numpy(float),actions,rtol=0,atol=1e-12)
    gt=float(x['T_gradient_C'].iloc[0]); ga=float(x['AH_gradient_g_m3'].iloc[0])
    assert abs(gt)>0 and abs(ga)>0
    T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt)
    bA=-np.sign(ga)*(AH-AH[0])/abs(ga)
    B=wT*bT+wAH*bA
    return {'actions':actions.copy(),'T':T,'AH':AH,'B':B,'bT':bT,'bAH':bA,'GT':gt,'GAH':ga,
            'event_date':x.event_date.iloc[0],'daynight':x.daynight.iloc[0],'team':x.team.iloc[0]}

def breakpoints(B,C,lam_max):
    vals=[0.0,float(lam_max)]
    for i in range(len(B)):
        for j in range(i+1,len(B)):
            dc=C[i]-C[j]
            if abs(dc)<1e-15: continue
            lam=(B[i]-B[j])/dc
            if 0.0 < lam < lam_max:
                vals.append(float(lam))
    vals=sorted(vals)
    out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,C,lam):
    q=B-lam*C
    mx=np.max(q)
    idx=np.where(q>=mx-1e-12)[0]
    return int(idx[0])

def norm_regret(B,C,chosen_idx,opt_idx,lam):
    q=B-lam*C
    rg=float(q[opt_idx]-q[chosen_idx])
    span=float(np.max(q)-np.min(q))
    if span<=1e-15: return 0.0
    return max(0.0,rg/span)

def event_metrics(event_id,horizon,lam_max,grid_label,weight_label,cost_form):
    actions=GRIDS[grid_label]
    wT,wAH=WEIGHTS[weight_label]
    C=cost(actions,cost_form)
    a=benefit_table(m1,event_id,horizon,actions,wT,wAH)
    b=benefit_table(m2,event_id,horizon,actions,wT,wAH)
    bp=sorted(set(breakpoints(a['B'],C,lam_max)+breakpoints(b['B'],C,lam_max)))
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)
    cover=0.0; disagree=0.0; gap=0.0; sreg=0.0; odiv=0.0
    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=(lo+hi)/2
        i1=choose(a['B'],C,mid); i2=choose(b['B'],C,mid)
        cover+=width
        if i1!=i2: disagree+=width
        gap += width*abs(actions[i1]-actions[i2])/0.8
        r21=norm_regret(b['B'],C,i1,i2,mid)
        r12=norm_regret(a['B'],C,i2,i1,mid)
        sreg += width*0.5*(r21+r12)
        d1=0.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT']) + abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=0.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT']) + abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        odiv += width*0.5*(d1+d2)
    if abs(cover-lam_max)>1e-10:
        raise AssertionError((event_id,horizon,lam_max,grid_label,weight_label,cost_form,cover))
    return {
      'event_id':event_id,'team':a['team'],'event_date':a['event_date'],'daynight':a['daynight'],
      'horizon_min':horizon,'lambda_max':lam_max,'grid':grid_label,'n_actions':len(actions),
      'weight':weight_label,'wT':wT,'wAH':wAH,'cost_form':cost_form,
      'decision_disagreement_measure':disagree/lam_max,
      'normalized_action_gap_integral':gap/lam_max,
      'symmetric_cross_model_regret_integral':sreg/lam_max,
      'symmetric_outcome_divergence_integral':odiv/lam_max,
      'any_disagreement':bool(disagree>1e-12)
    }

event_rows=[]
events=sorted(set(m1.event_id))
for h in HORIZONS:
  for lm in LAMBDA_MAXES:
    for grid_label in GRIDS:
      for weight_label in WEIGHTS:
        for cost_form in COST_FORMS:
          for eid in events:
            event_rows.append(event_metrics(eid,h,lm,grid_label,weight_label,cost_form))
ev=pd.DataFrame(event_rows)
ev.to_csv(OUT/'event_robustness_metrics.csv',index=False,float_format='%.12g')
assert len(ev)==61*2*2*3*3*2
assert not ev.duplicated(['event_id','horizon_min','lambda_max','grid','weight','cost_form']).any()

def cluster_boot_ci(df,col):
    g=df.groupby('event_date')[col].agg(['sum','count']).sort_index()
    sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(int)
    rng=np.random.default_rng(SEED)
    vals=np.empty(NBOOT); n=len(g)
    for k in range(NBOOT):
        ix=rng.integers(0,n,size=n)
        vals[k]=sums[ix].sum()/counts[ix].sum()
    return np.quantile(vals,[0.025,0.975])

metrics=['decision_disagreement_measure','normalized_action_gap_integral',
         'symmetric_cross_model_regret_integral','symmetric_outcome_divergence_integral']
summary_rows=[]
for h in HORIZONS:
  for lm in LAMBDA_MAXES:
    for grid_label in GRIDS:
      for weight_label in WEIGHTS:
        for cost_form in COST_FORMS:
          base=ev[(ev.horizon_min==h)&(ev.lambda_max==lm)&(ev.grid==grid_label)&
                  (ev.weight==weight_label)&(ev.cost_form==cost_form)]
          for stratum in ('all','day','night'):
            x=base if stratum=='all' else base[base.daynight==stratum]
            for col in metrics:
                lo,hi=cluster_boot_ci(x,col)
                summary_rows.append({
                  'horizon_min':h,'lambda_max':lm,'grid':grid_label,'n_actions':len(GRIDS[grid_label]),
                  'weight':weight_label,'wT':WEIGHTS[weight_label][0],'wAH':WEIGHTS[weight_label][1],
                  'cost_form':cost_form,'stratum':stratum,'metric':col,'n_events':len(x),
                  'mean':float(x[col].mean()),'median':float(x[col].median()),
                  'ci_low':float(lo),'ci_high':float(hi),
                  'fraction_any_disagreement':float(x.any_disagreement.mean())
                })
summary=pd.DataFrame(summary_rows)
summary.to_csv(OUT/'robustness_summary.csv',index=False,float_format='%.12g')
assert len(summary)==2*2*3*3*2*3*4

primary=ev[(ev.horizon_min==15)&(ev.lambda_max==1.0)].copy()
pers=primary.groupby(['event_id','team','event_date','daynight'],as_index=False).agg(
    specifications=('any_disagreement','size'),
    specifications_with_any_disagreement=('any_disagreement','sum'),
    disagreement_persistence=('any_disagreement','mean'),
    mean_disagreement_measure=('decision_disagreement_measure','mean'),
    max_disagreement_measure=('decision_disagreement_measure','max'),
    mean_action_gap_integral=('normalized_action_gap_integral','mean')
)
assert (pers.specifications==18).all()
pers.to_csv(OUT/'primary_event_persistence.csv',index=False,float_format='%.12g')

spec=primary.groupby(['grid','n_actions','weight','wT','wAH','cost_form'],as_index=False).agg(
    n_events=('event_id','count'),
    fraction_events_any_disagreement=('any_disagreement','mean'),
    mean_disagreement_measure=('decision_disagreement_measure','mean'),
    median_disagreement_measure=('decision_disagreement_measure','median'),
    mean_action_gap_integral=('normalized_action_gap_integral','mean'),
    mean_cross_model_regret_integral=('symmetric_cross_model_regret_integral','mean'),
    mean_outcome_divergence_integral=('symmetric_outcome_divergence_integral','mean')
)
assert len(spec)==18
spec.to_csv(OUT/'primary_specification_overview.csv',index=False,float_format='%.12g')

ref=summary[(summary.horizon_min==15)&(summary.lambda_max==1.0)&
            (summary.grid=='g5')&(summary.weight=='equal')&
            (summary.cost_form=='quadratic')&(summary.stratum=='all')]
assert len(ref)==4
refev=ev[(ev.horizon_min==15)&(ev.lambda_max==1.0)&
         (ev.grid=='g5')&(ev.weight=='equal')&(ev.cost_form=='quadratic')]
ref_ids=refev.loc[refev.any_disagreement,'event_id'].astype(int).tolist()

repro={'status':'PASS','tolerance':TOL,'locked_disagreement_event_ids':[27,86,89],
       'observed_disagreement_event_ids':ref_ids,'checks':{}}
if ref_ids != [27,86,89]:
    repro['status']='FAIL'
for metric,expected in LOCKED_EXP31.items():
    r=ref[ref.metric==metric].iloc[0]
    checks={}
    for field,val in expected.items():
        got=float(r[field])
        err=abs(got-val)
        checks[field]={'expected':val,'observed':got,'abs_error':err,'pass':bool(err<=TOL)}
        if err>TOL: repro['status']='FAIL'
    repro['checks'][metric]=checks
(OUT/'exp3_1_reference_reproduction.json').write_text(json.dumps(repro,indent=2),encoding='utf-8')
if repro['status']!='PASS':
    raise SystemExit('EXP3.1 reference reproduction failed')

overview={
 'experiment':'PhysBench-GH EXP3.2 Decision Robustness',
 'status':'PASS_DECISION_ROBUSTNESS' if len(ev)==4392 and len(summary)==864 and repro['status']=='PASS' else 'FAIL',
 'locked_exp3_1_commit':'8c68e2262f9cbaffdc1a14a2309e9fdf7644df11',
 'events':61,'primary_specifications':18,
 'grids':{k:v.tolist() for k,v in GRIDS.items()},
 'weights':{k:list(v) for k,v in WEIGHTS.items()},
 'cost_forms':list(COST_FORMS),'horizons_min':list(HORIZONS),'lambda_domains':[[0,1],[0,3]],
 'm1_summary':s1,'m2_summary':s2,
 'reference_reproduction_status':repro['status'],
 'primary_persistence':{
   'events_with_disagreement_in_at_least_one_spec':int((pers.specifications_with_any_disagreement>0).sum()),
   'events_with_disagreement_in_all_18_specs':int((pers.specifications_with_any_disagreement==18).sum()),
   'median_persistence':float(pers.disagreement_persistence.median()),
   'mean_persistence':float(pers.disagreement_persistence.mean())
 }
}
(OUT/'SUMMARY.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
print(json.dumps(overview,indent=2))
if overview['status']!='PASS_DECISION_ROBUSTNESS':
    raise SystemExit('EXP3.2 aggregate failed')
