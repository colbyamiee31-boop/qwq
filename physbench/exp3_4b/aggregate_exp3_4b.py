import json
from pathlib import Path
import numpy as np
import pandas as pd

from common import Q_LEVELS,TARGET_SHA256,INPUT_SHA256,dose_tolerance

ROOT=Path(__file__).resolve().parents[2]
M1D=ROOT/'evidence/EXP3_4B_M1'; M2D=ROOT/'evidence/EXP3_4B_M2'; OLD=ROOT/'evidence/EXP3_1_LOCKED'; OUT=ROOT/'evidence/EXP3_4B_FINAL'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261002; NBOOT=2000; TIE_TOL=1e-12
SPECS=[('primary_quadratic_L1','quadratic',1.0),('sensitivity_linear_L1','linear',1.0),('sensitivity_quadratic_L3','quadratic',3.0)]
LOCKED_EVENTS=[27,86,89]

m1=pd.read_csv(M1D/'matched_physical_dose_outcomes.csv'); m2=pd.read_csv(M2D/'matched_physical_dose_outcomes.csv')
s1=json.loads((M1D/'summary.json').read_text()); s2=json.loads((M2D/'summary.json').read_text())
assert s1['gate_pass'] and s2['gate_pass'] and s1['target_sha256']==s2['target_sha256']==TARGET_SHA256
assert s1['input_sha256']==s2['input_sha256']==INPUT_SHA256
REQ={'model_id','event_id','team','event_date','daynight','horizon_min','coordinate','q','native_command','target_dose','actual_dose',
     'dose_abs_error','dose_tolerance','dose_match_pass','T','AH','T_gradient_C','AH_gradient_g_m3'}
for name,df in [('M1',m1),('M2',m2)]:
    miss=REQ-set(df.columns); assert not miss,(name,miss)
    assert len(df)==515 and not df.duplicated(['coordinate','event_id','q']).any()
    assert bool(df.dose_match_pass.all()) and np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
    assert set(np.round(df.q.unique(),12))==set(Q_LEVELS)
    for _,r in df.iterrows(): assert float(r.dose_abs_error)<=dose_tolerance(str(r.coordinate),float(r.target_dose))+1e-15

# Cross-model target identity is mandatory; actual doses need only satisfy their frozen numerical tolerance.
keys=['coordinate','event_id','q']
z=m1.merge(m2,on=keys,suffixes=('_M1','_M2'),validate='one_to_one')
assert len(z)==515
assert np.max(np.abs(z.target_dose_M1-z.target_dose_M2))<=1e-12
assert (z.horizon_min_M1==15).all() and (z.horizon_min_M2==15).all()

# Dose matching and inversion audit.
audit=z[[*keys,'team_M1','event_date_M1','daynight_M1','target_dose_M1','native_command_M1','native_command_M2',
         'actual_dose_M1','actual_dose_M2','dose_abs_error_M1','dose_abs_error_M2','dose_tolerance_M1','dose_tolerance_M2',
         'inversion_evaluations_M1','inversion_evaluations_M2']].copy()
audit.columns=['coordinate','event_id','q','team','event_date','daynight','target_dose','native_command_M1','native_command_M2',
               'actual_dose_M1','actual_dose_M2','dose_abs_error_M1','dose_abs_error_M2','dose_tolerance_M1','dose_tolerance_M2',
               'inversion_evaluations_M1','inversion_evaluations_M2']
audit['cross_model_actual_dose_abs_difference']=np.abs(audit.actual_dose_M1-audit.actual_dose_M2)
audit.to_csv(OUT/'matched_dose_inversion_audit.csv',index=False,float_format='%.12g')

def cost(q,form):
    q=np.asarray(q,float)
    if form=='quadratic': return q*q
    if form=='linear': return q
    raise ValueError(form)

def table(df,eid,coord):
    x=df[(df.event_id==eid)&(df.coordinate==coord)].sort_values('q')
    assert len(x)==5 and np.allclose(x.q.to_numpy(float),Q_LEVELS,rtol=0,atol=1e-12)
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0]); assert abs(gt)>0 and abs(ga)>0
    T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt); bAH=-np.sign(ga)*(AH-AH[0])/abs(ga); B=.5*(bT+bAH)
    return {'T':T,'AH':AH,'bT':bT,'bAH':bAH,'B':B,'GT':gt,'GAH':ga,
            'cmd':x.native_command.to_numpy(float),'dose':x.target_dose.to_numpy(float),
            'team':x.team.iloc[0],'date':x.event_date.iloc[0],'daynight':x.daynight.iloc[0]}

def breakpoints(B,C,lam_max):
    vals=[0.0,float(lam_max)]
    for i in range(5):
        for j in range(i+1,5):
            dc=C[i]-C[j]
            if abs(dc)<1e-15: continue
            lam=(B[i]-B[j])/dc
            if 0<lam<lam_max: vals.append(float(lam))
    vals=sorted(vals); out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,C,lam):
    q=np.asarray(B)-lam*np.asarray(C); mx=q.max(); return int(np.where(q>=mx-TIE_TOL)[0][0])

def norm_regret(B,C,chosen,opt,lam):
    q=np.asarray(B)-lam*np.asarray(C); span=float(q.max()-q.min())
    if span<=1e-15: return 0.0
    return max(0.0,float(q[opt]-q[chosen])/span)

def negative_stats(B,C,lo,hi):
    # measure and integral area of max(0,-(B-lambda*C)) over [lo,hi]
    if hi<=lo: return 0.0,0.0
    f0=B-lo*C; f1=B-hi*C
    if f0>=0 and f1>=0: return 0.0,0.0
    if f0<0 and f1<0:
        area=-(B*(hi-lo)-0.5*C*(hi*hi-lo*lo)); return hi-lo,max(0.0,area)
    if abs(C)<=1e-15: return (hi-lo,max(0.0,-B*(hi-lo))) if B<0 else (0.0,0.0)
    root=B/C; root=min(max(root,lo),hi)
    if f0>=0 and f1<0:
        area=-(B*(hi-root)-0.5*C*(hi*hi-root*root)); return hi-root,max(0.0,area)
    area=-(B*(root-lo)-0.5*C*(root*root-lo*lo)); return root-lo,max(0.0,area)

def event_metrics(eid,coord,spec_name,cost_form,lam_max):
    a=table(m1,eid,coord); b=table(m2,eid,coord); C=cost(Q_LEVELS,cost_form)
    bp=sorted(set(breakpoints(a['B'],C,lam_max)+breakpoints(b['B'],C,lam_max)))
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)
    cover=dis=gap=qshift=nativegap=dosegap=r12=r21=odiv=bv12=bv21=bd12=bd21=0.0; segs=[]
    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=(lo+hi)/2; i1=choose(a['B'],C,mid); i2=choose(b['B'],C,mid); cover+=width
        diff=(i1!=i2); dis+=width*diff; gap+=width*abs(Q_LEVELS[i1]-Q_LEVELS[i2]); qshift+=width*(Q_LEVELS[i1]-Q_LEVELS[i2])
        nativegap+=width*abs(a['cmd'][i1]-b['cmd'][i2]); dosegap+=width*abs(a['dose'][i1]-a['dose'][i2])
        rr12=norm_regret(b['B'],C,i1,i2,mid); rr21=norm_regret(a['B'],C,i2,i1,mid); r12+=width*rr12; r21+=width*rr21
        d1=.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT'])+abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT'])+abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        odiv+=width*.5*(d1+d2)
        mm,aa=negative_stats(float(b['B'][i1]),float(C[i1]),lo,hi); bv12+=mm; bd12+=aa
        mm,aa=negative_stats(float(a['B'][i2]),float(C[i2]),lo,hi); bv21+=mm; bd21+=aa
        segs.append({'coordinate':coord,'event_id':eid,'specification':spec_name,'cost_form':cost_form,'lambda_max':lam_max,
                     'lambda_lo':lo,'lambda_hi':hi,'m1_q':Q_LEVELS[i1],'m2_q':Q_LEVELS[i2],
                     'm1_native_command':a['cmd'][i1],'m2_native_command':b['cmd'][i2],
                     'm1_target_dose':a['dose'][i1],'m2_target_dose':b['dose'][i2],'disagree':bool(diff)})
    if abs(cover-lam_max)>1e-10: raise AssertionError(('partition coverage',coord,eid,spec_name,cover))
    r12/=lam_max; r21/=lam_max
    return {'coordinate':coord,'event_id':eid,'team':a['team'],'event_date':a['date'],'daynight':a['daynight'],
            'specification':spec_name,'cost_form':cost_form,'lambda_max':lam_max,
            'decision_disagreement_measure':dis/lam_max,'any_disagreement':bool(dis>1e-12),
            'physical_q_action_gap_integral':gap/lam_max,'signed_q_shift_integral':qshift/lam_max,
            'native_command_gap_diagnostic_integral':nativegap/lam_max,'physical_dose_gap_integral':dosegap/lam_max,
            'regret_m1_to_m2':r12,'regret_m2_to_m1':r21,'regret_asymmetry_m1to2_minus_m2to1':r12-r21,
            'symmetric_cross_model_regret_integral':.5*(r12+r21),'symmetric_outcome_divergence_integral':odiv/lam_max,
            'baseline_violation_m1_to_m2':bv12/lam_max,'baseline_violation_m2_to_m1':bv21/lam_max,
            'baseline_deficit_area_m1_to_m2':bd12/lam_max,'baseline_deficit_area_m2_to_m1':bd21/lam_max},segs

events_by_coord={'DV':sorted(m1[m1.coordinate=='DV'].event_id.unique().astype(int).tolist()),
                 'DN':sorted(m1[m1.coordinate=='DN'].event_id.unique().astype(int).tolist())}
assert len(events_by_coord['DV'])==42 and len(events_by_coord['DN'])==61 and set(LOCKED_EVENTS).issubset(events_by_coord['DV'])
rows=[]; segrows=[]
for coord,events in events_by_coord.items():
    for spec,cform,lmax in SPECS:
        for eid in events:
            r,s=event_metrics(eid,coord,spec,cform,lmax); rows.append(r); segrows.extend(s)
ev=pd.DataFrame(rows); seg=pd.DataFrame(segrows)
assert len(ev)==(42+61)*len(SPECS) and np.isfinite(ev.select_dtypes(include=[np.number]).to_numpy()).all()
ev.to_csv(OUT/'event_decision_consequence_matched.csv',index=False,float_format='%.12g')
seg.to_csv(OUT/'decision_partitions_matched.csv',index=False,float_format='%.12g')

# Anchor decisions, including the predeclared no-cost lambda=0 view.
anchors=[]
for coord,events in events_by_coord.items():
    for eid in events:
        a=table(m1,eid,coord); b=table(m2,eid,coord); C=cost(Q_LEVELS,'quadratic')
        for lam in [0.0,0.05,0.10,0.25,0.50,1.00]:
            i1=choose(a['B'],C,lam); i2=choose(b['B'],C,lam)
            anchors.append({'coordinate':coord,'event_id':eid,'event_date':a['date'],'daynight':a['daynight'],'lambda':lam,
                            'm1_q':Q_LEVELS[i1],'m2_q':Q_LEVELS[i2],'disagree':bool(i1!=i2),
                            'm1_native_command':a['cmd'][i1],'m2_native_command':b['cmd'][i2]})
pd.DataFrame(anchors).to_csv(OUT/'lambda_anchor_decisions_matched.csv',index=False,float_format='%.12g')
anchor_df=pd.DataFrame(anchors)
anchor_summary=anchor_df.groupby(['coordinate','lambda'],as_index=False).agg(
    n_events=('event_id','count'),disagreement_count=('disagree','sum'),disagreement_fraction=('disagree','mean'),
    m1_mean_q=('m1_q','mean'),m2_mean_q=('m2_q','mean')
)
anchor_summary.to_csv(OUT/'lambda_anchor_summary_matched.csv',index=False,float_format='%.12g')

# Date-cluster bootstrap for primary specification.
def boot_ci(df,col):
    g=df.groupby('event_date')[col].agg(['sum','count']).sort_index(); sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(int)
    rng=np.random.default_rng(SEED); vals=np.empty(NBOOT); n=len(g)
    for k in range(NBOOT):
        ix=rng.integers(0,n,size=n); vals[k]=sums[ix].sum()/counts[ix].sum()
    return np.quantile(vals,[.025,.975])

primary=ev[ev.specification=='primary_quadratic_L1'].copy()
metrics=['decision_disagreement_measure','physical_q_action_gap_integral','symmetric_cross_model_regret_integral',
         'regret_m1_to_m2','regret_m2_to_m1','regret_asymmetry_m1to2_minus_m2to1','symmetric_outcome_divergence_integral',
         'physical_dose_gap_integral','native_command_gap_diagnostic_integral']
srows=[]
for coord in ['DV','DN']:
    x=primary[primary.coordinate==coord]
    for metric in metrics:
        lo,hi=boot_ci(x,metric)
        srows.append({'coordinate':coord,'specification':'primary_quadratic_L1','metric':metric,'n_events':len(x),
                      'mean':float(x[metric].mean()),'median':float(x[metric].median()),'ci_low':float(lo),'ci_high':float(hi),
                      'fraction_any_disagreement':float(x.any_disagreement.mean())})
summary=pd.DataFrame(srows); summary.to_csv(OUT/'primary_summary.csv',index=False,float_format='%.12g')

# Sensitivity summary (predeclared; no post-hoc promotion).
sens=[]
for (coord,spec),x in ev.groupby(['coordinate','specification']):
    sens.append({'coordinate':coord,'specification':spec,'n_events':len(x),'any_disagreement_count':int(x.any_disagreement.sum()),
                 'any_disagreement_fraction':float(x.any_disagreement.mean()),
                 'mean_disagreement_measure':float(x.decision_disagreement_measure.mean()),
                 'mean_q_action_gap':float(x.physical_q_action_gap_integral.mean()),
                 'mean_symmetric_regret':float(x.symmetric_cross_model_regret_integral.mean()),
                 'mean_outcome_divergence':float(x.symmetric_outcome_divergence_integral.mean())})
pd.DataFrame(sens).to_csv(OUT/'predeclared_sensitivity_summary.csv',index=False,float_format='%.12g')

# Locked event detail.
locked=primary[(primary.coordinate=='DV')&primary.event_id.isin(LOCKED_EVENTS)].copy().sort_values('event_id')
assert locked.event_id.astype(int).tolist()==LOCKED_EVENTS
locked.to_csv(OUT/'locked_events_27_86_89_matched_DV.csv',index=False,float_format='%.12g')

# Cohort-matched before/after against locked EXP3.1 event-level primary results.
old=pd.read_csv(OLD/'event_decision_consequence.csv')
old=old[(old.horizon_min==15)&np.isclose(old.lambda_max,1.0)].copy(); assert len(old)==61
old_metrics={'decision_disagreement_measure':'decision_disagreement_measure',
             'normalized_action_gap_integral':'physical_q_action_gap_integral',
             'symmetric_cross_model_regret_integral':'symmetric_cross_model_regret_integral',
             'symmetric_outcome_divergence_integral':'symmetric_outcome_divergence_integral'}
ba=[]
for coord,ids in events_by_coord.items():
    o=old[old.event_id.isin(ids)]; n=primary[primary.coordinate==coord]
    assert len(o)==len(n)==len(ids)
    for om,nm in old_metrics.items():
        ov=float(o[om].mean()); nv=float(n[nm].mean())
        ba.append({'coordinate':coord,'cohort_events':len(ids),'old_metric':om,'matched_metric':nm,
                   'old_EXP3_1_mean':ov,'matched_EXP3_4B_mean':nv,'matched_minus_old':nv-ov,
                   'old_basis':'native-command coordinate','matched_basis':'common physical-dose q coordinate'})
    ba.append({'coordinate':coord,'cohort_events':len(ids),'old_metric':'any_disagreement_count','matched_metric':'any_disagreement_count',
               'old_EXP3_1_mean':int(o.any_disagreement.sum()),'matched_EXP3_4B_mean':int(n.any_disagreement.sum()),
               'matched_minus_old':int(n.any_disagreement.sum()-o.any_disagreement.sum()),
               'old_basis':'native-command coordinate','matched_basis':'common physical-dose q coordinate'})
pd.DataFrame(ba).to_csv(OUT/'before_after_EXP3_1_vs_EXP3_4B.csv',index=False,float_format='%.12g')

old_locked=old[old.event_id.isin(LOCKED_EVENTS)].copy()
old_locked=old_locked[['event_id','decision_disagreement_measure','normalized_action_gap_integral','symmetric_cross_model_regret_integral','symmetric_outcome_divergence_integral']]
new_locked=locked[['event_id','decision_disagreement_measure','physical_q_action_gap_integral','symmetric_cross_model_regret_integral','symmetric_outcome_divergence_integral','regret_m1_to_m2','regret_m2_to_m1']]
lockcmp=old_locked.merge(new_locked,on='event_id',suffixes=('_EXP3_1','_EXP3_4B'),validate='one_to_one')
lockcmp.to_csv(OUT/'locked_events_27_86_89_before_after.csv',index=False,float_format='%.12g')

sets={}
for coord in ['DV','DN']:
    x=primary[primary.coordinate==coord]; sets[coord]=sorted(x.loc[x.any_disagreement,'event_id'].astype(int).tolist())

result={
 'experiment':'PhysBench-GH EXP3.4B — Physical-Dose-Matched 61-Event Decision Consequence',
 'status':'PASS_EXP3_4B',
 'target_sha256':TARGET_SHA256,'input_sha256':INPUT_SHA256,
 'primary':{'coordinate':'DV','horizon_min':15,'events':42,'q_levels':Q_LEVELS.tolist(),'cost':'q^2','lambda_domain':[0,1],
            'disagreement_event_ids':sets['DV'],'disagreement_count':len(sets['DV']),
            'locked_27_86_89_persist':[eid for eid in LOCKED_EVENTS if eid in sets['DV']]},
 'secondary':{'coordinate':'DN','horizon_min':15,'events':61,'q_levels':Q_LEVELS.tolist(),'cost':'q^2','lambda_domain':[0,1],
              'disagreement_event_ids':sets['DN'],'disagreement_count':len(sets['DN'])},
 'dose_match':{'M1_max_DV_abs_error':s1['max_DV_abs_error'],'M2_max_DV_abs_error':s2['max_DV_abs_error'],
               'M1_max_DN_abs_error':s1['max_DN_abs_error'],'M2_max_DN_abs_error':s2['max_DN_abs_error'],
               'M1_max_inversion_evaluations':s1['max_inversion_evaluations'],'M2_max_inversion_evaluations':s2['max_inversion_evaluations']},
 'scientific_disagreement_change_is_not_runtime_failure':True
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
