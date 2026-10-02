import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1DIR=ROOT/'evidence/EXP3_2_M1'
M2DIR=ROOT/'evidence/EXP3_2_M2'
OUT=ROOT/'evidence/EXP3_3_FINAL'; OUT.mkdir(parents=True,exist_ok=True)

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
SEED=20261002; NBOOT=2000; TOL=1e-10
LOCKED_SUBGROUP=[27,86,89]

LOCKED_EXP31={
 'decision_disagreement_measure':{
   'mean':0.024552427907011506,'median':0.0,'ci_low':0.0,'ci_high':0.059165829930892304,
   'fraction_any_disagreement':0.04918032786885246},
 'normalized_action_gap_integral':{
   'mean':0.006138106976752877,'median':0.0,'ci_low':0.0,'ci_high':0.014791457482723078,
   'fraction_any_disagreement':0.04918032786885246},
 'reconstructed_symmetric_cross_model_regret_integral':{
   'mean':0.001645164794050128,'median':0.0,'ci_low':0.0,'ci_high':0.003910665621685459,
   'fraction_any_disagreement':0.04918032786885246},
 'reconstructed_symmetric_outcome_divergence_integral':{
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

_REQUIRED={'event_id','team','event_date','daynight','horizon_min','action','T_gradient_C','AH_gradient_g_m3','T','AH'}
for _name,_df in (('M1',m1),('M2',m2)):
    assert not (_REQUIRED-set(_df.columns)), (_name,sorted(_REQUIRED-set(_df.columns)))
    assert set(_df.horizon_min.to_numpy(int))=={15,30}
    assert not _df.duplicated(['event_id','horizon_min','action']).any()
    _counts=_df.groupby(['event_id','horizon_min']).size().to_numpy(int)
    assert len(_counts)==122 and np.all(_counts==9)
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
            if 0.0 < lam < lam_max: vals.append(float(lam))
    vals=sorted(vals); out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,C,lam):
    q=B-lam*C; mx=np.max(q)
    return int(np.where(q>=mx-1e-12)[0][0])

def norm_regret(B,C,chosen_idx,opt_idx,lam):
    q=B-lam*C
    rg=float(q[opt_idx]-q[chosen_idx])
    span=float(np.max(q)-np.min(q))
    if span<=1e-15: return 0.0
    return max(0.0,rg/span)

def negative_interval_stats(B,C,lo,hi):
    width=hi-lo
    if width<=1e-15: return 0.0,0.0
    if abs(C)<=1e-15:
        return (width,-B*width) if B<0 else (0.0,0.0)
    root=B/C
    start=max(lo,root)
    if start>=hi: return 0.0,0.0
    measure=hi-start
    area=0.5*C*(hi*hi-start*start)-B*(hi-start)
    return measure,max(0.0,area)

def event_metrics(event_id,horizon,lam_max,grid_label,weight_label,cost_form):
    actions=GRIDS[grid_label]; wT,wAH=WEIGHTS[weight_label]; C=cost(actions,cost_form)
    a=benefit_table(m1,event_id,horizon,actions,wT,wAH)
    b=benefit_table(m2,event_id,horizon,actions,wT,wAH)
    bp=sorted(set(breakpoints(a['B'],C,lam_max)+breakpoints(b['B'],C,lam_max)))
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)

    cover=0.0; disagree=0.0; gap=0.0; sym_odiv=0.0
    r12=0.0; r21=0.0
    bv12=0.0; bv21=0.0; bd12=0.0; bd21=0.0
    lt12=0.0; la12=0.0; lt21=0.0; la21=0.0
    signed_shift=0.0; abs_shift=0.0

    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=(lo+hi)/2
        i1=choose(a['B'],C,mid); i2=choose(b['B'],C,mid)
        cover+=width
        if i1!=i2: disagree+=width
        shift=(actions[i1]-actions[i2])/0.8
        signed_shift += width*shift
        abs_shift += width*abs(shift)
        gap += width*abs(shift)

        rr12=norm_regret(b['B'],C,i1,i2,mid)
        rr21=norm_regret(a['B'],C,i2,i1,mid)
        r12 += width*rr12; r21 += width*rr21

        m,aarea=negative_interval_stats(float(b['B'][i1]),float(C[i1]),lo,hi)
        bv12 += m; bd12 += aarea
        m,aarea=negative_interval_stats(float(a['B'][i2]),float(C[i2]),lo,hi)
        bv21 += m; bd21 += aarea

        lt12 += width*float(b['bT'][i2]-b['bT'][i1])
        la12 += width*float(b['bAH'][i2]-b['bAH'][i1])
        lt21 += width*float(a['bT'][i1]-a['bT'][i2])
        la21 += width*float(a['bAH'][i1]-a['bAH'][i2])

        d1=0.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT']) + abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=0.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT']) + abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        sym_odiv += width*0.5*(d1+d2)

    if abs(cover-lam_max)>1e-10: raise AssertionError((event_id,horizon,lam_max,grid_label,weight_label,cost_form,cover))
    rr12=r12/lam_max; rr21=r21/lam_max
    return {
      'event_id':event_id,'team':a['team'],'event_date':a['event_date'],'daynight':a['daynight'],
      'horizon_min':horizon,'lambda_max':lam_max,'grid':grid_label,'n_actions':len(actions),
      'weight':weight_label,'wT':wT,'wAH':wAH,'cost_form':cost_form,
      'decision_disagreement_measure':disagree/lam_max,'any_disagreement':bool(disagree>1e-12),
      'normalized_action_gap_integral':gap/lam_max,
      'signed_action_shift_integral':signed_shift/lam_max,
      'absolute_action_shift_integral':abs_shift/lam_max,
      'regret_m1_to_m2':rr12,'regret_m2_to_m1':rr21,
      'regret_asymmetry_m1to2_minus_m2to1':rr12-rr21,
      'baseline_violation_m1_to_m2':bv12/lam_max,
      'baseline_violation_m2_to_m1':bv21/lam_max,
      'baseline_deficit_area_m1_to_m2':bd12/lam_max,
      'baseline_deficit_area_m2_to_m1':bd21/lam_max,
      'T_loss_m1_to_m2':lt12/lam_max,'AH_loss_m1_to_m2':la12/lam_max,
      'T_loss_m2_to_m1':lt21/lam_max,'AH_loss_m2_to_m1':la21/lam_max,
      'reconstructed_symmetric_cross_model_regret_integral':0.5*(rr12+rr21),
      'reconstructed_symmetric_outcome_divergence_integral':sym_odiv/lam_max
    }

events=sorted(set(m1.event_id)); rows=[]
for h in HORIZONS:
  for lm in LAMBDA_MAXES:
    for grid_label in GRIDS:
      for weight_label in WEIGHTS:
        for cost_form in COST_FORMS:
          for eid in events:
            rows.append(event_metrics(eid,h,lm,grid_label,weight_label,cost_form))
ev=pd.DataFrame(rows)
ev.to_csv(OUT/'event_directional_model_swap.csv',index=False,float_format='%.12g')
assert len(ev)==4392
assert not ev.duplicated(['event_id','horizon_min','lambda_max','grid','weight','cost_form']).any()
assert np.isfinite(ev.select_dtypes(include=[np.number]).to_numpy()).all()

def cluster_boot_ci(df,col):
    g=df.groupby('event_date')[col].agg(['sum','count']).sort_index()
    sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(int)
    rng=np.random.default_rng(SEED); vals=np.empty(NBOOT); n=len(g)
    for k in range(NBOOT):
        ix=rng.integers(0,n,size=n)
        vals[k]=sums[ix].sum()/counts[ix].sum()
    return np.quantile(vals,[0.025,0.975])

DIR_METRICS=[
 'regret_m1_to_m2','regret_m2_to_m1','regret_asymmetry_m1to2_minus_m2to1',
 'baseline_violation_m1_to_m2','baseline_violation_m2_to_m1',
 'baseline_deficit_area_m1_to_m2','baseline_deficit_area_m2_to_m1',
 'T_loss_m1_to_m2','AH_loss_m1_to_m2','T_loss_m2_to_m1','AH_loss_m2_to_m1',
 'signed_action_shift_integral','absolute_action_shift_integral'
]
summary_rows=[]
for h in HORIZONS:
  for lm in LAMBDA_MAXES:
    for grid_label in GRIDS:
      for weight_label in WEIGHTS:
        for cost_form in COST_FORMS:
          base=ev[(ev.horizon_min==h)&(ev.lambda_max==lm)&(ev.grid==grid_label)&(ev.weight==weight_label)&(ev.cost_form==cost_form)]
          for stratum in ('all','day','night'):
            x=base if stratum=='all' else base[base.daynight==stratum]
            for col in DIR_METRICS:
                summary_rows.append({
                  'horizon_min':h,'lambda_max':lm,'grid':grid_label,'n_actions':len(GRIDS[grid_label]),
                  'weight':weight_label,'wT':WEIGHTS[weight_label][0],'wAH':WEIGHTS[weight_label][1],
                  'cost_form':cost_form,'stratum':stratum,'metric':col,'n_events':len(x),
                  'mean':float(x[col].mean()),'median':float(x[col].median()),
                  'fraction_positive':float((x[col]>1e-12).mean()),'fraction_negative':float((x[col]<-1e-12).mean())
                })
dir_summary=pd.DataFrame(summary_rows)
dir_summary.to_csv(OUT/'directional_summary.csv',index=False,float_format='%.12g')
assert len(dir_summary)==2*2*3*3*2*3*len(DIR_METRICS)

ref=ev[(ev.horizon_min==15)&(ev.lambda_max==1.0)&(ev.grid=='g5')&(ev.weight=='equal')&(ev.cost_form=='quadratic')].copy()
ref.to_csv(OUT/'primary_reference_event_consequence.csv',index=False,float_format='%.12g')
assert len(ref)==61
ref_ids=ref.loc[ref.any_disagreement,'event_id'].astype(int).tolist()

RECON_COLS=[
 'decision_disagreement_measure','normalized_action_gap_integral',
 'reconstructed_symmetric_cross_model_regret_integral','reconstructed_symmetric_outcome_divergence_integral'
]
recon_rows=[]
for col in RECON_COLS:
    lo,hi=cluster_boot_ci(ref,col)
    recon_rows.append({'metric':col,'n_events':len(ref),'mean':float(ref[col].mean()),'median':float(ref[col].median()),
                       'ci_low':float(lo),'ci_high':float(hi),'fraction_any_disagreement':float(ref.any_disagreement.mean())})
recon=pd.DataFrame(recon_rows)
recon.to_csv(OUT/'exp3_1_reference_reconstruction.csv',index=False,float_format='%.12g')

repro={'status':'PASS','tolerance':TOL,'locked_disagreement_event_ids':LOCKED_SUBGROUP,
       'observed_disagreement_event_ids':ref_ids,'checks':{}}
if ref_ids!=LOCKED_SUBGROUP: repro['status']='FAIL'
for metric,expected in LOCKED_EXP31.items():
    r=recon[recon.metric==metric].iloc[0]; checks={}
    for field,val in expected.items():
        got=float(r[field]); err=abs(got-val)
        checks[field]={'expected':val,'observed':got,'abs_error':err,'pass':bool(err<=TOL)}
        if err>TOL: repro['status']='FAIL'
    repro['checks'][metric]=checks
(OUT/'exp3_1_reference_reconstruction_gate.json').write_text(json.dumps(repro,indent=2),encoding='utf-8')
if repro['status']!='PASS': raise SystemExit('EXP3.1 directional reconstruction failed')

panel=ev[(ev.horizon_min==15)&(ev.lambda_max==1.0)].copy()
assert panel.groupby('event_id').size().eq(18).all()
persist=[]
for eid,x in panel.groupby('event_id'):
    first=x.iloc[0]
    asym=x['regret_asymmetry_m1to2_minus_m2to1'].to_numpy(float)
    persist.append({
      'event_id':int(eid),'team':first.team,'event_date':first.event_date,'daynight':first.daynight,
      'specifications':18,
      'specs_any_disagreement':int(x.any_disagreement.sum()),
      'specs_regret_m1_to_m2_positive':int((x.regret_m1_to_m2>1e-12).sum()),
      'specs_regret_m2_to_m1_positive':int((x.regret_m2_to_m1>1e-12).sum()),
      'specs_baseline_violation_m1_to_m2':int((x.baseline_violation_m1_to_m2>1e-12).sum()),
      'specs_baseline_violation_m2_to_m1':int((x.baseline_violation_m2_to_m1>1e-12).sum()),
      'specs_asymmetry_positive':int((asym>1e-12).sum()),
      'specs_asymmetry_negative':int((asym<-1e-12).sum()),
      'specs_asymmetry_zero':int((np.abs(asym)<=1e-12).sum()),
      'mean_regret_m1_to_m2':float(x.regret_m1_to_m2.mean()),
      'mean_regret_m2_to_m1':float(x.regret_m2_to_m1.mean()),
      'mean_regret_asymmetry':float(x.regret_asymmetry_m1to2_minus_m2to1.mean()),
      'mean_baseline_violation_m1_to_m2':float(x.baseline_violation_m1_to_m2.mean()),
      'mean_baseline_violation_m2_to_m1':float(x.baseline_violation_m2_to_m1.mean())
    })
persistence=pd.DataFrame(persist).sort_values('event_id')
persistence.to_csv(OUT/'primary_18spec_directional_persistence.csv',index=False,float_format='%.12g')
assert len(persistence)==61

sub=panel[panel.event_id.isin(LOCKED_SUBGROUP)].copy()
assert len(sub)==3*18
sub.to_csv(OUT/'locked_subgroup_27_86_89_18spec.csv',index=False,float_format='%.12g')

primary_summary={
 'n_events':61,
 'disagreement_event_ids':ref_ids,
 'mean_regret_m1_to_m2':float(ref.regret_m1_to_m2.mean()),
 'mean_regret_m2_to_m1':float(ref.regret_m2_to_m1.mean()),
 'mean_regret_asymmetry':float(ref.regret_asymmetry_m1to2_minus_m2to1.mean()),
 'fraction_events_any_baseline_violation_m1_to_m2':float((ref.baseline_violation_m1_to_m2>1e-12).mean()),
 'fraction_events_any_baseline_violation_m2_to_m1':float((ref.baseline_violation_m2_to_m1>1e-12).mean()),
 'mean_baseline_violation_m1_to_m2':float(ref.baseline_violation_m1_to_m2.mean()),
 'mean_baseline_violation_m2_to_m1':float(ref.baseline_violation_m2_to_m1.mean()),
 'mean_signed_action_shift_integral':float(ref.signed_action_shift_integral.mean()),
 'mean_absolute_action_shift_integral':float(ref.absolute_action_shift_integral.mean())
}

result={
 'experiment':'PhysBench-GH EXP3.3 Non-Redundant Directional Model-Swap Consequence',
 'status':'PASS_DIRECTIONAL_MODEL_SWAP' if len(ev)==4392 and repro['status']=='PASS' else 'FAIL',
 'locked_exp3_1_commit':'8c68e2262f9cbaffdc1a14a2309e9fdf7644df11',
 'locked_exp3_2_commit':'48feabd403e9a4f65aa438a5da3d507fc754e630',
 'events':61,'locked_subgroup':LOCKED_SUBGROUP,
 'primary_specification':{'horizon_min':15,'lambda_domain':[0,1],'grid':'g5','weights':[0.5,0.5],'cost_form':'quadratic'},
 'm1_summary':s1,'m2_summary':s2,
 'reference_reconstruction_status':repro['status'],
 'primary_reference':primary_summary,
 'robustness_18spec':{
   'events_with_any_m1_to_m2_baseline_violation':int((persistence.specs_baseline_violation_m1_to_m2>0).sum()),
   'events_with_any_m2_to_m1_baseline_violation':int((persistence.specs_baseline_violation_m2_to_m1>0).sum()),
   'events_with_regret_asymmetry_both_signs_across_specs':int(((persistence.specs_asymmetry_positive>0)&(persistence.specs_asymmetry_negative>0)).sum())
 }
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if result['status']!='PASS_DIRECTIONAL_MODEL_SWAP': raise SystemExit('EXP3.3 failed')
