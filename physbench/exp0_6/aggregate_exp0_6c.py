import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/EXP0_6C_FINAL'
OUT.mkdir(parents=True,exist_ok=True)

HIST_DERIVED=['H00','H01','H04','H06','H07','H10','H11','H13','H16']
HIST_ALL=['HNR']+HIST_DERIVED
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
COST=((ACTIONS-0.1)/0.8)**2
LOCKED_IDS=[27,86,89]
TOL=1e-10
LOCKED={
 'decision_disagreement_measure':0.024552427907011506,
 'normalized_action_gap_integral':0.006138106976752877,
 'symmetric_cross_model_regret_integral':0.001645164794050128,
 'symmetric_outcome_divergence_integral':0.01691271675501554
}

def load_model(model):
    frames=[]; summaries=[]
    for hid in HIST_ALL:
        d=ROOT/f'evidence/EXP0_6C_{model}_{hid}'
        s=json.loads((d/'summary.json').read_text())
        if not s['gate_pass']:
            raise AssertionError((model,hid,'runtime gate failed'))
        summaries.append(s)
        x=pd.read_csv(d/'action_grid_responses.csv')
        frames.append(x)
    df=pd.concat(frames,ignore_index=True)
    return df,summaries

m1,s1=load_model('M1')
m2,s2=load_model('M2')

for ss,model,commit in [
    (s1,'M1','2d3febb1ea002b24b452e32293e990beb78d3ce1'),
    (s2,'M2','bea8c3b0a1324162a4b5487db578aa674c8b587c')
]:
    assert [x['history_id'] for x in ss]==HIST_ALL
    assert all(x['events']==61 for x in ss)
    assert all(x['actions']==ACTIONS.tolist() for x in ss)
    assert all(x['input_sha256']=='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023' for x in ss)
    assert all(x['frozen_commit']==commit for x in ss)

REQ={'model_id','history_id','event_id','team','event_date','daynight','horizon_min','action',
     'T_gradient_C','AH_gradient_g_m3','T','AH'}
for name,df in [('M1',m1),('M2',m2)]:
    miss=REQ-set(df.columns)
    assert not miss,(name,miss)
    assert len(df)==10*61*2*5
    assert set(df.history_id)==set(HIST_ALL)
    assert set(df.horizon_min.astype(int))=={15,30}
    assert not df.duplicated(['history_id','event_id','horizon_min','action']).any()
    counts=df.groupby(['history_id','event_id','horizon_min']).size().to_numpy(int)
    assert len(counts)==10*61*2 and np.all(counts==5)
    for col in ['action','T_gradient_C','AH_gradient_g_m3','T','AH']:
        v=pd.to_numeric(df[col],errors='raise').to_numpy(float)
        assert np.all(np.isfinite(v)),(name,col)
assert set(m1.event_id)==set(m2.event_id) and len(set(m1.event_id))==61

def benefit_table(df,hid,eid,horizon):
    x=df[(df.history_id==hid)&(df.event_id==eid)&(df.horizon_min==horizon)].sort_values('action')
    assert len(x)==len(ACTIONS)
    assert np.allclose(x.action.to_numpy(float),ACTIONS,rtol=0,atol=1e-12)
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0])
    assert abs(gt)>0 and abs(ga)>0
    T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt)
    bA=-np.sign(ga)*(AH-AH[0])/abs(ga)
    B=0.5*(bT+bA)
    return {'T':T,'AH':AH,'bT':bT,'bAH':bA,'B':B,'GT':gt,'GAH':ga,
            'team':x.team.iloc[0],'event_date':x.event_date.iloc[0],'daynight':x.daynight.iloc[0]}

def breakpoints(B,lam_max):
    vals=[0.0,float(lam_max)]
    for i in range(len(ACTIONS)):
        for j in range(i+1,len(ACTIONS)):
            dc=COST[i]-COST[j]
            if abs(dc)<1e-15: continue
            lam=(B[i]-B[j])/dc
            if 0.0<lam<lam_max: vals.append(float(lam))
    vals=sorted(vals); out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-12: out.append(v)
    return out

def choose(B,lam):
    q=B-lam*COST
    mx=np.max(q)
    return int(np.where(q>=mx-1e-12)[0][0])

def norm_regret(B,chosen_idx,opt_idx,lam):
    q=B-lam*COST
    rg=float(q[opt_idx]-q[chosen_idx])
    span=float(np.max(q)-np.min(q))
    if span<=1e-15: return 0.0
    return max(0.0,rg/span)

def event_metrics(hid,eid,horizon,lam_max):
    a=benefit_table(m1,hid,eid,horizon)
    b=benefit_table(m2,hid,eid,horizon)
    bp=sorted(set(breakpoints(a['B'],lam_max)+breakpoints(b['B'],lam_max)))
    bpu=[]
    for v in bp:
        if not bpu or abs(v-bpu[-1])>1e-12: bpu.append(v)
    cover=0.0; disagree=0.0; gap=0.0
    r_m1_to_m2=0.0; r_m2_to_m1=0.0; odiv=0.0
    seg=[]
    for lo,hi in zip(bpu[:-1],bpu[1:]):
        width=hi-lo
        if width<=1e-15: continue
        mid=(lo+hi)/2
        i1=choose(a['B'],mid); i2=choose(b['B'],mid)
        cover+=width
        dis=i1!=i2
        if dis: disagree+=width
        gap += width*abs(ACTIONS[i1]-ACTIONS[i2])/0.8
        r12=norm_regret(b['B'],i1,i2,mid)
        r21=norm_regret(a['B'],i2,i1,mid)
        r_m1_to_m2 += width*r12
        r_m2_to_m1 += width*r21
        d1=0.5*(abs(a['T'][i1]-a['T'][i2])/abs(a['GT']) + abs(a['AH'][i1]-a['AH'][i2])/abs(a['GAH']))
        d2=0.5*(abs(b['T'][i1]-b['T'][i2])/abs(b['GT']) + abs(b['AH'][i1]-b['AH'][i2])/abs(b['GAH']))
        odiv += width*0.5*(d1+d2)
        seg.append({
          'history_id':hid,'event_id':eid,'horizon_min':horizon,'lambda_max':lam_max,
          'lambda_lo':lo,'lambda_hi':hi,'m1_action':ACTIONS[i1],'m2_action':ACTIONS[i2],
          'disagree':dis,'regret_m1_to_m2_mid':r12,'regret_m2_to_m1_mid':r21,
          'symmetric_outcome_divergence_mid':0.5*(d1+d2)
        })
    if abs(cover-lam_max)>1e-10:
        raise AssertionError((hid,eid,horizon,lam_max,cover))
    r12=r_m1_to_m2/lam_max; r21=r_m2_to_m1/lam_max
    return {
      'history_id':hid,'event_id':eid,'team':a['team'],'event_date':a['event_date'],'daynight':a['daynight'],
      'horizon_min':horizon,'lambda_max':lam_max,
      'decision_disagreement_measure':disagree/lam_max,
      'normalized_action_gap_integral':gap/lam_max,
      'regret_m1_to_m2':r12,
      'regret_m2_to_m1':r21,
      'regret_asymmetry_m1to2_minus_m2to1':r12-r21,
      'symmetric_cross_model_regret_integral':0.5*(r12+r21),
      'symmetric_outcome_divergence_integral':odiv/lam_max,
      'any_disagreement':bool(disagree>1e-12)
    },seg

event_rows=[]; seg_rows=[]
events=sorted(set(m1.event_id.astype(int)))
for hid in HIST_ALL:
    for h in (15,30):
        for lm in (1.0,3.0):
            for eid in events:
                r,s=event_metrics(hid,eid,h,lm)
                event_rows.append(r); seg_rows.extend(s)
ev=pd.DataFrame(event_rows)
seg=pd.DataFrame(seg_rows)
assert len(ev)==10*2*2*61
assert not ev.duplicated(['history_id','event_id','horizon_min','lambda_max']).any()
assert np.isfinite(ev.select_dtypes(include=[np.number]).to_numpy()).all()
ev.to_csv(OUT/'event_decision_latent_robustness.csv',index=False,float_format='%.12g')
seg.to_csv(OUT/'decision_partitions.csv',index=False,float_format='%.12g')

# HNR reconstruction gate.
ref=ev[(ev.history_id=='HNR')&(ev.horizon_min==15)&(ev.lambda_max==1.0)].copy()
obs_ids=ref.loc[ref.any_disagreement,'event_id'].astype(int).tolist()
recon={
 'status':'PASS','tolerance':TOL,
 'locked_disagreement_event_ids':LOCKED_IDS,
 'observed_disagreement_event_ids':obs_ids,
 'checks':{}
}
if obs_ids!=LOCKED_IDS: recon['status']='FAIL'
for col,expected in LOCKED.items():
    got=float(ref[col].mean())
    err=abs(got-expected)
    recon['checks'][col]={'expected':expected,'observed':got,'abs_error':err,'pass':bool(err<=TOL)}
    if err>TOL: recon['status']='FAIL'
(OUT/'HNR_EXP3_1_RECONSTRUCTION_GATE.json').write_text(json.dumps(recon,indent=2),encoding='utf-8')
if recon['status']!='PASS':
    raise SystemExit('EXP0.6C HNR EXP3.1 reconstruction failed')

# Per-history summaries, including secondary horizon/domain.
summary_rows=[]
for hid in HIST_ALL:
  for h in (15,30):
    for lm in (1.0,3.0):
      base=ev[(ev.history_id==hid)&(ev.horizon_min==h)&(ev.lambda_max==lm)]
      for stratum in ('all','day','night'):
        x=base if stratum=='all' else base[base.daynight==stratum]
        dset=sorted(x.loc[x.any_disagreement,'event_id'].astype(int).tolist())
        summary_rows.append({
          'history_id':hid,'history_derived':hid!='HNR','horizon_min':h,'lambda_max':lm,'stratum':stratum,
          'n_events':int(len(x)),'n_any_disagreement':int(x.any_disagreement.sum()),
          'fraction_any_disagreement':float(x.any_disagreement.mean()),
          'mean_decision_disagreement_measure':float(x.decision_disagreement_measure.mean()),
          'median_decision_disagreement_measure':float(x.decision_disagreement_measure.median()),
          'mean_normalized_action_gap_integral':float(x.normalized_action_gap_integral.mean()),
          'mean_symmetric_cross_model_regret_integral':float(x.symmetric_cross_model_regret_integral.mean()),
          'mean_symmetric_outcome_divergence_integral':float(x.symmetric_outcome_divergence_integral.mean()),
          'mean_regret_m1_to_m2':float(x.regret_m1_to_m2.mean()),
          'mean_regret_m2_to_m1':float(x.regret_m2_to_m1.mean()),
          'mean_regret_asymmetry_m1to2_minus_m2to1':float(x.regret_asymmetry_m1to2_minus_m2to1.mean()),
          'disagreement_event_ids':'|'.join(map(str,dset))
        })
hs=pd.DataFrame(summary_rows)
hs.to_csv(OUT/'history_decision_summary.csv',index=False,float_format='%.12g')

# Primary history sets and Jaccard relative to HNR.
primary=ev[(ev.horizon_min==15)&(ev.lambda_max==1.0)].copy()
hnr_set=set(LOCKED_IDS)
set_rows=[]
sets={}
for hid in HIST_ALL:
    x=primary[primary.history_id==hid]
    s=set(x.loc[x.any_disagreement,'event_id'].astype(int).tolist())
    sets[hid]=s
    union=s|hnr_set
    jac=1.0 if not union else len(s&hnr_set)/len(union)
    set_rows.append({
      'history_id':hid,'history_derived':hid!='HNR','n_disagreement_events':len(s),
      'disagreement_event_ids':'|'.join(map(str,sorted(s))),
      'jaccard_vs_HNR':float(jac),
      'retained_HNR_event_count':len(s&hnr_set),
      'retained_HNR_event_ids':'|'.join(map(str,sorted(s&hnr_set))),
      'lost_HNR_event_ids':'|'.join(map(str,sorted(hnr_set-s))),
      'new_latent_sensitive_event_ids':'|'.join(map(str,sorted(s-hnr_set)))
    })
set_df=pd.DataFrame(set_rows)
set_df.to_csv(OUT/'primary_disagreement_sets.csv',index=False,float_format='%.12g')

# Per-event persistence across nine history-derived states.
persist=[]
for eid in events:
    hs_with=[hid for hid in HIST_DERIVED if eid in sets[hid]]
    persist.append({
      'event_id':eid,
      'HNR_disagreement':bool(eid in hnr_set),
      'prelocked_HNR_event':bool(eid in LOCKED_IDS),
      'latent_history_disagreement_count':len(hs_with),
      'latent_history_disagreement_fraction':len(hs_with)/len(HIST_DERIVED),
      'histories_with_disagreement':'|'.join(hs_with),
      'new_latent_sensitive_event':bool(eid not in hnr_set and len(hs_with)>0),
      'HNR_event_lost_in_any_history':bool(eid in hnr_set and len(hs_with)<len(HIST_DERIVED))
    })
persistence=pd.DataFrame(persist)
persistence.to_csv(OUT/'event_disagreement_persistence.csv',index=False,float_format='%.12g')

hist_sets=[sets[h] for h in HIST_DERIVED]
latent_union=set().union(*hist_sets)
latent_intersection=set(hist_sets[0]).intersection(*hist_sets[1:])
new_union=latent_union-hnr_set
lost_all=hnr_set-latent_union

locked_event_persistence={
 str(eid):{
   'count_out_of_9':int(persistence.loc[persistence.event_id==eid,'latent_history_disagreement_count'].iloc[0]),
   'fraction':float(persistence.loc[persistence.event_id==eid,'latent_history_disagreement_fraction'].iloc[0]),
   'histories':persistence.loc[persistence.event_id==eid,'histories_with_disagreement'].iloc[0].split('|')
              if persistence.loc[persistence.event_id==eid,'histories_with_disagreement'].iloc[0] else []
 } for eid in LOCKED_IDS
}

result={
 'experiment':'PhysBench-GH EXP0.6C — 61-Event Decision Latent-State Robustness',
 'status':'PASS_EXP0_6C',
 'parent_exp0_6b_commit':'91c5a1aba73c41ac63895f04891988a211d2ddcb',
 'history_derived_ids':HIST_DERIVED,
 'reference_history_id':'HNR',
 'events':61,'actions':ACTIONS.tolist(),
 'HNR_reconstruction':recon,
 'primary_15min_lambda_0_1':{
   'HNR_disagreement_event_ids':LOCKED_IDS,
   'latent_union_event_ids':sorted(latent_union),
   'latent_intersection_event_ids':sorted(latent_intersection),
   'new_latent_sensitive_union_event_ids':sorted(new_union),
   'HNR_events_absent_from_all_latent_histories':sorted(lost_all),
   'locked_event_persistence':locked_event_persistence,
   'history_summaries':set_df[set_df.history_id.isin(HIST_DERIVED)].to_dict(orient='records')
 },
 'runtime':{
   'M1_all_history_gates_pass':bool(all(x['gate_pass'] for x in s1)),
   'M2_all_history_gates_pass':bool(all(x['gate_pass'] for x in s2)),
   'M1_max_init_error':float(max(x['max_init_error'] for x in s1)),
   'M2_max_init_error':float(max(x['max_init_error'] for x in s2)),
   'M1_max_action_error':float(max(x['max_action_error'] for x in s1)),
   'M2_max_action_error':float(max(x['max_action_error'] for x in s2)),
   'M1_all_same_run_state_restore_exact':bool(all(x['same_run_state_restore_exact'] for x in s1)),
   'M2_all_same_run_state_restore_exact':bool(all(x['same_run_state_restore_exact'] for x in s2))
 },
 'interpretation_limit':(
   'EXP0.6C isolates event-conditioned latent-state initialization under the locked EXP3.1 command-coordinate utility. '
   'It does not establish equal physical ventilation dose, empirical controller optimality, or realized-actuator validation.'
 )
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
