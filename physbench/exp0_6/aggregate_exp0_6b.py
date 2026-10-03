import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1=ROOT/'evidence/EXP0_6B_M1'
M2=ROOT/'evidence/EXP0_6B_M2'
OUT=ROOT/'evidence/EXP0_6B_FINAL'; OUT.mkdir(parents=True,exist_ok=True)

s1=json.loads((M1/'summary.json').read_text())
s2=json.loads((M2/'summary.json').read_text())
r1=pd.read_csv(M1/'root_results.csv')
r2=pd.read_csv(M2/'root_results.csv')
m1=pd.read_csv(M1/'mechanism_results.csv')
m2=pd.read_csv(M2/'mechanism_results.csv')

assert s1['gate_pass'] and s2['gate_pass']
assert len(r1)==39 and len(r2)==39
assert set(r1.history_id)==set(r2.history_id)
HIST=['H00','H01','H04','H06','H07','H10','H11','H13','H16']

def dist_summary(df,dur,h):
    x=df[(df.duration_h==dur)&(df.horizon_s==h)&(df.history_id.isin(HIST))].copy()
    assert len(x)==9
    u=x[(x.root_count==1)&x.primary_boundary_ppm.notna()].primary_boundary_ppm.astype(float)
    return {
      'n_histories':9,'n_unique_root':int(len(u)),
      'unique_root_fraction':float(len(u)/9),
      'min_ppm':None if len(u)==0 else float(u.min()),
      'q25_ppm':None if len(u)==0 else float(u.quantile(.25)),
      'median_ppm':None if len(u)==0 else float(u.median()),
      'q75_ppm':None if len(u)==0 else float(u.quantile(.75)),
      'max_ppm':None if len(u)==0 else float(u.max()),
      'span_ppm':None if len(u)==0 else float(u.max()-u.min())
    }

def paired_cross_model(dur,h):
    a=r1[(r1.duration_h==dur)&(r1.horizon_s==h)&(r1.history_id.isin(HIST))][['history_id','root_count','primary_boundary_ppm']].copy()
    b=r2[(r2.duration_h==dur)&(r2.horizon_s==h)&(r2.history_id.isin(HIST))][['history_id','root_count','primary_boundary_ppm']].copy()
    x=a.merge(b,on='history_id',suffixes=('_M1','_M2'),validate='one_to_one')
    p=x[(x.root_count_M1==1)&(x.root_count_M2==1)&x.primary_boundary_ppm_M1.notna()&x.primary_boundary_ppm_M2.notna()].copy()
    p['signed_M1_minus_M2_ppm']=p.primary_boundary_ppm_M1-p.primary_boundary_ppm_M2
    p['absolute_gap_ppm']=p.signed_M1_minus_M2_ppm.abs()
    if len(p):
        lo1=float(p.primary_boundary_ppm_M1.min()); hi1=float(p.primary_boundary_ppm_M1.max())
        lo2=float(p.primary_boundary_ppm_M2.min()); hi2=float(p.primary_boundary_ppm_M2.max())
        overlap=max(0.0,min(hi1,hi2)-max(lo1,lo2))
        sep=max(0.0,max(lo1,lo2)-min(hi1,hi2)) if overlap==0 else 0.0
        # Correct ordered interval separation if disjoint.
        if hi2 < lo1: sep=lo1-hi2
        elif hi1 < lo2: sep=lo2-hi1
        else: sep=0.0
        out={
          'paired_unique_root_histories':int(len(p)),
          'same_direction_M1_gt_M2_count':int((p.signed_M1_minus_M2_ppm>0).sum()),
          'same_direction_M1_lt_M2_count':int((p.signed_M1_minus_M2_ppm<0).sum()),
          'gap_min_ppm':float(p.absolute_gap_ppm.min()),
          'gap_median_ppm':float(p.absolute_gap_ppm.median()),
          'gap_max_ppm':float(p.absolute_gap_ppm.max()),
          'signed_gap_median_ppm':float(p.signed_M1_minus_M2_ppm.median()),
          'root_envelope_overlap_ppm':float(overlap),
          'ordered_separation_margin_ppm':float(sep)
        }
    else:
        out={'paired_unique_root_histories':0,'same_direction_M1_gt_M2_count':0,'same_direction_M1_lt_M2_count':0,
             'gap_min_ppm':None,'gap_median_ppm':None,'gap_max_ppm':None,'signed_gap_median_ppm':None,
             'root_envelope_overlap_ppm':None,'ordered_separation_margin_ppm':None}
    p['duration_h']=dur; p['horizon_s']=h
    return out,p

root_summary={'M1':{},'M2':{}}
for label,df in [('M1',r1),('M2',r2)]:
    root_summary[label]['72h']={str(h):dist_summary(df,72,h) for h in [300,900,1800]}
    root_summary[label]['168h']={'900':dist_summary(df,168,900)}

cross={}; gap_frames=[]
for dur,h in [(72,300),(72,900),(72,1800),(168,900)]:
    key=f'{dur}h_H{h}'
    cross[key],p=paired_cross_model(dur,h)
    gap_frames.append(p)
pd.concat(gap_frames,ignore_index=True).to_csv(OUT/'cross_model_root_gaps.csv',index=False,float_format='%.12g')

# Slow-memory root shifts: 168h versus 72h at H900 within each model.
slow_rows=[]
slow_summary={}
for label,df in [('M1',r1),('M2',r2)]:
    a=df[(df.duration_h==72)&(df.horizon_s==900)&(df.history_id.isin(HIST))][['history_id','root_count','primary_boundary_ppm']]
    b=df[(df.duration_h==168)&(df.horizon_s==900)&(df.history_id.isin(HIST))][['history_id','root_count','primary_boundary_ppm']]
    x=a.merge(b,on='history_id',suffixes=('_72h','_168h'),validate='one_to_one')
    x['model_id']=label
    x['signed_shift_168_minus_72_ppm']=x.primary_boundary_ppm_168h-x.primary_boundary_ppm_72h
    x['absolute_shift_ppm']=x.signed_shift_168_minus_72_ppm.abs()
    valid=x[(x.root_count_72h==1)&(x.root_count_168h==1)&x.primary_boundary_ppm_72h.notna()&x.primary_boundary_ppm_168h.notna()]
    slow_summary[label]={
      'paired_unique_root_histories':int(len(valid)),
      'signed_shift_min_ppm':None if len(valid)==0 else float(valid.signed_shift_168_minus_72_ppm.min()),
      'signed_shift_median_ppm':None if len(valid)==0 else float(valid.signed_shift_168_minus_72_ppm.median()),
      'signed_shift_max_ppm':None if len(valid)==0 else float(valid.signed_shift_168_minus_72_ppm.max()),
      'absolute_shift_median_ppm':None if len(valid)==0 else float(valid.absolute_shift_ppm.median()),
      'absolute_shift_max_ppm':None if len(valid)==0 else float(valid.absolute_shift_ppm.max())
    }
    slow_rows.append(x)
pd.concat(slow_rows,ignore_index=True).to_csv(OUT/'slow_memory_root_shifts.csv',index=False,float_format='%.12g')

# Mechanism summaries.
def mech_summary(df,label,dur):
    x=df[(df.duration_h==dur)&(df.history_id.isin(HIST))].copy()
    # There can be fewer than 9 rows only if H900 has no unique root in some histories.
    resolved=x[x.dominant_resolved_under_closure_bound.astype(bool)] if 'dominant_resolved_under_closure_bound' in x.columns else x
    return {
      'n_mechanism_rows':int(len(x)),
      'n_resolved':int(len(resolved)),
      'dominant_term_counts_all':{str(k):int(v) for k,v in x.dominant_term.value_counts().to_dict().items()},
      'dominant_term_counts_resolved':{str(k):int(v) for k,v in resolved.dominant_term.value_counts().to_dict().items()},
      'resolved_switches_from_HNR':int(resolved.dominant_switch_from_hnr.astype(bool).sum()) if len(resolved) else 0,
      'all_switches_from_HNR':int(x.dominant_switch_from_hnr.astype(bool).sum()) if len(x) else 0,
      'anchor_cosine_min_resolved':None if len(resolved)==0 else float(resolved.anchor_cosine_similarity.min()),
      'anchor_cosine_median_resolved':None if len(resolved)==0 else float(resolved.anchor_cosine_similarity.median()),
      'anchor_l1_distance_median_resolved':None if len(resolved)==0 else float(resolved.anchor_l1_distance.median()),
      'active_sign_flip_rows_resolved':int((resolved.n_active_term_sign_flips_vs_hnr>0).sum()) if len(resolved) else 0
    }

mechanism_summary={'M1':{'HNR_dominant':str(m1[m1.history_id=='HNR'].iloc[0].dominant_term),
                         '72h':mech_summary(m1,'M1',72),'168h':mech_summary(m1,'M1',168)},
                   'M2':{'HNR_dominant':str(m2[m2.history_id=='HNR'].iloc[0].dominant_term),
                         '72h':mech_summary(m2,'M2',72),'168h':mech_summary(m2,'M2',168)}}

# Direct 72h -> 168h mechanism-label persistence.
persist_rows=[]; persistence={}
for label,df in [('M1',m1),('M2',m2)]:
    a=df[(df.duration_h==72)&(df.history_id.isin(HIST))][['history_id','dominant_term','dominant_resolved_under_closure_bound','anchor_cosine_similarity','active_sign_pattern']]
    b=df[(df.duration_h==168)&(df.history_id.isin(HIST))][['history_id','dominant_term','dominant_resolved_under_closure_bound','anchor_cosine_similarity','active_sign_pattern']]
    x=a.merge(b,on='history_id',suffixes=('_72h','_168h'),validate='one_to_one')
    x['model_id']=label
    x['dominant_same_72_vs_168']=x.dominant_term_72h==x.dominant_term_168h
    x['both_resolved']=x.dominant_resolved_under_closure_bound_72h.astype(bool)&x.dominant_resolved_under_closure_bound_168h.astype(bool)
    y=x[x.both_resolved]
    persistence[label]={
      'paired_mechanism_histories':int(len(x)),
      'both_resolved_histories':int(len(y)),
      'dominant_same_all':int(x.dominant_same_72_vs_168.sum()),
      'dominant_changed_all':int((~x.dominant_same_72_vs_168).sum()),
      'dominant_same_when_both_resolved':int(y.dominant_same_72_vs_168.sum()),
      'dominant_changed_when_both_resolved':int((~y.dominant_same_72_vs_168).sum())
    }
    persist_rows.append(x)
pd.concat(persist_rows,ignore_index=True).to_csv(OUT/'mechanism_72h_168h_persistence.csv',index=False)

# Horizon monotonicity/descriptive per matched history.
horizon_rows=[]
for label,df in [('M1',r1),('M2',r2)]:
    x=df[(df.duration_h==72)&(df.history_id.isin(HIST))].pivot(index='history_id',columns='horizon_s',values='primary_boundary_ppm')
    for hid,row in x.iterrows():
        horizon_rows.append({'model_id':label,'history_id':hid,
                             'root_300_ppm':row.get(300,np.nan),'root_900_ppm':row.get(900,np.nan),'root_1800_ppm':row.get(1800,np.nan),
                             'shift_900_minus_300_ppm':row.get(900,np.nan)-row.get(300,np.nan),
                             'shift_1800_minus_900_ppm':row.get(1800,np.nan)-row.get(900,np.nan)})
pd.DataFrame(horizon_rows).to_csv(OUT/'horizon_root_shifts_72h.csv',index=False,float_format='%.12g')

result={
 'experiment':'PhysBench-GH EXP0.6B — 168 h Slow-Memory + Horizon + Mechanism Robustness',
 'status':'PASS_EXP0_6B',
 'history_ids':HIST,
 'runtime':{'M1':s1,'M2':s2},
 'root_distributions':root_summary,
 'cross_model':cross,
 'slow_memory_168_vs_72':slow_summary,
 'mechanism':mechanism_summary,
 'mechanism_72_vs_168_persistence':persistence,
 'interpretation_guard':(
   'Root motion, root loss, envelope overlap, dominant-mechanism changes, and numerical near-ties are scientific results, not CI failures. '
   'EXP0.6B addresses slow-memory, horizon, and native mechanism robustness only; physical airflow-dose equivalence and empirical ground truth remain separate questions.'
 )
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

# Compact summary.
rows=[]
for label in ['M1','M2']:
    for h in [300,900,1800]:
        q=root_summary[label]['72h'][str(h)]
        rows.append({'model_id':label,'duration_h':72,'horizon_s':h,**q})
    rows.append({'model_id':label,'duration_h':168,'horizon_s':900,**root_summary[label]['168h']['900']})
pd.DataFrame(rows).to_csv(OUT/'root_distribution_summary.csv',index=False,float_format='%.12g')

print(json.dumps(result,indent=2))
