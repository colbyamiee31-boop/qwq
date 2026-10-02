import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1=ROOT/'evidence/EXP0_6A_M1'
M2=ROOT/'evidence/EXP0_6A_M2'
OUT=ROOT/'evidence/EXP0_6A_FINAL'; OUT.mkdir(parents=True,exist_ok=True)

s1=json.loads((M1/'summary.json').read_text())
s2=json.loads((M2/'summary.json').read_text())
a=pd.read_csv(M1/'history_summary.csv')
b=pd.read_csv(M2/'history_summary.csv')

assert s1['gate_pass'] and s2['gate_pass']
assert len(a)==18 and len(b)==18
assert set(a.history_id)==set(b.history_id)
assert (a.history_derived.sum()==17) and (b.history_derived.sum()==17)

ha=a[a.history_derived.astype(bool)].copy()
hb=b[b.history_derived.astype(bool)].copy()
m=ha.merge(hb,on='history_id',suffixes=('_M1','_M2'),validate='one_to_one')

def root_stats(df):
    u=df[df.root_count==1].primary_boundary_ppm.dropna().astype(float)
    return {
      'n_history_derived':int(len(df)),
      'n_unique_root':int(len(u)),
      'unique_root_fraction':float(len(u)/len(df)),
      'min_ppm':None if len(u)==0 else float(u.min()),
      'q25_ppm':None if len(u)==0 else float(u.quantile(.25)),
      'median_ppm':None if len(u)==0 else float(u.median()),
      'q75_ppm':None if len(u)==0 else float(u.quantile(.75)),
      'max_ppm':None if len(u)==0 else float(u.max()),
      'span_ppm':None if len(u)==0 else float(u.max()-u.min())
    }

st1=root_stats(ha); st2=root_stats(hb)
paired=m[(m.root_count_M1==1)&(m.root_count_M2==1)&m.primary_boundary_ppm_M1.notna()&m.primary_boundary_ppm_M2.notna()].copy()
paired['signed_M1_minus_M2_ppm']=paired.primary_boundary_ppm_M1-paired.primary_boundary_ppm_M2
paired['absolute_gap_ppm']=paired.signed_M1_minus_M2_ppm.abs()
paired.to_csv(OUT/'matched_history_root_gaps.csv',index=False,float_format='%.12g')

if st1['n_unique_root'] and st2['n_unique_root']:
    lo1,hi1=st1['min_ppm'],st1['max_ppm']; lo2,hi2=st2['min_ppm'],st2['max_ppm']
    overlap=max(0.0,min(hi1,hi2)-max(lo1,lo2))
    if hi1 < lo2: sep=lo2-hi1
    elif hi2 < lo1: sep=lo1-hi2
    else: sep=0.0
else:
    overlap=None; sep=None

gap_median=None if len(paired)==0 else float(paired.absolute_gap_ppm.median())
gap_min=None if len(paired)==0 else float(paired.absolute_gap_ppm.min())
gap_max=None if len(paired)==0 else float(paired.absolute_gap_ppm.max())
signed_median=None if len(paired)==0 else float(paired.signed_M1_minus_M2_ppm.median())

def ratio(gap,span):
    if gap is None or span is None: return None
    if span<=0: return None
    return float(gap/span)

# H900 500-ppm response sign/magnitude stability.
resp={}
for model,df in [('M1',ha),('M2',hb)]:
    q={}
    for col in ['delta_T_C_at_500ppm','delta_AH_g_m3_at_500ppm','delta_CO2_ppm_at_500ppm']:
        v=df[col].to_numpy(float)
        q[col]={
          'min':float(np.min(v)),'median':float(np.median(v)),'max':float(np.max(v)),
          'all_positive':bool(np.all(v>0)),'all_negative':bool(np.all(v<0)),
          'signs':sorted(set(np.sign(v).astype(int).tolist()))
        }
    resp[model]=q

# Projection-correction transparency.
projection={}
for model,df in [('M1',ha),('M2',hb)]:
    projection[model]={}
    for col in ['projection_T_C','projection_VP_Pa','projection_CO2_ppm_at_500','projection_canopy_T_C']:
        v=np.abs(df[col].to_numpy(float))
        projection[model][col]={'median_abs':float(np.median(v)),'max_abs':float(np.max(v))}

# Root topology table.
topology=m[[
    'history_id','root_count_M1','primary_boundary_ppm_M1',
    'root_count_M2','primary_boundary_ppm_M2'
]].copy()
topology.to_csv(OUT/'root_topology_by_history.csv',index=False,float_format='%.12g')

result={
 'experiment':'PhysBench-GH EXP0.6A Latent-State Ensemble + H900 CO2 Boundary',
 'status':'PASS_EXP0_6A' if s1['gate_pass'] and s2['gate_pass'] else 'FAIL',
 'protocol_branch':'exp0-6-latent-state-robustness',
 'history_derived_members':17,
 'reference_condition':'HNR',
 'M1':{'runtime':s1,'root_distribution':st1},
 'M2':{'runtime':s2,'root_distribution':st2},
 'cross_model':{
   'paired_unique_root_histories':int(len(paired)),
   'matched_history_signed_M1_minus_M2_median_ppm':signed_median,
   'matched_history_absolute_gap_min_ppm':gap_min,
   'matched_history_absolute_gap_median_ppm':gap_median,
   'matched_history_absolute_gap_max_ppm':gap_max,
   'root_envelope_overlap_ppm':overlap,
   'ordered_separation_margin_ppm':sep,
   'median_gap_over_M1_latent_span':ratio(gap_median,st1['span_ppm']),
   'median_gap_over_M2_latent_span':ratio(gap_median,st2['span_ppm'])
 },
 'H900_response_at_500ppm':resp,
 'projection_correction_audit':projection,
 'interpretation_guard':(
   'Scientific movement, envelope overlap, root disappearance, or topology changes are results, not gate failures. '
   'EXP0.6A addresses sensitivity to history-derived native latent states only; equal physical airflow is not assumed.'
 )
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

# compact scientific summary table
rows=[
 {'quantity':'M1 history-derived unique-root count','value':st1['n_unique_root']},
 {'quantity':'M1 latent root span ppm','value':st1['span_ppm']},
 {'quantity':'M2 history-derived unique-root count','value':st2['n_unique_root']},
 {'quantity':'M2 latent root span ppm','value':st2['span_ppm']},
 {'quantity':'paired histories with unique roots','value':len(paired)},
 {'quantity':'matched-history absolute root gap median ppm','value':gap_median},
 {'quantity':'root-envelope overlap ppm','value':overlap},
 {'quantity':'ordered separation margin ppm','value':sep},
 {'quantity':'median gap / M1 latent span','value':ratio(gap_median,st1['span_ppm'])},
 {'quantity':'median gap / M2 latent span','value':ratio(gap_median,st2['span_ppm'])},
]
pd.DataFrame(rows).to_csv(OUT/'scientific_summary.csv',index=False)

print(json.dumps(result,indent=2))
if result['status']!='PASS_EXP0_6A':
    raise SystemExit('EXP0.6A aggregate gate failed')
