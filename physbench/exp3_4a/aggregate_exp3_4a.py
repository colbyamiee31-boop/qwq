import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
M1=ROOT/'evidence/EXP3_4A_M1'
M2=ROOT/'evidence/EXP3_4A_M2'
OUT=ROOT/'evidence/EXP3_4A_FINAL'
OUT.mkdir(parents=True,exist_ok=True)

U9=np.round(np.arange(0.1,1.0,0.1),1)
Q=np.array([0.0,0.25,0.5,0.75,1.0],dtype=float)
COORDS={
 'specific_volume_dose_m3_m2':'DV',
 'air_volume_equiv':'DN'
}
MONO_RTOL=1e-10

m1=pd.read_csv(M1/'physical_dose_grid.csv')
m2=pd.read_csv(M2/'physical_dose_grid.csv')
s1=json.loads((M1/'summary.json').read_text())
s2=json.loads((M2/'summary.json').read_text())
assert s1['gate_pass'] and s2['gate_pass']
assert s1['input_sha256']==s2['input_sha256']=='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'

REQ={'model_id','event_id','team','event_date','daynight','horizon_min','action',
     'specific_volume_dose_m3_m2','air_volume_equiv','average_specific_flux_m3_m2_s','average_ach_h_1','T','AH'}
for name,df in [('M1',m1),('M2',m2)]:
    miss=REQ-set(df.columns); assert not miss,(name,miss)
    assert len(df)==61*2*9
    assert not df.duplicated(['event_id','horizon_min','action']).any()
    assert set(np.round(df.action.unique(),10))==set(U9)
    assert set(df.horizon_min.astype(int))=={15,30}
    assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()

keys=['event_id','horizon_min','action']
z=m1.merge(m2,on=keys,suffixes=('_M1','_M2'),validate='one_to_one')
assert len(z)==61*2*9

# Same-command distortion table.
dist=[]
for _,r in z.iterrows():
    row={
      'event_id':int(r.event_id),'horizon_min':int(r.horizon_min),'action':float(r.action),
      'team':r.team_M1,'event_date':r.event_date_M1,'daynight':r.daynight_M1
    }
    for col,label in COORDS.items():
        d1=float(r[f'{col}_M1']); d2=float(r[f'{col}_M2'])
        row[f'{label}_M1']=d1; row[f'{label}_M2']=d2
        row[f'{label}_M1_minus_M2']=d1-d2
        row[f'{label}_abs_difference']=abs(d1-d2)
        row[f'{label}_M1_over_M2']=np.nan if abs(d2)<=1e-15 else d1/d2
        scale=max(abs(d1),abs(d2),1e-15)
        row[f'{label}_relative_difference_over_maxabs']=abs(d1-d2)/scale
    row['average_ach_M1']=float(r.average_ach_h_1_M1)
    row['average_ach_M2']=float(r.average_ach_h_1_M2)
    dist.append(row)
dist=pd.DataFrame(dist)
dist.to_csv(OUT/'same_command_physical_dose_distortion.csv',index=False,float_format='%.12g')

cell_rows=[]; targets=[]; reversal_rows=[]
for eid in sorted(set(m1.event_id.astype(int))):
  for h in (15,30):
    a=m1[(m1.event_id==eid)&(m1.horizon_min==h)].sort_values('action')
    b=m2[(m2.event_id==eid)&(m2.horizon_min==h)].sort_values('action')
    assert len(a)==len(b)==9
    assert np.allclose(a.action.to_numpy(float),U9,rtol=0,atol=1e-12)
    assert np.allclose(b.action.to_numpy(float),U9,rtol=0,atol=1e-12)
    for col,label in COORDS.items():
        x=a[col].to_numpy(float); y=b[col].to_numpy(float)
        tx=MONO_RTOL*max(1.0,float(np.max(np.abs(x))))
        ty=MONO_RTOL*max(1.0,float(np.max(np.abs(y))))
        dx=np.diff(x); dy=np.diff(y)
        mono1=bool(np.all(dx>=-tx)); mono2=bool(np.all(dy>=-ty))
        for model,dv,tol in [('M1',dx,tx),('M2',dy,ty)]:
            for k,v in enumerate(dv):
                if v < -tol:
                    reversal_rows.append({
                      'event_id':eid,'horizon_min':h,'coordinate':label,'model_id':model,
                      'u_lo':float(U9[k]),'u_hi':float(U9[k+1]),'dose_increment':float(v),'tolerance':float(tol)
                    })
        lo=max(float(x[0]),float(y[0]))
        hi=min(float(x[-1]),float(y[-1]))
        width=float(hi-lo)
        positive=bool(width>0)
        span1=float(x[-1]-x[0]); span2=float(y[-1]-y[0])
        eligible=bool(mono1 and mono2 and positive)
        lower_src='M1' if x[0]>=y[0] else 'M2'
        upper_src='M1' if x[-1]<=y[-1] else 'M2'
        cell={
          'event_id':eid,'team':a.team.iloc[0],'event_date':a.event_date.iloc[0],'daynight':a.daynight.iloc[0],
          'horizon_min':h,'coordinate':label,
          'M1_monotone':mono1,'M2_monotone':mono2,
          'M1_min_adjacent_increment':float(dx.min()),'M2_min_adjacent_increment':float(dy.min()),
          'M1_low':float(x[0]),'M1_high':float(x[-1]),'M1_span':span1,
          'M2_low':float(y[0]),'M2_high':float(y[-1]),'M2_span':span2,
          'common_low':lo,'common_high':hi,'common_width':width,
          'positive_overlap':positive,'eligible_for_exp3_4b':eligible,
          'overlap_fraction_M1_span':float(width/span1) if positive and span1>0 else 0.0,
          'overlap_fraction_M2_span':float(width/span2) if positive and span2>0 else 0.0,
          'lower_bound_determined_by':lower_src,'upper_bound_determined_by':upper_src
        }
        cell_rows.append(cell)
        for q in Q:
            targets.append({
              'event_id':eid,'team':a.team.iloc[0],'event_date':a.event_date.iloc[0],'daynight':a.daynight.iloc[0],
              'horizon_min':h,'coordinate':label,'q':float(q),
              'eligible_for_exp3_4b':eligible,
              'common_low':lo,'common_high':hi,'common_width':width,
              'target_dose':float(lo+q*width) if eligible else np.nan
            })

cells=pd.DataFrame(cell_rows)
target_df=pd.DataFrame(targets)
reversals=pd.DataFrame(reversal_rows,columns=['event_id','horizon_min','coordinate','model_id','u_lo','u_hi','dose_increment','tolerance'])
cells.to_csv(OUT/'physical_overlap_cells.csv',index=False,float_format='%.12g')
target_df.to_csv(OUT/'common_physical_dose_targets_for_EXP3_4B.csv',index=False,float_format='%.12g')
reversals.to_csv(OUT/'monotonicity_reversals.csv',index=False,float_format='%.12g')

# Aggregate summaries.
summary_rows=[]
for coord in ('DV','DN'):
  for h in (15,30):
    x=cells[(cells.coordinate==coord)&(cells.horizon_min==h)]
    e=x[x.eligible_for_exp3_4b]
    summary_rows.append({
      'coordinate':coord,'horizon_min':h,'n_events':int(len(x)),
      'M1_monotone_count':int(x.M1_monotone.sum()),
      'M2_monotone_count':int(x.M2_monotone.sum()),
      'positive_overlap_count':int(x.positive_overlap.sum()),
      'eligible_count':int(x.eligible_for_exp3_4b.sum()),
      'common_width_min':None if len(e)==0 else float(e.common_width.min()),
      'common_width_median':None if len(e)==0 else float(e.common_width.median()),
      'common_width_max':None if len(e)==0 else float(e.common_width.max()),
      'overlap_fraction_M1_median':None if len(e)==0 else float(e.overlap_fraction_M1_span.median()),
      'overlap_fraction_M2_median':None if len(e)==0 else float(e.overlap_fraction_M2_span.median())
    })
summary_df=pd.DataFrame(summary_rows)
summary_df.to_csv(OUT/'overlap_summary.csv',index=False,float_format='%.12g')

ratio_rows=[]
for coord in ('DV','DN'):
  for h in (15,30):
    for u in U9:
      col=f'{coord}_M1_over_M2'
      x=dist[(dist.horizon_min==h)&np.isclose(dist.action,u)][col].dropna().astype(float)
      ratio_rows.append({
        'coordinate':coord,'horizon_min':h,'action':float(u),'n_events':int(len(x)),
        'M1_over_M2_min':float(x.min()),'M1_over_M2_q25':float(x.quantile(.25)),
        'M1_over_M2_median':float(x.median()),'M1_over_M2_q75':float(x.quantile(.75)),
        'M1_over_M2_max':float(x.max())
      })
ratio_df=pd.DataFrame(ratio_rows)
ratio_df.to_csv(OUT/'same_command_dose_ratio_summary.csv',index=False,float_format='%.12g')

geometry={
 'M1':{
   'air_volume_per_floor_area_m':s1['air_volume_per_floor_area_m'],
   'max_aperture_per_floor_area_m2_m2':s1['max_roof_aperture_per_floor_area_m2_m2'],
   'definition':s1['physical_flux_definition']
 },
 'M2':{
   'air_volume_per_floor_area_m':s2['air_volume_per_floor_area_m'],
   'max_aperture_per_floor_area_m2_m2':s2['max_top_aperture_per_floor_area_m2_m2'],
   'definition':s2['physical_flux_definition']
 }
}
geometry['max_aperture_per_floor_area_ratio_M1_over_M2']=float(
    geometry['M1']['max_aperture_per_floor_area_m2_m2']/geometry['M2']['max_aperture_per_floor_area_m2_m2']
)

result={
 'experiment':'PhysBench-GH EXP3.4A — Physical Ventilation-Dose Equivalence',
 'status':'PASS_EXP3_4A',
 'events':61,'diagnostic_actions':U9.tolist(),'future_common_q_levels':Q.tolist(),
 'primary_coordinate':'DV = cumulative external specific ventilation volume [m3 m-2]',
 'secondary_coordinate':'DN = cumulative air-volume equivalents [-]',
 'M1_summary':s1,'M2_summary':s2,
 'geometry':geometry,
 'overlap_summary':summary_df.to_dict(orient='records'),
 'monotonicity_reversal_count':int(len(reversals)),
 'exp3_4b_rule':(
   'EXP3.4B must use the predeclared q={0,.25,.5,.75,1} targets from this artifact. '
   'No extrapolation outside each event/horizon/model native u=[0.1,0.9] interval is allowed.'
 )
}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
