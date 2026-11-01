import runpy,pathlib,numpy as np,pandas as pd
root=pathlib.Path(r'E:\Users\PC\Documents\ChatGPT\1019，PHYBENCH');p=root/'B2_A1_main_review_20261101/CF_A1_aggregate_cf.py';m=runpy.run_path(str(p),run_name='synthetic_test');g=m['compute_summary'].__globals__;g['BOOT']=2;rows=[]
for condition in g['CONDITIONS']:
 for h in g['HORIZONS']:
  for i in range(97):
   f=1.0+(g['CONDITIONS'].index(condition)*.01);dt=(-.1-.001*i)*f;ah=(-.03-.0002*i)*f
   rows.append(dict(condition=condition,horizon_min=h,event_id=f's{i:03}',event_date=f'2020-01-{i%5+1:02}',anchor_primary_3_6=i<40,delta_u=.5,T_gradient_C=2.,AH_gradient_g_m3=1.,model_delta_T=dt,model_delta_AH=ah,obs_delta_T=dt*.9,obs_delta_AH=ah*.9,pre_T=20.,post_T=20+dt,pre_AH=8.,post_AH=8+ah,q_pre_top=.001,q_post_top=.002,q_pre_main=0.,q_post_main=0.))
df=pd.DataFrame(rows);g['compute_summary'](df);g['compute_paired'](df)
assert not g['FAILURES'];assert len(g['SUMMARY_ROWS'])==56 and len(g['PAIRED_ROWS'])==112
print('PASS: synthetic 97/40 condition aggregation produced 56 endpoint rows and 112 paired rows, using exact R5 functions with 2 bootstrap replicates for testing only; no GreenLight invoked.')
