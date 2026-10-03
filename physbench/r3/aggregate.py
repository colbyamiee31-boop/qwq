import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physbench/r3'))
from common import *
OUT=ROOT/'evidence/R3/FINAL'; OUT.mkdir(parents=True,exist_ok=True)

M1C=pd.read_csv(ROOT/'evidence/R3/M1/ctifl_event_responses.csv')
M2C=pd.read_csv(ROOT/'evidence/R3/M2/ctifl_event_responses.csv')
M1G=pd.read_csv(ROOT/'evidence/R3/M1/d2_action_grid.csv')
M2G=pd.read_csv(ROOT/'evidence/R3/M2/d2_action_grid.csv')
S1=json.loads((ROOT/'evidence/R3/M1/summary.json').read_text())
S2=json.loads((ROOT/'evidence/R3/M2/summary.json').read_text())
PIN=pd.read_csv(ROOT/'physbench/r3/generated/PRIMARY_97_R2_INPUT.csv')
SIN=pd.read_csv(ROOT/'physbench/r3/generated/STRICT_36_R2_INPUT.csv')

# inherited frozen native artifacts
NATIVE_R12=pd.read_csv(ROOT/'baseline_r12/evidence/R1_2/FINAL/event_level_validation.csv')
NATIVE_R12_MET=pd.read_csv(ROOT/'baseline_r12/evidence/R1_2/FINAL/validation_metrics.csv')
NATIVE_R2_MET=pd.read_csv(ROOT/'baseline_r2/evidence/R2/FINAL/coordinate_metrics.csv')
NATIVE_D_EVENT=pd.read_csv(ROOT/'baseline_exp31/evidence/FINAL/event_decision_consequence.csv')
NATIVE_D_SUM=pd.read_csv(ROOT/'baseline_exp31/evidence/FINAL/decision_consequence_summary.csv')

SEED=20261004; B=2000

def kappa(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if len(a)==0:return np.nan
    p0=np.mean(a==b); pa=np.mean(a==1); pb=np.mean(b==1); pe=pa*pb+(1-pa)*(1-pb)
    return np.nan if abs(1-pe)<1e-15 else float((p0-pe)/(1-pe))

def val_metric(f,var):
    ms=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(f[f'model_{var}_delta'],f[f'post_{var}'],f[f'pre_{var}'])])
    os=np.array([sign_obs(x) for x in f[f'obs_{var}_delta']])
    keep=(ms!=0)&(os!=0); ff=f.loc[keep].copy(); ms=ms[keep]; os=os[keep]
    conc=float(np.mean(ms==os)) if len(ms) else np.nan; kap=kappa(ms,os)
    den=np.abs(ff[f'{var}_gradient_C' if var=='T' else 'AH_gradient_g_m3'].to_numpy(float))*np.abs(ff.delta_u.to_numpy(float))
    good=den>1e-12
    rho=float(spearmanr(ff.loc[good,f'model_{var}_delta']/den[good],ff.loc[good,f'obs_{var}_delta']/den[good]).statistic) if good.sum()>=3 else np.nan
    return {'n_resolved':int(len(ms)),'concordance':conc,'kappa':kap,'spearman_norm':rho,'unresolved':int(len(f)-len(ms))}

def boot_val(f,var):
    dates=np.array(sorted(f.event_date.unique())); groups={d:f[f.event_date==d] for d in dates}; rng=np.random.default_rng(SEED); vals=[]
    for _ in range(B):
        ss=rng.choice(dates,size=len(dates),replace=True); q=pd.concat([groups[d] for d in ss],ignore_index=True); m=val_metric(q,var); vals.append([m['concordance'],m['kappa'],m['spearman_norm']])
    a=np.asarray(vals,float); out={}
    for j,n in enumerate(['concordance','kappa','spearman_norm']):
        v=a[:,j]; v=v[np.isfinite(v)]; out[n]=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] if len(v) else [np.nan,np.nan]
    return out

CUR=pd.concat([M1C,M2C],ignore_index=True)
# validation metrics H/HV
vrows=[]
for cohort in ['primary','strict']:
  for stage in STAGES:
    for model in ['M1','M2']:
      q=CUR[(CUR.cohort==cohort)&(CUR.stage==stage)&(CUR.model==model)]
      for h in ([15,30] if cohort=='primary' else [15]):
        f=q[q.horizon_min==h]
        for var in ['T','AH']:
          m=val_metric(f,var); ci=boot_val(f,var)
          vrows.append({'reference':'R3','cohort':cohort,'stage':stage,'model':model,'horizon_min':h,'variable':var,**m,'concordance_ci_low':ci['concordance'][0],'concordance_ci_high':ci['concordance'][1],'kappa_ci_low':ci['kappa'][0],'kappa_ci_high':ci['kappa'][1],'spearman_ci_low':ci['spearman_norm'][0],'spearman_ci_high':ci['spearman_norm'][1]})
VAL=pd.DataFrame(vrows)
VAL.to_csv(OUT/'01_harmonised_empirical_validation_metrics.csv',index=False,float_format='%.12g')
# native primary all metrics for direct table
nat=NATIVE_R12_MET[(NATIVE_R12_MET.cohort=='primary')&(NATIVE_R12_MET.group=='all')][['model','horizon_min','variable','concordance','concordance_ci_low','concordance_ci_high','kappa','kappa_ci_low','kappa_ci_high','spearman_norm','spearman_ci_low','spearman_ci_high']].copy(); nat['stage']='N'
cmp=pd.concat([nat,VAL[VAL.cohort=='primary'][nat.columns]],ignore_index=True)
cmp.to_csv(OUT/'02_native_H_HV_empirical_comparison.csv',index=False,float_format='%.12g')

# cross-model response divergence

def current_div(stage,cohort='primary',h=15):
    a=CUR[(CUR.stage==stage)&(CUR.cohort==cohort)&(CUR.horizon_min==h)&(CUR.model=='M1')]
    b=CUR[(CUR.stage==stage)&(CUR.cohort==cohort)&(CUR.horizon_min==h)&(CUR.model=='M2')]
    x=a[['event_id','event_date','model_T_delta','model_AH_delta','pre_T','post_T','pre_AH','post_AH']].merge(
      b[['event_id','model_T_delta','model_AH_delta','pre_T','post_T','pre_AH','post_AH']],on='event_id',suffixes=('_M1','_M2'),validate='one_to_one')
    x['stage']=stage
    return x

def native_div(cohort='primary',h=15):
    q=NATIVE_R12[(NATIVE_R12.cohort==cohort)&(NATIVE_R12.horizon_min==h)]
    a=q[q.model=='M1']; b=q[q.model=='M2']
    x=a[['event_id','event_date','model_T_delta','model_AH_delta','model_T_sign','model_AH_sign']].merge(
      b[['event_id','model_T_delta','model_AH_delta','model_T_sign','model_AH_sign']],on='event_id',suffixes=('_M1','_M2'),validate='one_to_one')
    x['stage']='N'; return x

def add_div(x,native=False):
    for var in ['T','AH']:
        a=x[f'model_{var}_delta_M1'].to_numpy(float); b=x[f'model_{var}_delta_M2'].to_numpy(float)
        eps=1e-12*np.maximum.reduce([np.ones(len(x)),np.abs(a),np.abs(b)])
        x[f'D_{var}']=np.abs(a-b)/(np.abs(a)+np.abs(b)+eps)
        if native:
            s1=x[f'model_{var}_sign_M1'].to_numpy(int); s2=x[f'model_{var}_sign_M2'].to_numpy(int)
        else:
            s1=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(x[f'model_{var}_delta_M1'],x[f'post_{var}_M1'],x[f'pre_{var}_M1'])])
            s2=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(x[f'model_{var}_delta_M2'],x[f'post_{var}_M2'],x[f'pre_{var}_M2'])])
        x[f'sign_M1_{var}']=s1; x[f'sign_M2_{var}']=s2
        x[f'sign_disagree_{var}']=(s1!=0)&(s2!=0)&(s1!=s2)
        x[f'both_resolved_{var}']=(s1!=0)&(s2!=0)
    x['D_resp']=0.5*(x.D_T+x.D_AH)
    return x

DIV=pd.concat([add_div(native_div(),True),add_div(current_div('H')),add_div(current_div('HV'))],ignore_index=True)
DIV.to_csv(OUT/'03_cross_model_response_divergence_event.csv',index=False,float_format='%.12g')
drows=[]
for stage in ['N','H','HV']:
    q=DIV[DIV.stage==stage]
    drows.append({'stage':stage,'n_events':len(q),'mean_D_resp':float(q.D_resp.mean()),'median_D_resp':float(q.D_resp.median()),'T_sign_disagreement_fraction':float(q.loc[q.both_resolved_T,'sign_disagree_T'].mean()),'AH_sign_disagreement_fraction':float(q.loc[q.both_resolved_AH,'sign_disagree_AH'].mean())})
DS=pd.DataFrame(drows)
# paired cluster-bootstrap stage minus native
n0=DIV[DIV.stage=='N'][['event_id','event_date','D_resp']].rename(columns={'D_resp':'D_N'})
diffrows=[]
for stage in ['H','HV']:
    q=DIV[DIV.stage==stage][['event_id','D_resp']].rename(columns={'D_resp':f'D_{stage}'}).merge(n0,on='event_id',validate='one_to_one')
    q['diff']=q[f'D_{stage}']-q.D_N
    dates=np.array(sorted(q.event_date.unique())); gd={d:q[q.event_date==d] for d in dates}; rng=np.random.default_rng(SEED); vals=[]
    for _ in range(B):
        ss=rng.choice(dates,size=len(dates),replace=True); z=pd.concat([gd[d] for d in ss],ignore_index=True); vals.append(float(z['diff'].mean()))
    diffrows.append({'stage':stage,'minus_native_mean_D_resp':float(q['diff'].mean()),'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5))})
DIF=pd.DataFrame(diffrows)
DS=DS.merge(DIF,left_on='stage',right_on='stage',how='left')
DS.to_csv(OUT/'04_cross_model_response_divergence_summary.csv',index=False,float_format='%.12g')

# R2 external anchor replay
def anchor_metric(q,coord,model):
    z=q if model=='pooled' else q[q.model==model]
    emp=z[f'emp_{coord}'].to_numpy(float); mod=z[f'delta_{coord}'].to_numpy(float); nz=(np.abs(emp)>1e-12)&(np.abs(mod)>1e-12); same=nz&(np.sign(emp)==np.sign(mod))
    return {'model':model,'coordinate':coord,'n_events':z.event_id.nunique(),'sign_concordance':float(np.mean(np.sign(emp[nz])==np.sign(mod[nz]))),'spearman':float(spearmanr(emp,mod).statistic),'median_abs_log_ratio':float(np.median(np.abs(np.log(np.abs(mod[same])/np.abs(emp[same]))))),'through_origin_slope':float(np.sum(emp*mod)/np.sum(emp*emp)),'normalized_abs_error':float(np.median(np.abs(mod-emp))/np.median(np.abs(emp)))}
arows=[]
for stage in STAGES:
    q=CUR[(CUR.cohort=='primary')&(CUR.horizon_min==15)&(CUR.stage==stage)].merge(PIN[['event_id','anchor_primary_3_6','emp_delta_DV_23_23','emp_delta_DN_23_23']],on='event_id',validate='many_to_one')
    q=q[q.anchor_primary_3_6.astype(bool)].copy(); q['emp_DV']=q.emp_delta_DV_23_23; q['emp_DN']=q.emp_delta_DN_23_23
    for coord in ['DV','DN']:
      for model in ['M1','M2','pooled']:
        a=anchor_metric(q,coord,model); a['stage']=stage; arows.append(a)
ANCH=pd.DataFrame(arows)
native_anchor=NATIVE_R2_MET[(NATIVE_R2_MET['case']=='primary_3_6_23_23')][['model','coordinate','n_events','sign_concordance','spearman','median_abs_log_ratio','through_origin_slope','normalized_abs_error']].copy(); native_anchor['stage']='N'
ANCHALL=pd.concat([native_anchor,ANCH],ignore_index=True)
ANCHALL.to_csv(OUT/'05_external_anchor_native_H_HV.csv',index=False,float_format='%.12g')

# common-height identity
height_err=float(np.max(np.abs(CUR.delta_DN.to_numpy(float)*H_TARGET-CUR.delta_DV.to_numpy(float))))

# D2 decision replay
COST=((ACTIONS-0.1)/0.8)**2
def benefit_table(df,eid,h):
    x=df[(df.event_id==eid)&(df.horizon_min==h)].sort_values('action')
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0]); T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt); bA=-np.sign(ga)*(AH-AH[0])/abs(ga); Bx=.5*(bT+bA)
    return Bx,x
def bps(Bx,lmax):
    v=[0.,float(lmax)]
    for i in range(len(ACTIONS)):
      for j in range(i+1,len(ACTIONS)):
        dc=COST[i]-COST[j]
        if abs(dc)>1e-15:
          z=(Bx[i]-Bx[j])/dc
          if 0<z<lmax:v.append(float(z))
    return sorted(set(round(x,14) for x in v))
def choose(Bx,l):
    q=Bx-l*COST; mx=q.max(); return int(np.where(q>=mx-1e-12)[0][0])
def decision_stage(stage,h=15,lmax=1.0):
    a=M1G[(M1G.stage==stage)&(M1G.horizon_min==h)]; b=M2G[(M2G.stage==stage)&(M2G.horizon_min==h)]
    rows=[]
    for eid in sorted(a.event_id.unique()):
      B1,x1=benefit_table(a,eid,h); B2,x2=benefit_table(b,eid,h); bp=sorted(set(bps(B1,lmax)+bps(B2,lmax))); dis=0.; gap=0.
      for lo,hi in zip(bp[:-1],bp[1:]):
        if hi<=lo:continue
        mid=(lo+hi)/2; i1=choose(B1,mid); i2=choose(B2,mid); w=hi-lo
        if i1!=i2:dis+=w
        gap+=w*abs(ACTIONS[i1]-ACTIONS[i2])/0.8
      rows.append({'stage':stage,'event_id':eid,'event_date':x1.event_date.iloc[0],'horizon_min':h,'lambda_max':lmax,'decision_disagreement_measure':dis/lmax,'normalized_action_gap_integral':gap/lmax,'any_disagreement':bool(dis>1e-12)})
    return pd.DataFrame(rows)
DE=pd.concat([decision_stage(s,h,lm) for s in STAGES for h in [15,30] for lm in [1.0,3.0]],ignore_index=True)
DE.to_csv(OUT/'06_d2_harmonised_decision_event_metrics.csv',index=False,float_format='%.12g')
dsummary=[]
for stage in STAGES:
  for h in [15,30]:
    for lm in [1.0,3.0]:
      q=DE[(DE.stage==stage)&(DE.horizon_min==h)&(DE.lambda_max==lm)]
      dates=np.array(sorted(q.event_date.unique())); gd={d:q[q.event_date==d] for d in dates}; rng=np.random.default_rng(SEED); vals=[]
      for _ in range(B):
        ss=rng.choice(dates,size=len(dates),replace=True); z=pd.concat([gd[d] for d in ss],ignore_index=True); vals.append(float(z.decision_disagreement_measure.mean()))
      dsummary.append({'stage':stage,'horizon_min':h,'lambda_max':lm,'n_events':len(q),'mean_decision_disagreement':float(q.decision_disagreement_measure.mean()),'median_decision_disagreement':float(q.decision_disagreement_measure.median()),'fraction_any_disagreement':float(q.any_disagreement.mean()),'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5))})
DSUM=pd.DataFrame(dsummary)
# native primary row
nr=NATIVE_D_SUM[(NATIVE_D_SUM.horizon_min==15)&(NATIVE_D_SUM.lambda_max==1.0)&(NATIVE_D_SUM.stratum=='all')&(NATIVE_D_SUM.metric=='decision_disagreement_measure')].iloc[0]
native_dec={'stage':'N','horizon_min':15,'lambda_max':1.0,'n_events':int(nr.n_events),'mean_decision_disagreement':float(nr['mean']),'median_decision_disagreement':float(nr['median']),'fraction_any_disagreement':float(nr.fraction_any_disagreement),'ci_low':float(nr.ci_low),'ci_high':float(nr.ci_high)}
DSUMALL=pd.concat([pd.DataFrame([native_dec]),DSUM],ignore_index=True)
DSUMALL.to_csv(OUT/'07_d2_decision_native_H_HV_summary.csv',index=False,float_format='%.12g')

# locked events bridge
locked=[27,86,89]
bridge=NATIVE_D_EVENT[(NATIVE_D_EVENT.event_id.isin(locked))&(NATIVE_D_EVENT.horizon_min==15)&(NATIVE_D_EVENT.lambda_max==1.0)][['event_id','decision_disagreement_measure','normalized_action_gap_integral','any_disagreement']].copy(); bridge['stage']='N'
for stage in STAGES:
    q=DE[(DE.stage==stage)&(DE.horizon_min==15)&(DE.lambda_max==1.0)&(DE.event_id.isin(locked))][['event_id','decision_disagreement_measure','normalized_action_gap_integral','any_disagreement']].copy()
    q['stage']=stage
    bridge=pd.concat([bridge,q],ignore_index=True)
bridge.to_csv(OUT/'08_locked_events_27_86_89_bridge.csv',index=False,float_format='%.12g')

# M2 numerical sensitivity: final 10-s H versus complete finite 30-s H from first run
OLDM2C=pd.read_csv(ROOT/'baseline_m2_h30/evidence/R3/M2/ctifl_event_responses.csv')
OLDM2G=pd.read_csv(ROOT/'baseline_m2_h30/evidence/R3/M2/d2_action_grid.csv')
solver_rows=[]
old=OLDM2C[OLDM2C.stage=='H']; new=M2C[M2C.stage=='H']
z=old.merge(new,on=['cohort','event_id','horizon_min'],suffixes=('_30s','_10s'),validate='one_to_one')
for col in ['model_T_delta','model_AH_delta','delta_DV','delta_DN']:
    d=np.abs(z[f'{col}_10s'].to_numpy(float)-z[f'{col}_30s'].to_numpy(float))
    solver_rows.append({'track':'CTIFL_H','quantity':col,'n':len(d),'max_abs_difference':float(np.max(d)),'median_abs_difference':float(np.median(d))})
old=OLDM2G[OLDM2G.stage=='H']; new=M2G[M2G.stage=='H']
z=old.merge(new,on=['event_id','horizon_min','action'],suffixes=('_30s','_10s'),validate='one_to_one')
for col in ['T','AH','DV','DN']:
    d=np.abs(z[f'{col}_10s'].to_numpy(float)-z[f'{col}_30s'].to_numpy(float))
    solver_rows.append({'track':'D2_H','quantity':col,'n':len(d),'max_abs_difference':float(np.max(d)),'median_abs_difference':float(np.median(d))})
SOLVER=pd.DataFrame(solver_rows)
SOLVER.to_csv(OUT/'09_M2_solver_10s_vs_30s_H.csv',index=False,float_format='%.12g')

# machine summary and hard runtime gates
primary_div=DS.set_index('stage').to_dict(orient='index')
primary_dec=DSUMALL[(DSUMALL.horizon_min==15)&(DSUMALL.lambda_max==1.0)].set_index('stage').to_dict(orient='index')
summary={
 'experiment':'PhysBench-GH R3 common physical envelope / geometry harmonisation',
 'M1_gate':S1['gate_pass'],'M2_gate':S2['gate_pass'],
 'common_height_target_m':H_TARGET,'projected_aperture_ratio_target':VENT_PROJECTED_RATIO,
 'common_height_identity_max_abs_error':height_err,
 'primary_response_divergence':primary_div,
 'primary_decision_bridge':primary_dec,
 'anchor_primary_events':40,
 'scientific_improvement_not_runtime_gate':True,'M2_solver_sensitivity':SOLVER.to_dict(orient='records')
}
summary['gate_pass']=bool(S1['gate_pass'] and S2['gate_pass'] and height_err<=1e-9 and len(DIV[DIV.stage=='HV'])==97 and len(DE[(DE.stage=='HV')&(DE.horizon_min==15)&(DE.lambda_max==1.0)])==61)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if not summary['gate_pass']: raise SystemExit('R3 aggregate integrity gate failed')
